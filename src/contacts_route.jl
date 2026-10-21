# ── Models that name each case's contacts, on the race ────────────────
#
# A model written against `contacts_of` names, for each case, the people it
# exposes and when: people who already exist, as on a network, or new ones it
# creates with `make_contact!`. On the race that is a route which asks the model
# once, when the case settles, and proposes each exposure at the time the model
# gave. Someone exposed by several cases is decided once, by the earliest
# exposure that transmits.
#
# An exposure is one contact at one time. A blocked one is lost, as in a
# branching process: there is no pair kernel to draw a later contact from. The
# model's times are already drawn, so the built-in risks are those of a single
# exposure, as in a branching process.

# The race's members when a model creates people as it goes: everyone in the
# state, numbered by id: a member's position is its id, and the population
# grows by appending.
mutable struct _Grown <: AbstractVector{Int}
    n::Int
end
Base.size(g::_Grown) = (g.n,)
Base.@propagate_inbounds function Base.getindex(g::_Grown, k::Int)
    @boundscheck checkbounds(g, k)
    return k
end
_member_positions(g::_Grown) = g
# A person the infector created is listed as its contact when created; only
# a blocked exposure of someone else is offered to the recorder.
_logs_blocked(::_Grown, source, ind) = !_created_by(ind, source)

# Add the people created since the race last grew, each as an unsettled member
# offered what a new person is offered.
function _grow!(g::_Grown, race)
    (; state) = race
    while g.n < length(state.individuals)
        g.n += 1
        push!(race.best, oftype(first(race.best), Inf))
        push!(race.processed, false)
        push!(race.represents, 0)
        race.may_block && push!(race.head, 0)
        isempty(race.interventions) && continue
        contact = state.individuals[g.n]
        for intervention in race.interventions
            initialise_individual!(intervention, contact, state)
        end
        _apply_population_actions!(state, race.interventions, (contact.id,))
        _apply_creation_actions!(state, race.creating, contact)
    end
    return nothing
end

# The route a `contacts_of` model transmits along. `pending` holds the settling
# case's exposures until the race proposes them, `reached` who each case
# exposed, which is what tracing acts on, and `early` the exposures a ring asked
# for before a person settled, as delays from their infection.
struct _ContactsOf{M, T <: Real}
    model::M
    pending::Vector{Tuple{Int, T}}
    reached::Dict{Int, Vector{Int}}
    early::Dict{Int, Vector{Tuple{Int, T}}}
end
_ContactsOf(model, ::Type{T}) where {T} = _ContactsOf(
    model, Tuple{Int, T}[], Dict{Int, Vector{Int}}(), Dict{Int, Vector{Tuple{Int, T}}}()
)

# The exposures exist only as the model names them: the route names no
# targets for a walker and no pair kernel to redraw from.
(::_ContactsOf)(infector_id, state) = ()
_pair_kernel(::_ContactsOf, infector_id, target_id, state) = nothing

# The model times each exposure from the case's infection, and an intervention
# that removes the case ends its exposures.
_contacts_window() = RouteWindow(
    :transmission; from = :infection, until = (INTERVENTION_REMOVAL,), kernel = nothing
)
_route_open(::_ContactsOf, ind, w::RouteWindow) = ind.infection_time
_route_may_block(::_ContactsOf) = true
_route_builtin_blocks(::_ContactsOf) = _offspring_builtin_risk_blocks

# A contact the model created was listed on its infector by `make_contact!`.
function _record_infection!(::_ContactsOf, infector, ind, w, named)
    ind.id in infector.secondary_case_ids || push!(infector.secondary_case_ids, ind.id)
    return nothing
end

# Ask the model whom the settling case exposes, or take what a ring already
# asked for, timed from the infection now known. A case at the generation bound
# transmits to nobody.
function _mint!(c::_ContactsOf, race, ind, max_generation)
    empty!(c.pending)
    early = isempty(c.early) ? nothing : pop!(c.early, ind.id, nothing)
    ind.generation >= max_generation && return nothing
    (; state) = race
    if early === nothing
        for (target, t) in contacts_of(c.model, ind, state)
            push!(c.pending, (target.id, t))
        end
        _grow!(race.members, race)
        c.reached[ind.id] = Int[id for (id, _) in c.pending]
    else
        for (id, delay) in early
            t = ind.infection_time + delay
            _created_by(state.individuals[id], ind) && (state.individuals[id].infection_time = t)
            push!(c.pending, (id, t))
        end
    end
    return nothing
end

# Whether `case`'s `contacts_of` created `contact`, rather than naming someone
# who already existed.
_created_by(contact, case) = contact.parent_id == case.id && !is_infected(contact)

# What tracing reads as a case's contacts: the people the model named when the
# case settled. A ring that grows through someone not yet settled needs their
# contacts too. A model whose `contacts_of` reads nothing settling writes says
# so with `reads_settled_state`, and is then asked about that person when the
# ring reaches them, as though infected at time zero. Its answer is kept as
# delays from the infection, and they are the exposures the person makes if
# they settle, as a branching process keeps the contacts it drew early. Someone
# the model created for the person has no infection time until then. Otherwise
# a ring grows past a person only once they settle, as it does for an offspring
# given as a function.
function _case_contacts(
        contacts, rts::Tuple{Tuple{RouteWindow, _ContactsOf}}, race, max_generation
    )
    c = only(rts)[2]
    reads_settled_state(c.model) && return contacts
    return function (id, state)
        named = get(c.reached, id, nothing)
        named === nothing || return named
        race.processed[id] && return ()
        ind = state.individuals[id]
        # A person at the generation bound exposes nobody; the model is not
        # asked to create anyone for them.
        ind.generation >= max_generation && return ()
        infection_time = ind.infection_time
        ind.infection_time = zero(infection_time)
        exposures = [(target.id, t) for (target, t) in contacts_of(c.model, ind, state)]
        ind.infection_time = infection_time
        _grow!(race.members, race)
        # Only `ind` can infect someone the model created for it.
        for (cid, _) in exposures
            contact = state.individuals[cid]
            _created_by(contact, ind) || continue
            contact.infection_time = oftype(contact.infection_time, NaN)
            contact.generation = ind.generation + 1
        end
        c.early[id] = exposures
        named = c.reached[id] = Int[cid for (cid, _) in exposures]
        return named
    end
end

# Propose the case's exposures that fall inside its window. A time before the
# case's own infection is no exposure, as in a branching process.
function _propose_along!(
        c::_ContactsOf, race, infector_id, opening_id, open_t, close_t, traits, live, ri
    )
    for (id, t) in c.pending
        open_t <= t <= close_t || continue
        _propose_contact!(race, id, infector_id, opening_id, t)
    end
    empty!(c.pending)
    return nothing
end

# A person the model created but the run never infected stays a contact of its
# case. An exposure after the run ended is not part of the outbreak it
# describes, so such a person is left with no infection time, as a branching
# process leaves its undecided contacts.
function _finish!(c::_ContactsOf, race, stop_t)
    for ind in race.state.individuals
        !is_infected(ind) && ind.infection_time > stop_t || continue
        ind.infection_time = oftype(ind.infection_time, NaN)
    end
    isempty(race.interventions) && return nothing
    for ind in race.state.individuals
        _drop_stale_abort!(ind)
    end
    return nothing
end

# Run a model written against `contacts_of` on the race, over everyone its
# `initialise_state` creates and anyone its `contacts_of` creates later.
function _simulate_contacts_of(
        model::TransmissionModel, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    interventions = isempty(interventions) ? () : interventions
    state = initialise_state(model, sim_opts, interventions, progression, attributes, rng)
    # The race counts the index cases as it settles them.
    state.cumulative_cases = 0
    T = _timetype(state)
    route = _ContactsOf(model, T)
    seeds = [is_infected(ind) for ind in state.individuals]
    sellke_race!(
        state, _Grown(length(state.individuals)), rng, sim_opts;
        seed! = (best, _, _) -> (best[seeds] .= zero(T)),
        routes = ((_contacts_window(), route),), interventions,
        contacts = (id, state) -> get(route.reached, id, ()),
        risks = transmission_risks(model), recorder
    )
    apply_observation!(observation, state, rng)
    return state
end
