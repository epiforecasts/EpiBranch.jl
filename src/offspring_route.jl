# ── Branching processes on the race ──────────────────────────────────
#
# A branching process names no population in advance: each case creates its
# own contacts, as many as its offspring distribution draws, each timed from
# the case's window. On the race that is a route which creates the case's
# contacts when the case settles (`_mint!`) and proposes each one once. The
# race then decides them in time order with the rest, so a contact infected
# earlier than another case's contact is settled first whichever generation
# either belongs to, and the population grows as the race runs.
#
# A contact is one exposure, not a pair that goes on meeting, so a blocked
# contact is lost: the route offers no pair kernel to draw a later contact
# from. Its built-in risks are those of a single exposure. The contact's
# susceptibility and the infector's infectiousness each block it with one
# minus their value, and a finite population blocks it with the share already
# infected.

# Who drew the contact at a place, on which window, of which type (0 in a
# single-type process), and when it falls: read together when the place
# becomes a person, so they are kept together.
struct _Drawn{T <: Real}
    infector::Int
    route::Int
    type::Int
    time::T
end

# The race's members for a branching process: a place for every contact drawn.
# A place becomes a person only when something needs one: when the race comes
# to decide the contact's infection, or earlier when tracing or an
# intervention's actions reach it. Creating every contact as its infector
# settles would leave each one to be read again long after, in time order, once
# it is far out of cache. Places are in the order contacts were drawn, and the
# index cases hold the first ones.
mutable struct _Tree{T <: Real} <: AbstractVector{Int}
    ids::Vector{Int}           # the person at each place, 0 until there is one
    places::Vector{Int}        # each person's place
    drawn::Vector{_Drawn{T}}   # temporary records for unresolved contacts
    slots::Vector{Int}        # draw slot: negative when queued, zero when resolved
    free::Int                # head of the recycled draw records
    untils::Vector{Tuple}      # each window's `until` states
    tagged::Vector{Bool}       # whether that window has any
end

_Tree{T}() where {T} = _Tree{T}(Int[], Int[], _Drawn{T}[], Int[], 0, Tuple[], Bool[])

# A tree with the index cases at the first places, reusing one this task has
# grown before, for the reason `_RaceBuffers` gives.
function _take_tree(::Type{T}, n_initial::Int, untils) where {T}
    tree = _take_pooled!(_Tree{T})
    copyto!(resize!(tree.slots, n_initial), 1:n_initial)
    tree.free = 0
    copyto!(resize!(tree.ids, n_initial), 1:n_initial)
    copyto!(resize!(tree.places, n_initial), 1:n_initial)
    fill!(resize!(tree.drawn, n_initial), _Drawn(0, 0, 0, zero(T)))
    append!(empty!(tree.untils), untils)
    append!(empty!(tree.tagged), map(u -> !isempty(u), untils))
    return tree
end
Base.size(t::_Tree) = (length(t.ids),)
Base.@propagate_inbounds Base.getindex(t::_Tree, k::Int) = t.ids[k]
_member_positions(t::_Tree) = t.places
# Each place is a single exposure. Blocking it loses that contact entirely.
_proposal_fallback(::_Tree) = false
# A contact is listed on its infector when drawn, so the recorder need not log it.
_logs_blocked(::_Tree) = false
# The index cases are infected from the start, and each contact is offered the
# population's actions when it is created.
_population_members(::_Tree) = ()

# A new place for a contact `infector` drew on window `route`, at `time`.
function _place!(tree::_Tree, race, infector, time, type_idx, route)
    push!(tree.ids, 0)
    drawn = _Drawn(infector.id, route, type_idx, time)
    slot = tree.free
    if slot == 0
        push!(tree.drawn, drawn)
        slot = length(tree.drawn)
    else
        tree.free = tree.drawn[slot].infector
        tree.drawn[slot] = drawn
    end
    push!(tree.slots, slot)
    push!(race.processed, false)
    race.may_block && push!(race.head, 0)
    return length(tree.ids)
end

# A tree contact has one proposal, so no competing representative or persistent
# opening is needed. The contact already records its infector and route.
_stale_proposal(tree::_Tree, processed, represents, j, p) = false
function _discard_proposal!(tree::_Tree, best, j)
    _release_draw!(tree, j)
    return nothing
end
_settled_proposal!(tree::_Tree, j) = _release_draw!(tree, j)
function _release_draw!(tree::_Tree{T}, j) where {T}
    slot = abs(tree.slots[j])
    slot == 0 && return nothing
    # A released record links to the next free slot through its infector field.
    tree.drawn[slot] = _Drawn(tree.free, 0, 0, zero(T))
    tree.free = slot
    tree.slots[j] = 0
    return nothing
end
function _event_opening(tree::_Tree, event, openings, proposals, may_block)
    drawn = tree.drawn[event.ref]
    return _RouteOpening(drawn.infector, drawn.route, zero(event.time), oftype(event.time, Inf))
end
_store_opening!(tree::_Tree, openings, infector, route, open_t, close_t) = route
function _propose_member!(tree::_Tree, race, k, opening_id, t)
    t < oftype(t, Inf) || return nothing
    tree.slots[k] = -tree.slots[k]
    _enqueue!(race.queue, _Event(t, k, -tree.slots[k]))
    return nothing
end

# The person at place `k`, created now if there is none yet. The interventions
# set up their state on them, as on anyone created.
_member!(tree::_Tree, race, k) = _birth!(tree, race, k)
function _birth!(tree::_Tree, race, k)
    (; state) = race
    tree.ids[k] == 0 || return state.individuals[tree.ids[k]]
    drawn = tree.drawn[abs(tree.slots[k])]
    parent = state.individuals[drawn.infector]
    contact = drawn.type == 0 ? make_contact!(state, parent, drawn.time) :
        make_contact!(state, parent, drawn.time; type_idx = drawn.type)
    contact.generation = parent.generation + 1
    tree.tagged[drawn.route] && _tag_window!(contact, tree.untils[drawn.route])
    tree.ids[k] = contact.id
    push!(tree.places, k)
    isempty(race.interventions) && return contact
    for intervention in race.interventions
        initialise_individual!(intervention, contact, state)
    end
    _apply_population_actions!(state, race.interventions, (contact.id,))
    _apply_creation_actions!(state, race.creating, contact)
    return contact
end

# One infectiousness window of a branching process as a route, numbered
# `route`. `pending` holds the contiguous places the settling case just drew
# until the race proposes them. `reached` holds the places drawn for
# people a tracing ring reached before they settled, by person, until they do.
mutable struct _Offspring{W <: Infectiousness, T <: Real}
    window::W
    route::Int
    tree::_Tree{T}
    pending::UnitRange{Int}
    reached::Dict{Int, UnitRange{Int}}
end
function _Offspring(window::Infectiousness, route::Int, tree::_Tree{T}) where {T}
    return _Offspring(
        window, route, tree, 1:0, Dict{Int, UnitRange{Int}}()
    )
end

# The window the race closes this route on: the window's own `until` states,
# and removal by an intervention, which a branching process applies on every
# route.
function _offspring_window(w::Infectiousness)
    return RouteWindow(
        :transmission; until = (w.until..., INTERVENTION_REMOVAL), kernel = nothing
    )
end

# A contact made by a branching process exists nowhere else, so no other case
# can reach it: the route names no targets, and no pair kernel to redraw from.
(::_Offspring)(infector_id, state) = ()
_pair_kernel(::_Offspring, infector_id, target_id, state) = nothing

_route_open(o::_Offspring, ind, w::RouteWindow) = _state_time(ind, o.window.from)
_route_may_block(::_Offspring) = true
_route_builtin_blocks(::_Offspring) = _offspring_builtin_risk_blocks

# The contact was registered on its infector when it was created, and a
# branching process names no route.
_record_infection!(::_Offspring, infector, ind, w, named) = nothing

function _offspring_builtin_risk_blocks(parent, contact, state, transmission_time)
    pop_suscept = susceptible_fraction(state)
    pop_suscept <= 0.0 && return true
    pop_suscept < 1.0 && rand(state.rng) > pop_suscept && return true
    return _builtin_risk_blocks(parent, contact, state, transmission_time)
end

# Draw the settling case's contacts on this window, or take those a tracing
# ring already drew for it.
function _mint!(o::_Offspring, race, ind, max_generation)
    if !isempty(o.reached)
        reached = pop!(o.reached, ind.id, nothing)
        reached === nothing || return (o.pending = reached)
    end
    o.pending = _draw_contacts!(o, race, ind, max_generation, false)
    return nothing
end

"""
    reads_settled_state(x) -> Bool

Whether an offspring specification or a generation time `x` can read anything
settling a case writes: a time a [`Transition`](@ref) records, a value an
intervention sets, or the state of the whole outbreak, such as its case count.
A branching process asks this before a ring of contact tracing reaches a
person whose infection is not yet decided. If neither the window's offspring
nor its generation time reads settled state, the person's contacts are drawn
then and the ring grows through them, whether or not the person is infected.
Otherwise they are drawn only once the person settles as a case, so the ring
does not grow past a ring member who is never infected.

The default is `true`, the safe answer for a function, which may read
anything. A `Distribution`, a [`MultiTypeOffspring`](@ref) (which reads only
the case's type, set when it is created), a [`ClusterMixed`](@ref) (which
reads its parent's draw) and no generation time answer `false`. An offspring
specification of your own that draws from nothing settling writes declares
`EpiBranch.reads_settled_state(::YourOffspring) = false`.

A model written against [`contacts_of`](@ref) is asked the same question. If it
declares `EpiBranch.reads_settled_state(::YourModel) = false`, a ring that
reaches a person not yet settled calls `contacts_of` for them then, as though
they were infected at time 0, and grows through the people it names, whether or
not the person is infected. If the person settles as a case, those are the
people they expose, each at the time `contacts_of` gave plus their infection
time. Otherwise the ring grows past a person only once they settle as a case.
"""
reads_settled_state(_) = true
reads_settled_state(::Distribution) = false
reads_settled_state(::MultiTypeOffspring) = false
reads_settled_state(::NoGenerationTime) = false

# Whether a window's contacts can be drawn for a person before it settles.
_draws_before_settling(w::Infectiousness) =
    !reads_settled_state(w.offspring) && !reads_settled_state(w.kernel)

# What tracing and the interventions' actions read as a case's contacts. The
# settling case's contacts are created as people, since tracing writes to them
# before their infection is decided. A ring that grows through people who have
# not settled needs their contacts too, which a branching process draws only
# when a case settles. The walk draws them when it reaches such a person whose
# window opens at a time already known, such as its potential infection time,
# and they are the contacts that person goes on to expose if it is infected
# later. An uninfected person exposes none of them, since only infected people
# transmit. A window that opens on a later state has no time yet, and one whose
# draws may read what settling writes must wait for it. Then none of the
# person's windows is drawn early: all are drawn when it settles, and the ring
# grows through them then.
function _case_contacts(
        contacts, rts::Tuple{Vararg{Tuple{RouteWindow, _Offspring}}}, race, max_generation
    )
    return function (id, state)
        ind = state.individuals[id]
        settled = race.processed[first(rts)[2].tree.places[id]]
        early = !settled && _draws_early(rts, ind)
        for (_, o) in rts
            if settled
                for k in o.pending
                    _birth!(o.tree, race, k)
                end
            elseif early && !haskey(o.reached, id)
                o.reached[id] = _draw_contacts!(o, race, ind, max_generation, true)
            end
        end
        return ind.secondary_case_ids
    end
end

# Whether every window can draw `ind`'s contacts before it settles. A ring grows
# past a person once, so drawing only some windows early would leave the rest,
# drawn at settlement, out of its reach.
function _draws_early(rts, ind)
    for (_, o) in rts
        _draws_before_settling(o.window) || return false
        isfinite(_state_time(ind, o.window.from)) || return false
    end
    return true
end

# Draw a person's contacts on this window, once it opens, unless the person is
# at the generation bound. Their places form a contiguous range; contact
# times stay in the draw records. Create the contacts as people when `born`.
function _draw_contacts!(o::_Offspring, race, ind, max_generation, born::Bool)
    first_place = length(o.tree) + 1
    ind.generation >= max_generation && return first_place:(first_place - 1)
    from_t = _state_time(ind, o.window.from)
    isfinite(from_t) || return first_place:(first_place - 1)
    (; state) = race
    counts = draw_offspring(state.rng, o.window.offspring, ind, state)
    kernel = _has_contacts(counts) ?
        get_generation_time(o.window.kernel, ind) : NoGenerationTime()
    _draw_counts!(o, race, ind, counts, from_t, kernel, born)
    return first_place:length(o.tree)
end

# Single-type: one count. Multi-type: a count per type.
function _draw_counts!(o::_Offspring, race, ind, n::Int, from_t, kernel, born)
    for _ in 1:n
        _draw_contact!(o, race, ind, from_t, kernel, 0, born)
    end
    return nothing
end
function _draw_counts!(
        o::_Offspring, race, ind, counts::Vector{Int}, from_t, kernel, born
    )
    for (type_idx, n) in enumerate(counts), _ in 1:n
        _draw_contact!(o, race, ind, from_t, kernel, type_idx, born)
    end
    return nothing
end

# A window timed from a later state, such as onset, can draw a time before the
# case's own infection. No exposure happens then, so the draw gets no place.
function _draw_contact!(o::_Offspring, race, ind, from_t, kernel, type_idx, born)
    t = _window_infection_time(kernel, from_t, race.state)
    t < ind.infection_time && return nothing
    k = _place!(o.tree, race, ind, t, type_idx, o.route)
    born && _birth!(o.tree, race, k)
    return nothing
end

# Propose the contacts the case drew on this window, those that fall before it
# closes. One that falls later is a contact who was never infected, and is
# created as one now.
function _propose_along!(
        o::_Offspring, race, infector_id, opening_id, open_t, close_t, traits, live, ri
    )
    for k in o.pending
        t = o.tree.drawn[abs(o.tree.slots[k])].time
        if t <= close_t
            _propose!(race, k, opening_id, t)
        else
            _birth!(o.tree, race, k)
            _release_draw!(o.tree, k)
        end
    end
    o.pending = 1:0
    return nothing
end

# A run cut off at `stop_t` by a stopping rule or a time limit leaves contacts
# whose infection was still to be decided: those queued, and those drawn early
# for a ring member who had not settled. Their exposure falls after the run
# ended, as does that of a contact drawn after its infector's window closed but
# timed after `stop_t`, so none is part of the outbreak the run describes. One
# that tracing or an intervention already reached stays a contact of its case,
# with what was done to it, but with no infection time, as a network leaves a
# member the run never reached; the others were never created. Routes share
# the places, so the first route to finish does this for all of them.
function _finish!(o::_Offspring, race, stop_t)
    (; state) = race
    for k in eachindex(o.tree.slots)
        undecided = o.tree.slots[k] != 0
        undecided && _release_draw!(o.tree, k)
        id = o.tree.ids[k]
        id == 0 && continue
        contact = state.individuals[id]
        # An index case the run stopped before reaching stays an index case.
        contact.parent_id == 0 && continue
        (undecided || contact.infection_time > stop_t) &&
            (contact.infection_time = oftype(contact.infection_time, NaN))
    end
    # An abort recorded on a contact before its infection was decided stands
    # only on one that was infected before it.
    isempty(race.interventions) && return nothing
    for id in o.tree.ids
        id == 0 || _drop_stale_abort!(state.individuals[id])
    end
    return nothing
end

supplies_contacts(::BranchingProcess) = true

"""
    simulate_once(model::BranchingProcess, sim_opts; interventions, attributes,
                  progression, observation, recorder, rng)

Simulate a branching process on the continuous-time race. Each case creates its
contacts when it settles, one route per [`Infectiousness`](@ref) window, and
the race decides every contact in order of infection time.
"""
function simulate_once(
        model::BranchingProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    # An empty tuple lets the race omit intervention setup and per-case hooks.
    return _simulate_offspring(
        model, model.infectiousness, sim_opts, isempty(interventions) ? () : interventions;
        attributes, progression, observation, recorder, rng
    )
end

# A model of its own that says how many contacts each case makes, through
# `generate_offspring`, is a branching process with one window: the count is
# its offspring and `model_generation_time` its kernel, so it runs on the same
# route. The count may read anything, so a tracing ring draws a person's
# contacts only once they settle.
function _simulate_generated(
        model::TransmissionModel, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    window = Infectiousness(
        (rng, ind, state) -> generate_offspring(model, ind, state);
        kernel = model_generation_time(model)
    )
    return _simulate_offspring(
        model, (window,), sim_opts, isempty(interventions) ? () : interventions;
        attributes, progression, observation, recorder, rng
    )
end

function _simulate_offspring(
        model::TransmissionModel, windows, sim_opts::SimOpts, interventions;
        attributes, progression, observation, recorder, rng
    )
    state = initialise_state(model, sim_opts, interventions, progression, attributes, rng)
    # The race counts the index cases as it settles them.
    state.cumulative_cases = 0
    T = _timetype(state)
    tree = _take_tree(T, length(state.individuals), map(w -> w.until, windows))
    routes = ntuple(
        i -> (_offspring_window(windows[i]), _Offspring(windows[i], i, tree)), length(windows)
    )
    sellke_race!(
        state, tree, rng, sim_opts;
        seed! = (best, _, _) -> fill!(best, zero(T)), routes,
        interventions, contacts = (id, state) -> state.individuals[id].secondary_case_ids,
        risks = transmission_risks(model), recorder
    )
    _return_pooled!(tree)
    apply_observation!(observation, state, rng)
    return state
end
