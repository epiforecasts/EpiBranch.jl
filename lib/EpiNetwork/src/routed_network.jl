# ── A network with several transmission routes ───────────────────────
#
# `NetworkProcess` gives every edge the same status: one kernel, one infectious
# window, so anything that closes the window cuts all transmission at once. That
# is the wrong shape for the most ordinary control measure there is. Someone who
# self-isolates stops mixing in the community and goes on infecting the people
# they live with, and often infects them more.
#
# `RoutedNetwork` separates the edges into routes. Each route is a
# `RouteWindow` carrying its own contact-interval kernel, its own adjacency, and
# its own set of states that end it — so isolation can cut one route and leave
# another running.

"""
    RoutedNetwork(windows; from = nothing, external_hazard = 0.0, obs_end = Inf, n = nothing)

Transmission over a contact network with several routes, such as household and
community, each with its own contacts, timing and response to control
measures. Isolation can then end a case's community transmission while they go
on infecting the people they live with.

# Example

Households as fully connected groups, community contact as a sparser network
over the same people. Self-isolation ends community transmission and leaves the
household route running. In this example `household_adjacency` and
`community_adjacency` are contact lists, one vector of people per person:

```julia
using EpiNetwork, EpiBranch, Distributions

household = RouteWindow(:household; until = (:recovered,),
    kernel = Weibull(1.5, 3.0), reach = household_adjacency)
community = RouteWindow(:community;
    until = (:recovered, EpiBranch.INTERVENTION_REMOVAL),
    kernel = Exponential(12.0), reach = community_adjacency)

model = ModelSpec(RoutedNetwork([household, community]);
    progression = [Transition(:recovered; from = :infection, delay = 7.0,
        terminal = true)],
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)],
    attributes = clinical_presentation(incubation_period = LogNormal(1.0, 0.3)))
```

Because the household route does not list `INTERVENTION_REMOVAL`, a case who
isolates keeps infecting their household to the end of their infectious
period, as self-isolation at home does. Each route's kernel describes
transmission as if the case were never removed; the realised number of
secondary cases falls according to which routes isolation cuts.

# Routes

`windows` is a collection of [`RouteWindow`](@ref)s, each with:

- `reach`: the route's contact list (`reach[i]` lists the people `i` can
  infect on this route), so different routes can connect different pairs of the
  same people. A route whose contacts are cheaper to sample than to list, such
  as a spatial kernel over a large population or contact with everyone in an
  area, can instead have a `reach` that proposes its own: a value with
  [`EpiBranch.proposes_own_contacts`](@ref) true, and methods for
  [`EpiBranch.propose_along`](@ref) and [`EpiBranch.route_pair_kernel`](@ref).
  Its `kernel` is then unused (pass `kernel = nothing`), since such a `reach`
  times its own contacts;
- `kernel`: the contact interval along those contacts, in days, in any of the
  forms [`NetworkProcess`](@ref) accepts (one distribution, a function
  `(infector, susceptible) -> Distribution`, a [`PairKernel`](@ref), or a
  per-contact vector parallel to `reach`). A `PairKernel` that reads people's
  records lists them in its `watches`, and only the routes reading a record
  that changes have their contacts redrawn;
- `until`: the states that end transmission on this route, so one route can be
  cut and another left alone. Include `EpiBranch.INTERVENTION_REMOVAL` for a
  route that isolation or quarantine should end;
- `from`: when the route's infectiousness starts. The default, `nothing`, takes
  the model's `from` or, if that is also `nothing`, the start the progression
  implies (`:infectious` when a latent period produces it). Name `:infection`
  for a route infectious from the moment of infection, or a later state for one
  that starts later, such as a funeral route from `:died`;
- `contacts_from`: when the route's contacts become the case's contacts for
  tracing. Leave it at `:infection` for standing relationships; give a funeral
  route `contacts_from = :died` so its contacts are traced only once the
  funeral has happened;
- `traceable`: the probability that a case can name a contact on this route.
  Contact tracing reaches only named contacts. Keep the default `1.0` for
  people a case can always name, such as their household, and lower it for
  casual contact.

All routes cover the same people, so every adjacency `reach` must have the
same length. `n`, the number of nodes, is read from one of those lists when
any route has one, and must agree with it if given; with no adjacency
list anywhere, because every route proposes its own contacts, pass `n`
yourself. `external_hazard` and `obs_end` are as for [`NetworkProcess`](@ref).

A route that proposes its own contacts is traced on what it actually
proposed, read back from the same call `propose_along` made for
transmission, rather than a fresh walk of its full reach: tracing and
transmission then never disagree about who a case's contacts were, and
tracing costs nothing beyond what proposing already paid.

Whether a case names a given contact is decided once, when the case's
contacts are passed to tracing. A contact reachable on several routes is
named with the highest of those routes' `traceable` probabilities, and can be
traced from the earliest `contacts_from` time among the routes that name
them; for example, with `traceable = 0.2` on the community route, a case can
name one community contact in five but everyone they live with, including
household members they also meet in the community. The intervention then traces a named contact with
its own probability, so the two multiply.

A case infected on one of these routes records the route's `name` in its
`:infection_route`, and a community introduction records `:external`. Both
appear in [`linelist`](@ref), so cases can be broken down by setting.
"""
struct RoutedNetwork{W <: AbstractVector, E} <: TransmissionModel
    windows::W                       # RouteWindows; each `reach` is an adjacency list
    from::Union{Symbol, Nothing}     # override the derived infectious start
    external_hazard::E               # community force of infection (0 = none)
    obs_end::Float64                 # end of the community-importation window
    n::Int                           # node count, shared by every route
end

function RoutedNetwork(
        windows::AbstractVector{<:RouteWindow};
        from = nothing, external_hazard = 0.0, obs_end = Inf, n = nothing
    )
    isempty(windows) && throw(ArgumentError("RoutedNetwork needs at least one route"))
    for w in windows
        if EpiBranch.proposes_own_contacts(w.reach)
            w.kernel === nothing || throw(
                ArgumentError(
                    "route :$(w.name) proposes its own contacts, so its `kernel` is " *
                        "unused; pass `kernel = nothing`"
                )
            )
        else
            w.reach isa AbstractVector{<:AbstractVector{<:Integer}} || throw(
                ArgumentError(
                    "route :$(w.name) must carry an adjacency list, or a `reach` " *
                        "that proposes its own contacts (see `EpiBranch.proposes_own_contacts`)"
                )
            )
        end
    end
    adjacency_lengths = [
        length(w.reach) for w in windows
            if w.reach isa AbstractVector{<:AbstractVector{<:Integer}}
    ]
    if isempty(adjacency_lengths)
        n === nothing && throw(
            ArgumentError(
                "RoutedNetwork needs `n`, the number of nodes, when no route's " *
                    "`reach` is an adjacency list to read it from"
            )
        )
    else
        all(==(first(adjacency_lengths)), adjacency_lengths) || throw(
            ArgumentError(
                "every route's adjacency must cover the same nodes (got lengths " *
                    "$(join(adjacency_lengths, ", ")))"
            )
        )
        (n === nothing || n == first(adjacency_lengths)) || throw(
            ArgumentError(
                "n = $n does not match the $(first(adjacency_lengths)) nodes the " *
                    "adjacency lists cover"
            )
        )
        n = first(adjacency_lengths)
    end
    # A route's kernel accepts the same forms as `NetworkProcess`'s edge
    # kernel; a per-edge vector is validated and normalised against this
    # route's own adjacency. A route that proposes its own contacts has no
    # kernel to validate.
    windows = [_validate_route_kernel(w) for w in windows]
    _check_external(external_hazard, n)
    obs_end_value = Float64(obs_end)
    (!isnan(obs_end_value) && obs_end_value >= 0) || throw(
        ArgumentError(
            "obs_end must be a non-negative number (Inf allowed), got $obs_end"
        )
    )
    # A model-level start applies to every route that leaves its own unset, so
    # the stored routes are the ones the simulation runs and `window_open` on
    # them agrees with it.
    if from !== nothing
        windows = [_start_unset(w, from) for w in windows]
    end
    return RoutedNetwork(
        windows, from, _normalise_external(external_hazard),
        obs_end_value, n
    )
end

# A route that leaves its start unset takes `from`.
function _start_unset(w::RouteWindow, from)
    w.from === nothing || return w
    return RouteWindow(
        w.name, from, w.until, w.kernel, w.reach, w.contacts_from,
        w.traceable
    )
end

# Validate and normalise a route's kernel against its own adjacency, exactly
# as `NetworkProcess` does for its edge kernel.
function _validate_route_kernel(w::RouteWindow)
    return RouteWindow(
        w.name, w.from, w.until,
        _validate_kernel(w.kernel, w.reach; route = w.name), w.reach,
        w.contacts_from, w.traceable
    )
end

population_size(::RoutedNetwork) = NoPopulation()
# The race honours a `stopping_rules` entry, or a non-default `max_cases` or
# `max_generations`, but leaves those two keywords' defaults off, so an
# existing run over the fixed network keeps going to extinction or `max_time`
# unless a cap is explicitly asked for (`_drop_default_shortcuts`).
_honours_termination_controls(::RoutedNetwork) = false

# Each route closes independently at the earliest of its own `until` states
# (see `_warn_uncovered_terminal_states` in EpiBranch's branching_process.jl),
# so a terminal state missing from one route's `until` leaves that route open
# even if every other route covers it.
function _validate_process_windows(m::RoutedNetwork, progression)
    for w in m.windows
        _warn_uncovered_terminal_states(w.until, progression; route = w.name, from = w.from)
    end
    return nothing
end

EpiBranch.supplies_contacts(::RoutedNetwork) = true

# Contacts for tracing are the union of the neighbours on every route, each
# paired with the earliest time it can be traced. Someone you live with and also
# see in the community is one contact, traced once. A route over standing
# relationships (`contacts_from = :infection`) reaches its neighbours as
# `NetworkProcess` does, whenever the case's trace happens. A route whose
# contacts come about through an event, such as a funeral, contributes nothing
# if the event never happened or the route was cut before it (a survivor's
# funeral, or a safe burial after isolation), and its contacts are traced no
# earlier than the event.
#
# Only contacts the case can name reach tracing. One uniform draw per neighbour
# decides naming on all its routes at once: the neighbour is nameable on each
# route whose `traceable` exceeds the draw, so it is named with the highest
# route probability and traced from the earliest time among the routes it was
# named on. Routes at exactly 0 or 1 decide without a draw, as does a neighbour
# that a route at 1 already names from its earliest time, so a fully traceable
# model uses no random numbers here.
#
# `routes` are the `(window, route_targets)` pairs built for the race itself,
# not just the windows: a route that proposes its own contacts has nothing in
# `w.reach[i]` to read, and its neighbours come from `route_targets` instead,
# the ones its own `propose_along` call actually proposed (see
# `_TracedProposals`).
function _route_contacts(
        routes, interventions, ind::Individual, i::Integer,
        rng::AbstractRNG
    )
    T = typeof(ind.infection_time)
    ids = Int[]
    certain = T[]                      # earliest time on a route at traceable 1
    uncertain = Tuple{Int, T, Float64}[]  # (contact slot, time, traceable) in (0, 1)
    for (w, route_targets) in routes
        w.traceable > 0 || continue
        if w.contacts_from === :infection
            t = T(-Inf)
        else
            t = _state_real(ind.state, Symbol(w.contacts_from, :_time), T(Inf))
            (isfinite(t) && window_close(ind, w, interventions) > t) || continue
        end
        neighbours = EpiBranch.proposes_own_contacts(route_targets) ?
            get(route_targets.proposed, i, Int[]) : w.reach[i]
        for nb in neighbours
            k = findfirst(==(nb), ids)
            if k === nothing
                push!(ids, nb)
                push!(certain, T(Inf))
                k = length(ids)
            end
            if w.traceable == 1
                certain[k] = min(certain[k], t)
            else
                push!(uncertain, (k, t, w.traceable))
            end
        end
    end
    named = Tuple{Int, T}[]
    for k in eachindex(ids)
        t = certain[k]
        # A draw is needed only if some uncertain route could name the neighbour
        # earlier than the certain ones already do.
        if any(e -> e[1] == k && e[2] < t, uncertain)
            u = rand(rng)
            for (slot, t_route, p) in uncertain
                slot == k && u < p && (t = min(t, t_route))
            end
        end
        t < Inf && push!(named, (ids[k], t))
    end
    return named
end

function Base.show(io::IO, m::RoutedNetwork)
    routes = join([":$(w.name)" for w in m.windows], ", ")
    return print(
        io, "RoutedNetwork(nodes=$(m.n), routes=[$routes]",
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")"
    )
end

function simulate_once(
        model::RoutedNetwork, sim_opts::SimOpts; interventions, attributes,
        progression, observation, recorder, rng
    )
    # Every route that did not name its own start takes the model's, or the one
    # the progression implies, so a latent period delays all of them together.
    derived = something(model.from, infectious_from(progression))
    Tobs = model.obs_end

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, model.n, interventions; setup = (ind, i) -> nothing)

    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(
        ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
                "the whole network); build the process with e.g. `obs_end = 30.0`"
        )
    )

    windows = [_start_unset(w, derived) for w in model.windows]
    # A route that proposes its own contacts records what it proposed only
    # when something reads contacts back: tracing, or an intervention that
    # proposes actions needing them (`_route_contacts` is also how
    # `_apply_case_actions!` reaches a case's contacts). Interventions never
    # change over a run, so this is decided once.
    tracing = any(
        iv -> EpiBranch.traces_contacts(iv) ||
            (EpiBranch._proposes_actions(iv) && EpiBranch.needs_contacts(iv)),
        interventions
    )
    routes = Tuple((w, _route_targets(w, tracing)) for w in windows)

    EpiBranch.sellke_race!(
        state, collect(1:model.n), rng, sim_opts;
        routes = routes, interventions = interventions,
        watches = Tuple(EpiBranch.watched_records(w.kernel) for w in windows),
        projections = Tuple(EpiBranch.kernel_projection(w.kernel) for w in windows),
        risks = EpiBranch.transmission_risks(model), recorder = recorder,
        seed! = (best, members, r) -> _seed_network!(
            best, members, state, model.external_hazard, sim_opts.n_initial, Tobs, r;
            initial_cases = sim_opts.initial_cases
        ),
        introduction = _ext_active(model.external_hazard) ?
            (EpiBranch._ext_prepare(model.external_hazard), Tobs) : nothing,
        contacts = (inf, st) -> _route_contacts(
            routes, interventions, st.individuals[inf], inf, st.rng
        )
    )

    apply_observation!(observation, state, rng)
    return state
end

# One route's susceptible targets, each with that route's kernel resolved for
# the pair: a shared distribution, a per-edge vector, a covariate callable or a
# `PairKernel`, exactly as `NetworkProcess` resolves its edge kernel (see
# `_resolve_kernel`). A `reach` that proposes its own contacts is wrapped so
# that what it proposes can be read back for tracing (`_TracedProposals`).
function _route_targets(w::RouteWindow, tracing::Bool)
    EpiBranch.proposes_own_contacts(w.reach) && return _TracedProposals(w.reach, tracing)
    adjacency, kernel = w.reach, w.kernel
    return (inf, st) -> (
        (nb, _resolve_kernel(kernel, adjacency, inf, pos, st, w.from))
            for (pos, nb) in enumerate(adjacency[inf])
            if !is_infected(st.individuals[nb])
    )
end

# A route whose `reach` proposes its own contacts, rather than listing them,
# has nothing for `_route_contacts` to read `w.reach[i]` from. Wrapping it
# records each infector's own proposals as they are made, so tracing reads
# back exactly the contacts transmission considered, at no extra cost beyond
# that one call: a fresh walk of the reach, which such a `reach` exists to
# avoid, would reintroduce the per-case cost it is meant to spare. Recording
# is skipped unless some intervention traces, since nothing else reads it.
struct _TracedProposals{R}
    reach::R
    proposed::Dict{Int, Vector{Int}}
    tracing::Bool
end
_TracedProposals(reach, tracing::Bool) = _TracedProposals(reach, Dict{Int, Vector{Int}}(), tracing)

EpiBranch.proposes_own_contacts(::_TracedProposals) = true
EpiBranch.route_pair_kernel(t::_TracedProposals, infector_id, target_id, state) =
    EpiBranch.route_pair_kernel(t.reach, infector_id, target_id, state)
EpiBranch.pending_beyond(t::_TracedProposals, state, reinfects::Bool) =
    EpiBranch.pending_beyond(t.reach, state, reinfects)

function EpiBranch.propose_along(
        t::_TracedProposals, infector_id, state, open_t, close_t, capped, traits
    )
    contacts = EpiBranch.propose_along(t.reach, infector_id, state, open_t, close_t, capped, traits)
    if t.tracing
        contacts = collect(contacts)
        t.proposed[infector_id] = [id for (id, _) in contacts if id != infector_id]
    end
    return contacts
end

function EpiBranch._validate_initial_cases(model::RoutedNetwork, opts::SimOpts)
    return EpiBranch._validate_initial_case_ids(opts, model.n)
end
