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
    RoutedNetwork(windows; from = nothing, external_hazard = 0.0, obs_end = Inf)

Network transmission over several routes at once.

`windows` is a collection of [`RouteWindow`](@ref)s. Each carries

- `reach`: an adjacency list giving that route's edges, so different routes can
  connect different pairs of the same nodes;
- `kernel`: the contact-interval distribution along those edges;
- `until`: the states that end this route, which is what lets one route be cut
  and another left alone. Include `EpiBranch.INTERVENTION_REMOVAL` for a route
  that a composed `Isolation` should end;
- `from`: where the route's infectiousness starts. Leave it at the default
  `nothing` to take the model's `from`, or, if that is also `nothing`, the start
  the progression implies (`:infectious` when a latent period produces it).
  Name `:infection` for a route open from the moment of infection, or a later
  state for a route that opens later, such as a funeral route from `:died`.
- `contacts_from`: when the route's neighbours become the case's contacts for
  tracing. Leave it at `:infection` for standing relationships, which tracing
  reaches as on `NetworkProcess`; give a funeral route `contacts_from = :died`
  so its contacts are traced only if the funeral happened, and not before it.
- `traceable`: the probability that a case can name a neighbour on this route.
  Contact tracing reaches only named neighbours. Keep the default `1.0` for
  people a case can always name, such as its household, and lower it for a
  route of casual contact.

The model decides naming once for each pair of case and neighbour, when it
passes the case's contacts to tracing. A neighbour reachable on several routes
is named with the highest of their `traceable` probabilities: someone a case
both lives with and sees in the community can be named because they live
together. The model draws a single uniform number `u` from the simulation's
random number generator, and the neighbour is nameable on every route whose
`traceable` exceeds `u`. It can be traced from the earliest `contacts_from` time
among those routes. Each route on its own therefore names the neighbour with its
own probability, and together they name it with the highest. The intervention
then traces a named neighbour with its own probability, so the two multiply. A
neighbour reachable only on routes at `1.0` or `0.0` needs no draw, so routes
left at the default use no random numbers for naming.

All routes run over the same node set, so every adjacency must have the same
length.

# Example

Households as cliques, community contact as a sparser graph over the same
people. Self-isolation ends community transmission and leaves the household
route running:

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
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0))],
    attributes = clinical_presentation(incubation_period = LogNormal(1.0, 0.3)))
```

Because the household route does not list the intervention removal, a case that
isolates keeps infecting its household to the end of its infectious period,
which is what self-isolation at home actually does. `R` is unchanged by any of
this: it stays what the case would achieve if never removed, and the realised
figure falls out of which routes were cut.

With contact tracing, `traceable = 0.2` on the community route means a case can
name only one community contact in five, while it can name everyone it lives
with. A household member who is also a community contact is always nameable.
"""
struct RoutedNetwork{W <: AbstractVector, E} <: TransmissionModel
    windows::W                       # RouteWindows; each `reach` is an adjacency list
    from::Union{Symbol, Nothing}     # override the derived infectious start
    external_hazard::E               # community force of infection (0 = none)
    obs_end::Float64                 # end of the community-importation window
    n::Int                           # node count, shared by every route
end

function RoutedNetwork(windows::AbstractVector{<:RouteWindow};
        from = nothing, external_hazard = 0.0, obs_end = Inf)
    isempty(windows) && throw(ArgumentError("RoutedNetwork needs at least one route"))
    for w in windows
        w.reach isa AbstractVector{<:AbstractVector{<:Integer}} || throw(ArgumentError(
            "route :$(w.name) must carry an adjacency list as its `reach`"))
    end
    n = length(first(windows).reach)
    all(length(w.reach) == n for w in windows) || throw(ArgumentError(
        "every route's adjacency must cover the same nodes (got lengths " *
        "$(join([length(w.reach) for w in windows], ", ")))"))
    _valid_external(external_hazard) ||
        throw(ArgumentError("external_hazard must be a non-negative number or a continuous distribution"))
    obs_end_value = Float64(obs_end)
    (!isnan(obs_end_value) && obs_end_value >= 0) || throw(ArgumentError(
        "obs_end must be a non-negative number (Inf allowed), got $obs_end"))
    # A model-level start applies to every route that leaves its own unset, so
    # the stored routes are the ones the simulation runs and `window_open` on
    # them agrees with it.
    if from !== nothing
        windows = [_start_unset(w, from) for w in windows]
    end
    return RoutedNetwork(windows, from, _normalise_external(external_hazard),
        obs_end_value, n)
end

# A route that leaves its start unset takes `from`.
function _start_unset(w::RouteWindow, from)
    w.from === nothing || return w
    return RouteWindow(w.name, from, w.until, w.kernel, w.reach, w.contacts_from,
        w.traceable)
end

population_size(::RoutedNetwork) = NoPopulation()
_honours_termination_controls(::RoutedNetwork) = false

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
function _route_contacts(windows, interventions, ind::Individual, i::Integer,
        rng::AbstractRNG)
    T = typeof(ind.infection_time)
    ids = Int[]
    certain = T[]                      # earliest time on a route at traceable 1
    uncertain = Tuple{Int, T, Float64}[]  # (contact slot, time, traceable) in (0, 1)
    for w in windows
        w.traceable > 0 || continue
        if w.contacts_from === :infection
            t = T(-Inf)
        else
            t = convert(T, get(ind.state, Symbol(w.contacts_from, :_time), T(Inf)))
            (isfinite(t) && window_close(ind, w, interventions) > t) || continue
        end
        for nb in w.reach[i]
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
    print(io, "RoutedNetwork(nodes=$(m.n), routes=[$routes]",
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end

function _simulate(model::RoutedNetwork, sim_opts::SimOpts; interventions, attributes,
        progression, observation, rng, condition, max_attempts)
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(model, sim_opts; interventions, attributes, progression,
            observation, rng, condition = nothing, max_attempts),
        condition, max_attempts)

    # Every route that did not name its own start takes the model's, or the one
    # the progression implies, so a latent period delays all of them together.
    derived = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, model.n, interventions; setup = (ind, i) -> nothing)

    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
            "the whole network); build the process with e.g. `obs_end = 30.0`"))

    windows = [_start_unset(w, derived) for w in model.windows]
    routes = Tuple((w, _route_targets(w)) for w in windows)

    EpiBranch._sellke_race!(state, collect(1:model.n), rng;
        routes = routes, interventions = interventions,
        risks = EpiBranch.transmission_risks(model),
        seed! = (best, members, r) -> _seed_network!(
            best, members, model.external_hazard, sim_opts.n_initial, Tobs, r),
        contacts = (inf, st) -> _route_contacts(
            windows, interventions, st.individuals[inf], inf, st.rng))

    _reconcile_sellke_bookkeeping!(state)
    apply_observation!(observation, state, rng)
    return state
end

# One route's susceptible targets, each with that route's kernel.
function _route_targets(w::RouteWindow)
    adjacency, kernel = w.reach, w.kernel
    return (inf, st) -> ((nb, kernel) for nb in adjacency[inf]
    if !is_infected(st.individuals[nb]))
end
