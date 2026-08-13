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
- `from`: where the route's infectiousness starts. Leave at the default
  `:infection` to take the same start the progression implies (`:infectious`
  when a latent period produces it), or name a state for a route that opens
  later, such as a funeral route from `:died`.

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
    return RoutedNetwork(windows, from, _normalise_external(external_hazard),
        Float64(obs_end), n)
end

population_size(::RoutedNetwork) = NoPopulation()
_honours_termination_controls(::RoutedNetwork) = false

# Contacts for tracing are the union of every route's neighbours: someone you
# live with and also see in the community is one contact, traced once.
EpiBranch.supplies_contacts(::RoutedNetwork) = true

function _all_neighbours(m::RoutedNetwork, i::Integer)
    length(m.windows) == 1 && return m.windows[1].reach[i]
    seen = Int[]
    for w in m.windows, nb in w.reach[i]

        nb in seen || push!(seen, nb)
    end
    return seen
end

function Base.show(io::IO, m::RoutedNetwork)
    routes = join([":$(w.name)" for w in m.windows], ", ")
    print(io, "RoutedNetwork(nodes=$(m.n), routes=[$routes]",
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end

function _simulate(model::RoutedNetwork, sim_opts::SimOpts; interventions, attributes,
        progression, observation, rng, condition, kwargs...)
    if condition !== nothing
        return _retry_for_condition(condition, sim_opts,
            () -> _simulate(model, sim_opts; interventions, attributes, progression,
                observation, rng, condition = nothing, kwargs...))
    end
    EpiBranch._warn_unhonoured_interventions(model, interventions)

    # Every route that did not name its own start takes the one the progression
    # implies, so a latent period delays all of them together.
    derived = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, model.n, interventions; setup = (ind, i) -> nothing)

    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
            "the whole network); build the process with e.g. `obs_end = 30.0`"))

    routes = Tuple((
                       w.from === :infection ?
                       RouteWindow(w.name, derived, w.until, w.kernel, w.reach) : w,
                       _route_targets(w))
    for w in model.windows)

    EpiBranch._sellke_race!(state, collect(1:model.n), rng;
        routes = routes, interventions = interventions,
        seed! = (best, members, r) -> _seed_network!(
            best, members, model.external_hazard, sim_opts.n_initial, Tobs, r),
        contacts = (inf, st) -> _all_neighbours(model, inf))

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
