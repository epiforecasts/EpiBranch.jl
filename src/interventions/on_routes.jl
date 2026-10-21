"""
    OnRoutes(intervention, routes)

Narrow `intervention` to only the named transmission routes, through
[`risk_applies`](@ref EpiBranch.risk_applies): its
[`competing_risk`](@ref EpiBranch.competing_risk), and any removal it reports
through [`infectious_removal_time`](@ref EpiBranch.infectious_removal_time),
then reach only a route — an [`Infectiousness`](@ref) window of a
[`BranchingProcess`](@ref), or a [`RouteWindow`](@ref) of a structured model —
whose `name` is in `routes`. Everything else about `intervention`, such as
who is eligible and when it acts, stays its own.

`routes` is the collection of route names to act on, such as `(:community,)`.
An infection from outside the population names no route, so an
`OnRoutes`-wrapped intervention never reaches one.

# Examples
```julia
using EpiBranch, Distributions

# Isolating at home stops community transmission but leaves household
# transmission running
household = Infectiousness(NegBin(1.0, 0.3); name = :household, kernel = Uniform(0, 5))
community = Infectiousness(NegBin(1.5, 0.3); name = :community, kernel = Uniform(0, 5))
model = BranchingProcess(household, community)
iso = OnRoutes(
    Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf),
    (:community,),
)
spec = ModelSpec(
    model; interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
)
```

`OnRoutes` composes with [`Scheduled`](@ref) and [`CapacityConstrained`](@ref)
in either order.
"""
struct OnRoutes{I <: AbstractIntervention, R} <: InterventionWrapper
    intervention::I
    routes::R
end

function risk_applies(o::OnRoutes, route)
    route !== nothing && route.name in o.routes || return false
    return risk_applies(o.intervention, route)
end
