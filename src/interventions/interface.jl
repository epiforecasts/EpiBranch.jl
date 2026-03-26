"""
Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`.

To support time-based scheduling via `start_time`, also implement
[`intervention_time`](@ref) and [`reset!`](@ref).
"""
abstract type AbstractIntervention end

"""Set up intervention-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""Determine intervention state before transmission. Default: no-op."""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""Act on contacts after creation. All contacts are received. Default: no-op."""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""Fraction of transmission that occurs after isolation. Default: 0 (no transmission after isolation)."""
post_isolation_transmission(::AbstractIntervention) = 0.0

"""Whether an intervention is currently active given the simulation state. Default: always."""
is_active(::AbstractIntervention, ::SimulationState) = true

"""
    intervention_time(intervention, individual)

Time at which this intervention's effect occurs for an individual.
Used by the framework to enforce `start_time`: if the intervention time
is earlier than `start_time`, the effect is undone via [`reset!`](@ref).

Default: `-Inf` (effect always applies).
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    reset!(intervention, individual)

Undo the effect of an intervention on an individual.  Called by the
framework when `intervention_time` falls before `start_time`.

Default: no-op.
"""
reset!(::AbstractIntervention, ::Individual) = nothing

"""
    start_time(intervention)

Policy start time for an intervention. Default: `0.0` (always active).
"""
start_time(::AbstractIntervention) = 0.0

"""
    _enforce_start_time!(intervention, individual)

Check whether the intervention's effect on this individual falls before
the policy start time. If so, undo it.
"""
function _enforce_start_time!(intervention, individual)
    t = start_time(intervention)
    t <= 0.0 && return nothing
    intervention_time(intervention, individual) < t && reset!(intervention, individual)
    return nothing
end

"""
    _post_isolation_transmission(interventions)

Maximum residual transmission across all interventions in the stack.
"""
function _post_isolation_transmission(interventions)
    r = 0.0
    for i in interventions
        r = max(r, post_isolation_transmission(i))
    end
    return r
end
