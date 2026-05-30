"""
Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`,
`competing_risk`.

To support time-based scheduling (`Scheduled(iv; start_time=...)`), also
implement [`intervention_time`](@ref) and [`reset!`](@ref). The default
implementations return `-Inf` and a no-op respectively, which is correct
for interventions whose effect time is always considered "now".

Tree-shaping interventions (a hard cap on offspring per parent,
gathering-size limits, etc.) are expressed by passing a state-aware
function-form offspring distribution to [`BranchingProcess`](@ref EpiBranch.Engine.BranchingProcess) —
not via the intervention protocol. See the
[Extending guide](@ref "Extending EpiBranch") for an example.
"""
abstract type AbstractIntervention end

"""Set up intervention-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""Determine intervention state before transmission. Default: no-op."""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""Act on contacts after creation. All contacts are received. Default: no-op."""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""Whether an intervention is currently active given the simulation state. Default: always."""
is_active(::AbstractIntervention, ::SimulationState) = true

"""
    Risk(event_time, block_probability)

A competing risk contributed by an intervention against a single
contact's transmission. The risk has fired by transmission time `T` if
`event_time <= T`; when it has fired, transmission is blocked with
probability `block_probability`. A contact is infected iff no
intervention's risk blocks it.

Both fields accept either a `Real` or a function
`(rng, parent, contact, state) -> Real`. The function form lets the
event time or block probability depend on per-individual state, e.g.
age-conditional vaccine efficacy.

Use `event_time = -Inf` (the default) for risks that are not
time-tagged — pop_suscept, per-individual susceptibility,
infectiousness, and the like.

Returned by [`competing_risk`](@ref).
"""
struct Risk{T, P}
    event_time::T
    block_probability::P
end
Risk(; event_time = -Inf, block_probability) = Risk(event_time, block_probability)

"""
    competing_risk(intervention, parent, contact, state)
        -> Union{Nothing, Risk, NTuple{N, Risk}}

Return the [`Risk`](@ref)(s) this intervention contributes against the
parent → contact transmission, or `nothing` if the intervention does
not gate this transmission. Default: `nothing`.

Most interventions gate transmission through a single mechanism and
return one `Risk`. Interventions that gate it through more than one
mechanism — e.g. ring vaccination's susceptibility reduction on the
contact *and* its onward-infectiousness reduction on the parent — may
return a tuple of risks instead; the engine applies each independently.

Resolution happens after `apply_post_transmission!` so that risks can
read state that other interventions have written on the contact
(e.g. `:vaccination_time` set by tracing-driven vaccination).
"""
competing_risk(::AbstractIntervention, parent, contact, state) = nothing

"""
    intervention_time(intervention, individual)

Time at which this intervention's effect occurs for an individual. Used
by [`Scheduled`](@ref EpiBranch.Interventions.Scheduled) to enforce `start_time`: if the intervention time
is earlier than `Scheduled`'s `start_time`, the effect is undone via
[`reset!`](@ref).

Default: `-Inf` (effect always applies).
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    reset!(intervention, individual)

Undo the effect of an intervention on an individual. Called by
[`Scheduled`](@ref EpiBranch.Interventions.Scheduled) when `intervention_time` falls before `start_time`.

Default: no-op.
"""
reset!(::AbstractIntervention, ::Individual) = nothing
