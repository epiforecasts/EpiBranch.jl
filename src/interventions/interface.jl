"""
Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`,
`competing_risk`, `cap_offspring`.

To support time-based scheduling (`Scheduled(iv; start_time=...)`), also
implement [`intervention_time`](@ref) and [`reset!`](@ref). The default
implementations return `-Inf` and a no-op respectively, which is correct
for interventions whose effect time is always considered "now".
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
    competing_risk(intervention, parent, contact, state) -> Union{Nothing, Risk}

Return the [`Risk`](@ref) this intervention contributes against the
parent → contact transmission, or `nothing` if the intervention does
not gate this transmission. Default: `nothing`.

Resolution happens after `apply_post_transmission!` so that risks can
read state that other interventions have written on the contact
(e.g. `:vaccination_time` set by tracing-driven vaccination).
"""
competing_risk(::AbstractIntervention, parent, contact, state) = nothing

"""
    cap_offspring(intervention, parent, state) -> Union{Nothing, Int}

Return a hard cap on the number of offspring this parent may produce
this generation, or `nothing` for no cap. When multiple interventions
return a cap, the engine applies the tightest. Used for tree-shaping
interventions (e.g. gathering-size limits) that are not naturally
expressed as per-contact competing risks.

Applied during tree generation in `step!`, before contacts are
created.
"""
cap_offspring(::AbstractIntervention, parent, state) = nothing

"""
    intervention_time(intervention, individual)

Time at which this intervention's effect occurs for an individual. Used
by [`Scheduled`](@ref) to enforce `start_time`: if the intervention time
is earlier than `Scheduled`'s `start_time`, the effect is undone via
[`reset!`](@ref).

Default: `-Inf` (effect always applies).
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    reset!(intervention, individual)

Undo the effect of an intervention on an individual. Called by
[`Scheduled`](@ref) when `intervention_time` falls before `start_time`.

Default: no-op.
"""
reset!(::AbstractIntervention, ::Individual) = nothing

# ── Internal helpers ─────────────────────────────────────────────────

"""Resolve a Risk field that may be a Real or a callable."""
_resolve_risk_value(x::Real, rng, parent, contact, state) = float(x)
_resolve_risk_value(f, rng, parent, contact, state) = float(f(rng, parent, contact, state))

"""Apply an offspring cap to either a single-type count or a per-type vector."""
_apply_cap(count::Int, cap::Int) = min(count, cap)
function _apply_cap(counts::Vector{Int}, cap::Int)
    total = sum(counts)
    total <= cap && return counts
    # Trim proportionally, keeping at least the original ratio.
    scaled = [floor(Int, c * cap / total) for c in counts]
    # Distribute any rounding shortfall to the highest-count types first.
    deficit = cap - sum(scaled)
    if deficit > 0
        order = sortperm(counts, rev = true)
        for i in 1:deficit
            scaled[order[((i - 1) % length(order)) + 1]] += 1
        end
    end
    return scaled
end
