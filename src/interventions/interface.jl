"""
    AbstractIntervention

Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`.
"""
abstract type AbstractIntervention end

"""Set up intervention-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""Determine intervention state before transmission. Default: no-op."""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""Act on contacts after creation. Receives all contacts. Default: no-op."""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""Residual transmission fraction while isolated. Default: 0 (perfect isolation)."""
residual_transmission(::AbstractIntervention) = 0.0

"""
    _residual_transmission(interventions)

Maximum residual transmission across all interventions in the stack.
"""
function _residual_transmission(interventions)
    r = 0.0
    for i in interventions
        r = max(r, residual_transmission(i))
    end
    return r
end
