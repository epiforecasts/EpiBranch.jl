"""
    AbstractIntervention

Base type for all interventions. Subtypes implement one or more of:

- `initialise_individual!(intervention, individual, state)` — set up
  intervention-specific fields on a newly created individual
- `resolve_individual!(intervention, individual, state)` — determine
  intervention state before transmission (e.g. set isolation time)
- `apply_post_transmission!(intervention, state, new_contacts)` — act
  on newly created contacts after transmission (e.g. contact tracing).
  Receives ALL contacts, both infected and non-infected.
"""
abstract type AbstractIntervention end

"""
    initialise_individual!(intervention, individual, state)

Set up intervention-specific fields on a newly created individual.
Called once per individual at creation, for each intervention.
Default: no-op.
"""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""
    resolve_individual!(intervention, individual, state)

Determine intervention state for `individual` before they transmit.
Default: no-op.
"""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""
    apply_post_transmission!(intervention, state, new_contacts)

Apply intervention effects to newly created contacts after transmission.
Receives all contacts (infected and non-infected). Can modify their state
(e.g. set traced, quarantined, vaccinated).
Default: no-op.
"""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""
    _residual_fraction(interventions)

Extract the residual transmission fraction from isolation interventions.
Used by the competing risk check in step!.
"""
function _residual_fraction(interventions)
    for i in interventions
        if i isa Isolation
            return i.residual_transmission
        end
    end
    return 0.0
end
