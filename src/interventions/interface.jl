"""
    AbstractIntervention

Base type for all interventions. Subtypes implement one or both of:

- `resolve_individual!(intervention, individual, state)` — set isolation/tracing
  state on an individual before they transmit (e.g. isolation based on onset time)
- `apply_post_transmission!(intervention, state, new_individuals)` — act on newly
  created cases after transmission (e.g. contact tracing)
"""
abstract type AbstractIntervention end

"""
    resolve_individual!(intervention, individual, state)

Determine intervention state for `individual` before they transmit.
Sets isolation_time, traced, quarantined etc. Default: no-op.
"""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""
    apply_post_transmission!(intervention, state, new_individuals)

Apply intervention effects to newly created individuals after transmission.
Modifies `state` and/or `new_individuals` in place. Default: no-op.
"""
apply_post_transmission!(::AbstractIntervention, state, new_individuals) = nothing

"""
    transmission_fraction(individual, gen_time_dist, interventions=[])

Compute the effective transmission fraction for an individual given their
isolation state and the generation time distribution.

Without leaky isolation: returns G(t_iso) — the fraction of the generation
time CDF before isolation.

With leaky isolation (residual_transmission > 0 on an Isolation intervention):
returns G(t_iso) + residual · (1 - G(t_iso)).

Returns a value in [0, 1]. If not isolated, returns 1.0 (full transmission).
"""
function transmission_fraction(individual, gen_time_dist::Distribution,
                                interventions=AbstractIntervention[])
    !individual.isolated && return 1.0
    individual.isolation_time == Inf && return 1.0

    # Time from infection to isolation
    t_iso = individual.isolation_time - individual.infection_time
    t_iso <= 0.0 && return _residual_fraction(interventions)

    pre_iso = cdf(gen_time_dist, t_iso)

    # Check for leaky isolation
    residual = _residual_fraction(interventions)
    residual == 0.0 && return pre_iso

    return pre_iso + residual * (1.0 - pre_iso)
end

function _residual_fraction(interventions)
    for i in interventions
        if i isa Isolation
            return i.residual_transmission
        end
    end
    return 0.0
end
