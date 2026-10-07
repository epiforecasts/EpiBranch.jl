"""
    Hospitalisation(; delay, probability = 0.2, from = :onset_time)

Hospital admission: each case is admitted with probability `probability`
(default 0.2), `delay` days after symptom onset (or after `from`). Each case
records `:admitted` (`true`/`false`) and `:admission_time` (`Inf` if never
admitted). `delay`, `probability` and `from` take the same forms as in
[`Reporting`](@ref); cases that never reached `from` are not admitted.

# Examples

```julia
# 20% admitted, a median of about 7 days after onset
Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.2)
```

To admit only cases that meet some other condition (reported, traced,
vaccinated, ...), return `0.0` from a `probability` function when the
condition does not hold. Here only reported cases can be admitted; list
[`Reporting`](@ref) before `Hospitalisation` in the `progression` so reporting
is drawn first:

```julia
Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0
)
```
"""
Base.@kwdef struct Hospitalisation{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.2
    from::F = :onset_time
end

required_fields(h::Hospitalisation) = _from_required(h.from)

function initialise_individual!(::Hospitalisation, individual, state)
    individual.state[:admitted] = false
    individual.state[:admission_time] = Inf
    return nothing
end

function resolve_individual!(h::Hospitalisation, individual, state)
    anchor = _resolve_anchor(h.from, individual)
    time = transition_time(
        state.rng, individual, anchor, h.delay;
        probability = h.probability
    )
    time === nothing && return nothing
    individual.state[:admitted] = true
    individual.state[:admission_time] = time
    return nothing
end

function transition_loglik(h::Hospitalisation, individual::Individual)
    anchor = _resolve_anchor(h.from, individual)
    _anchor_ok(anchor) || return 0.0
    occurred = individual.state[:admitted]::Bool
    ll = transition_term(h.probability, h.delay, individual, anchor, occurred)
    occurred || return ll
    return ll + _delay_loglik(h.delay, individual.state[:admission_time] - anchor)
end
