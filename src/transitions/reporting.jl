"""
    Reporting(; delay, probability = 1.0, from = :onset_time)

Case reporting: each case is reported with probability `probability`, `delay`
days after symptom onset (or after `from`, if given). Each case records
`:reported` (`true`/`false`) and `:reporting_time` (`Inf` if never reported).

- `delay`: a distribution, a fixed number of days, or a function of the random
  number generator and the individual, `(rng, ind) -> ...`.
- `probability`: the chance a case is reported, a number or a function
  `(rng, ind) -> ...` (for example a different detection probability per risk
  group).
- `from`: the time the delay is measured from. Symptom onset by default;
  another recorded time such as `:test_time` or `:admission_time`; or a
  function of the individual, such as `ind -> ind.infection_time`.

Cases that never reached `from` are not reported. With the default `from`,
this includes asymptomatic cases from [`clinical_presentation`](@ref), whose
onset time is `NaN`. Measuring from onset requires the model's `attributes`
to set onset times, and the simulation checks for this when it starts.

# Examples

```julia
# report 80% of symptomatic cases, on average 3 days after onset
Reporting(delay = Gamma(2.0, 1.5), probability = 0.8)
```
"""
Base.@kwdef struct Reporting{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 1.0
    from::F = :onset_time
end

required_fields(r::Reporting) = _from_required(r.from)

function initialise_individual!(::Reporting, individual, state)
    individual.state[:reported] = false
    individual.state[:reporting_time] = Inf
    return nothing
end

function resolve_individual!(r::Reporting, individual, state)
    anchor = _resolve_anchor(r.from, individual)
    time = transition_time(
        state.rng, individual, anchor, r.delay;
        probability = r.probability
    )
    time === nothing && return nothing
    individual.state[:reported] = true
    individual.state[:reporting_time] = time
    return nothing
end

function transition_loglik(r::Reporting, individual::Individual)
    anchor = _resolve_anchor(r.from, individual)
    _anchor_ok(anchor) || return 0.0
    occurred = individual.state[:reported]::Bool
    ll = transition_term(r.probability, r.delay, individual, anchor, occurred)
    occurred || return ll
    return ll + _delay_loglik(r.delay, individual.state[:reporting_time] - anchor)
end
