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
  another recorded time such as `:test_time` (`:test` works too) or
  `:admission_time`; or a function of the individual, such as
  `ind -> ind.infection_time`.

Cases that never reached `from` are not reported. With the default `from`,
this includes asymptomatic cases from [`clinical_presentation`](@ref), whose
onset time is `NaN`. The time measured from must be one that the model's
`attributes` set or that a step in its `progression` records; the simulation
checks this when it starts, and stops with an error if neither does. A report
drawn for a time after the case has died or recovered is reset to "not
reported".

`simulate`, [`progression_loglik`](@ref), the chain-size likelihood and
[`end_of_outbreak_probability`](@ref) all read this step's reporting
probability. It replaces the deprecated [`PerCaseObservation`](@ref).

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
    anchor = _state_time(individual, r.from)
    time = transition_time(
        state.rng, individual, anchor, r.delay;
        probability = r.probability
    )
    time === nothing && return nothing
    individual.state[:reported] = true
    individual.state[:reporting_time] = time
    return nothing
end

function censor_after_outcome!(::Reporting, individual)
    _censor_event!(individual, :reported, :reporting_time)
    return nothing
end

function transition_loglik(r::Reporting, individual::Individual)
    anchor = _state_time(individual, r.from)
    _anchor_ok(anchor) || return 0.0
    flag = individual.state[:reported]::Bool
    time = flag ? individual.state[:reporting_time] : Inf
    occurred = flag && time <= outcome_time(individual)
    ll = transition_term(
        r.probability, r.delay, individual, anchor, occurred;
        censor = min(infection_aborted_time(individual), outcome_time(individual))
    )
    occurred || return ll
    return ll + _delay_loglik(r.delay, time - anchor)
end
