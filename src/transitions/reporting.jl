"""
Cases are reported with probability `probability` after a `delay`
drawn per case, measured from `from`. `from` defaults to `:onset_time`.
`:infection` resolves to the individual's infection time; any other
`Symbol` names a state and resolves to its `:*_time` key (`from =
:onset` reads `:onset_time`, `from = :test` reads `:test_time`), unless
it already ends in `_time`, in which case it is looked up as-is (`from =
:test_time` also works). A `Function (ind) -> Real` is also accepted,
for composite anchors. If the anchor is not finite, the transition is
skipped: either the `from` state was never reached (its `:*_time` key
still at its `Inf` default), or the anchor is `NaN`,
most commonly an asymptomatic case under [`clinical_presentation`](@ref)
whose `:onset_time` is `NaN`.

`probability` is a `Real` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. risk-group-specific detection).
`delay` is a `Distribution` or a `Function (rng, ind) -> Real`.

Initialises: `:reported = false`, `:reporting_time = Inf`.

The simulation start-up validator requires the resolved `from` key to be
either an existing key (typically set by `attributes`) or one that a step
in `progression` produces. A `from` that names neither raises an error
at start-up, since the step timed from it would otherwise be silently
skipped for every case.

`simulate`, [`progression_loglik`](@ref), the chain-size likelihood
(`loglikelihood(::ChainSizes, ...)`, `chain_size_distribution`) and
`end_of_outbreak_probability` all read this step's detection probability: it
is the one per-case reporting route, replacing the deprecated
[`PerCaseObservation`](@ref) observation model.

A report drawn after the case's outcome (death or recovery) is reset to "did
not occur": see [`censor_after_outcome!`](@ref EpiBranch.censor_after_outcome!).
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
