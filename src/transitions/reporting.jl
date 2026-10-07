"""
Cases are reported with probability `probability` after a `delay`
drawn per case, measured from `from`. `from` defaults to `:onset_time`.
`:infection` resolves to the individual's infection time; any other
`Symbol` names a state and resolves to its `:*_time` key (so `from =
:onset` reads `:onset_time`, and `from = :test` reads `:test_time`),
unless it already ends in `_time`, in which case it is looked up as-is
(so `from = :test_time` works the same way). A `Function (ind) -> Real`
is also accepted, for composite anchors. If the anchor is not finite,
the transition is skipped — either the `from` state was never reached
(its `:*_time` key still at its `Inf` default), or the anchor is `NaN`,
most commonly an asymptomatic case under [`clinical_presentation`](@ref)
whose `:onset_time` is `NaN`.

`probability` is a `Real` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. risk-group-specific detection).
`delay` is a `Distribution` or a `Function (rng, ind) -> Real`.

Initialises: `:reported = false`, `:reporting_time = Inf`.

The simulation start-up validator requires the resolved `from` key to be
either an existing key (typically set by `attributes`) or one that a step
in `progression` produces. A `from` that names neither raises an error
at start-up, since the step it anchors would otherwise be silently
skipped for every case.
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

function transition_loglik(r::Reporting, individual::Individual)
    anchor = _state_time(individual, r.from)
    _anchor_ok(anchor) || return 0.0
    occurred = individual.state[:reported]::Bool
    ll = transition_term(r.probability, r.delay, individual, anchor, occurred)
    occurred || return ll
    return ll + _delay_loglik(r.delay, individual.state[:reporting_time] - anchor)
end
