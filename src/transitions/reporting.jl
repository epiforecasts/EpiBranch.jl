"""
Cases are reported with probability `probability` after a `delay`
drawn per case, measured from `from`. `from` defaults to `:onset_time`,
but can be any `Symbol` (looked up in `ind.state`, e.g. `:test_time`,
`:admission_time`) or a `Function (ind) -> Real` (use this to anchor on
fields like `ind.infection_time` that live on the `Individual` rather
than `ind.state`). If the anchor evaluates to `NaN`, the transition is
skipped — the most common case being asymptomatic cases under
[`clinical_presentation`](@ref) whose `:onset_time` is `NaN`.

`probability` is a `Real` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. risk-group-specific detection).
`delay` is a `Distribution` or a `Function (rng, ind) -> Real`.

Initialises: `:reported = false`, `:reporting_time = Inf`.

When `from = :onset_time` (the default) the simulation start-up
validator requires `:onset_time` to be set by `attributes`. For other
anchors the requirement is dropped — typically the anchor key is set
by an upstream transition rather than by an attributes function, so
the validator can't catch it; instead it'll be `NaN` at resolve time
and the transition skips, which is the correct behaviour.
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
    isnan(anchor) && return nothing
    p = _resolve_probability(r.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    delay = _resolve_delay(r.delay, state.rng, individual)
    individual.state[:reported] = true
    individual.state[:reporting_time] = anchor + delay
    return nothing
end
