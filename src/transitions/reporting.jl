"""
Symptomatic cases are reported with probability `probability` after a
`delay` drawn per case, measured from `:onset_time`. Asymptomatic cases
(`:asymptomatic = true`) are never reported.

`probability` is a `Real` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. risk-group-specific detection).
`delay` is a `Distribution` or a `Function (rng, ind) -> Real`.

Initialises: `:reported = false`, `:reporting_time = Inf`.
On resolve, if the case is reported, sets both to reflect the draw.

Requires `:onset_time` and `:asymptomatic` (set via
[`clinical_presentation`](@ref)).
"""
Base.@kwdef struct Reporting{D, P} <: AbstractClinicalTransition
    delay::D
    probability::P = 1.0
end

required_fields(::Reporting) = [:onset_time, :asymptomatic]

function initialise_individual!(::Reporting, individual, state)
    individual.state[:reported] = false
    individual.state[:reporting_time] = Inf
    return nothing
end

function resolve_individual!(r::Reporting, individual, state)
    is_asymptomatic(individual) && return nothing
    ot = onset_time(individual)
    isnan(ot) && return nothing
    p = _resolve_probability(r.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    individual.state[:reported] = true
    individual.state[:reporting_time] = ot +
                                        _resolve_delay(r.delay, state.rng,
        individual)
    return nothing
end
