"""
Cases with a finite `:onset_time` are reported with probability
`probability` after a `delay` drawn per case, measured from
`:onset_time`. Cases without a recorded onset (`isnan(:onset_time)` —
asymptomatic or pre-symptomatic) are skipped; they were never
clinically observed.

`probability` is a `Real` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. risk-group-specific detection).
`delay` is a `Distribution` or a `Function (rng, ind) -> Real`.

Initialises: `:reported = false`, `:reporting_time = Inf`.
On resolve, if the case is reported, sets both to reflect the draw.

Requires `:onset_time`. For diseases that model asymptomatic cases,
use [`clinical_presentation`](@ref)`(prob_asymptomatic = ...)` — it
sets onset to `NaN` for asymptomatic cases, which the transition's
NaN check handles. For diseases without an asymptomatic concept, a
minimal attributes function that only sets `:onset_time` is enough.
"""
Base.@kwdef struct Reporting{D, P} <: AbstractClinicalTransition
    delay::D
    probability::P = 1.0
end

required_fields(::Reporting) = [:onset_time]

function initialise_individual!(::Reporting, individual, state)
    individual.state[:reported] = false
    individual.state[:reporting_time] = Inf
    return nothing
end

function resolve_individual!(r::Reporting, individual, state)
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
