"""
Symptomatic cases are admitted to hospital with probability `probability`
after a `delay` drawn per case, measured from `:onset_time`. Asymptomatic
cases are never admitted.

Both `probability` and `delay` accept the heterogeneity shapes shared
across transitions: `probability` is a `Real` or `Function (rng, ind) -> Real`,
`delay` is a `Distribution` or `Function (rng, ind) -> Real`. Use the
function form for age-dependent admission rates, capacity-aware delays,
or anything else that should vary across cases.

If `requires_reporting = true`, only cases that have been reported (as
set by [`Reporting`](@ref)) are eligible for admission, with the delay
still measured from onset. This guards composability when admission
should follow detection.

Initialises: `:admitted = false`, `:admission_time = Inf`.

Requires `:onset_time` and `:asymptomatic`.
"""
Base.@kwdef struct Hospitalisation{D, P} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.2
    requires_reporting::Bool = false
end

required_fields(::Hospitalisation) = [:onset_time, :asymptomatic]

function initialise_individual!(::Hospitalisation, individual, state)
    individual.state[:admitted] = false
    individual.state[:admission_time] = Inf
    return nothing
end

function resolve_individual!(h::Hospitalisation, individual, state)
    is_asymptomatic(individual) && return nothing
    ot = onset_time(individual)
    isnan(ot) && return nothing
    if h.requires_reporting
        get(individual.state, :reported, false) || return nothing
    end
    p = _resolve_probability(h.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    individual.state[:admitted] = true
    individual.state[:admission_time] = ot +
                                        _resolve_delay(h.delay, state.rng,
        individual)
    return nothing
end
