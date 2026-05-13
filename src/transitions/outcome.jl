"""
Terminal transition: the case recovers. Always draws a candidate
recovery time on resolve (for symptomatic cases) by adding a sample
from `delay` to `:onset_time`. Asymptomatic cases never produce a
recovery candidate; their outcome is left unset by this transition
(the line-list projection treats absent outcomes as "no event recorded",
which is the correct default for asymptomatic cases that were never
clinically observed).

Initialises `:recovery_candidate_time = Inf`.

Requires `:onset_time` and `:asymptomatic`.

`Recovery` and [`Death`](@ref) compose as competing terminal events:
whichever has the earliest candidate time becomes the case's `:outcome`.
"""
Base.@kwdef struct Recovery <: AbstractClinicalTransition
    delay::Distribution
end

required_fields(::Recovery) = [:onset_time, :asymptomatic]
is_terminal(::Recovery) = true

function initialise_individual!(::Recovery, individual, state)
    individual.state[:recovery_candidate_time] = Inf
    return nothing
end

function resolve_individual!(r::Recovery, individual, state)
    is_asymptomatic(individual) && return nothing
    ot = onset_time(individual)
    isnan(ot) && return nothing
    individual.state[:recovery_candidate_time] = ot + rand(state.rng, r.delay)
    return nothing
end

function terminal_event(::Recovery, individual)
    t = get(individual.state, :recovery_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :recovered) : nothing
end

"""
Terminal transition: the case dies. The case has probability of death
`probability` (or, if `age_specific_cfr` is set, an age-banded CFR
overriding `probability`). When death is drawn, a candidate death time
is produced by adding a sample from `delay` to `:onset_time`.

`age_specific_cfr`, if provided, is a vector of `(lo, hi) => cfr`
pairs: the first interval containing the case's `:age` wins; ages
outside all intervals fall back to `probability`. This matches the
shape used by [`OutcomeOpts`](@ref).

Initialises `:death_candidate_time = Inf`.

Requires `:onset_time` and `:asymptomatic` (and `:age` if
`age_specific_cfr` is set).

`Death` and [`Recovery`](@ref) compose as competing terminal events.
"""
Base.@kwdef struct Death{C} <: AbstractClinicalTransition
    delay::Distribution
    probability::Float64 = 0.05
    age_specific_cfr::C = NoCFR()
end

required_fields(::Death{NoCFR}) = [:onset_time, :asymptomatic]
required_fields(::Death) = [:onset_time, :asymptomatic, :age]
is_terminal(::Death) = true

function initialise_individual!(::Death, individual, state)
    individual.state[:death_candidate_time] = Inf
    return nothing
end

function resolve_individual!(d::Death, individual, state)
    is_asymptomatic(individual) && return nothing
    ot = onset_time(individual)
    isnan(ot) && return nothing
    p = _death_probability(d, individual)
    rand(state.rng) < p || return nothing
    individual.state[:death_candidate_time] = ot + rand(state.rng, d.delay)
    return nothing
end

function terminal_event(::Death, individual)
    t = get(individual.state, :death_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :died) : nothing
end

_death_probability(d::Death{NoCFR}, individual) = d.probability

function _death_probability(d::Death, individual)
    age = get(individual.state, :age, missing)
    ismissing(age) && return d.probability
    for ((lo, hi), cfr) in d.age_specific_cfr
        lo <= age <= hi && return cfr
    end
    return d.probability
end
