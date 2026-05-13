"""
Terminal transition: the case recovers. Cases with a finite
`:onset_time` get a candidate recovery time drawn from `delay` and
added to onset. Cases without an onset (asymptomatic or pre-symptomatic)
do not produce a recovery candidate; their outcome is left unset by
this transition.

`delay` is a `Distribution` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. age-conditional recovery delay).

Initialises `:recovery_candidate_time = Inf`.

Requires `:onset_time`.

`Recovery` and [`Death`](@ref) compose as competing terminal events:
whichever has the earliest candidate time becomes the case's `:outcome`.
Other user-defined terminal transitions (with `is_terminal = true` and
a `terminal_event` method) participate in the same arbitration.
"""
Base.@kwdef struct Recovery{D} <: AbstractClinicalTransition
    delay::D
end

required_fields(::Recovery) = [:onset_time]
is_terminal(::Recovery) = true

function initialise_individual!(::Recovery, individual, state)
    individual.state[:recovery_candidate_time] = Inf
    return nothing
end

function resolve_individual!(r::Recovery, individual, state)
    ot = onset_time(individual)
    isnan(ot) && return nothing
    individual.state[:recovery_candidate_time] = ot +
                                                 _resolve_delay(
        r.delay, state.rng, individual)
    return nothing
end

function terminal_event(::Recovery, individual)
    t = get(individual.state, :recovery_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :recovered) : nothing
end

"""
Terminal transition: the case dies. When death is drawn, a candidate
death time is produced by adding a sample from `delay` to `:onset_time`.

`probability` is a `Real` or a `Function (rng, ind) -> Real`. Use the
function form for age- or risk-conditional CFRs:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02)
```

`delay` accepts a `Distribution` or `Function (rng, ind) -> Real`,
making time-to-death heterogeneity available the same way.

Initialises `:death_candidate_time = Inf`.

Requires `:onset_time`.

`Death` and [`Recovery`](@ref) compose as competing terminal events.
"""
Base.@kwdef struct Death{D, P} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.05
end

required_fields(::Death) = [:onset_time]
is_terminal(::Death) = true

function initialise_individual!(::Death, individual, state)
    individual.state[:death_candidate_time] = Inf
    return nothing
end

function resolve_individual!(d::Death, individual, state)
    ot = onset_time(individual)
    isnan(ot) && return nothing
    p = _resolve_probability(d.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    individual.state[:death_candidate_time] = ot +
                                              _resolve_delay(
        d.delay, state.rng, individual)
    return nothing
end

function terminal_event(::Death, individual)
    t = get(individual.state, :death_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :died) : nothing
end
