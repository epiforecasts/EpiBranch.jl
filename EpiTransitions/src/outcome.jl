"""
Terminal transition: the case recovers. A candidate recovery time is
drawn from `delay` and added to the value of `from`. `from` defaults
to `:onset_time` but accepts any `Symbol` (state-dict key) or
`Function (ind) -> Real` — see [`Reporting`](@ref EpiTransitions.Reporting) for the anchor
semantics. If the anchor is `NaN`, no recovery candidate is produced.

`delay` is a `Distribution` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. age-conditional recovery delay).

Initialises `:recovery_candidate_time = Inf`.

`Recovery` and [`Death`](@ref EpiTransitions.Death) compose as competing terminal events:
whichever has the earliest candidate time becomes the case's `:outcome`.
Other user-defined terminal transitions (with `is_terminal = true` and
a `terminal_event` method) participate in the same arbitration.
"""
Base.@kwdef struct Recovery{D, F} <: AbstractClinicalTransition
    delay::D
    from::F = :onset_time
end

EpiBranchCore.required_fields(r::Recovery) = _from_required(r.from)
EpiBranchCore.is_terminal(::Recovery) = true

function EpiBranchCore.initialise_individual!(::Recovery, individual, state)
    individual.state[:recovery_candidate_time] = Inf
    return nothing
end

function EpiBranchCore.resolve_individual!(r::Recovery, individual, state)
    anchor = _resolve_anchor(r.from, individual)
    isnan(anchor) && return nothing
    delay = _resolve_delay(r.delay, state.rng, individual)
    individual.state[:recovery_candidate_time] = anchor + delay
    return nothing
end

function EpiBranchCore.terminal_event(::Recovery, individual)
    t = get(individual.state, :recovery_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :recovered) : nothing
end

"""
Terminal transition: the case dies. When death is drawn, a candidate
death time is produced by adding a sample from `delay` to the value of
`from`. `from` defaults to `:onset_time` but accepts any `Symbol` or
`Function (ind) -> Real` — see [`Reporting`](@ref EpiTransitions.Reporting) for the anchor
semantics.

`probability` is a `Real` or a `Function (rng, ind) -> Real`. Use the
function form for age- or risk-conditional CFRs:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02)
```

`delay` accepts a `Distribution` or `Function (rng, ind) -> Real`,
making time-to-death heterogeneity available the same way.

Initialises `:death_candidate_time = Inf`.

`Death` and [`Recovery`](@ref EpiTransitions.Recovery) compose as competing terminal events.
"""
Base.@kwdef struct Death{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.05
    from::F = :onset_time
end

EpiBranchCore.required_fields(d::Death) = _from_required(d.from)
EpiBranchCore.is_terminal(::Death) = true

function EpiBranchCore.initialise_individual!(::Death, individual, state)
    individual.state[:death_candidate_time] = Inf
    return nothing
end

function EpiBranchCore.resolve_individual!(d::Death, individual, state)
    anchor = _resolve_anchor(d.from, individual)
    isnan(anchor) && return nothing
    p = _resolve_probability(d.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    delay = _resolve_delay(d.delay, state.rng, individual)
    individual.state[:death_candidate_time] = anchor + delay
    return nothing
end

function EpiBranchCore.terminal_event(::Death, individual)
    t = get(individual.state, :death_candidate_time, Inf)::Float64
    return isfinite(t) ? (t, :died) : nothing
end
