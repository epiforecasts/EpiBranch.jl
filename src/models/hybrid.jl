# ── Hybrid exact/approximate population dynamics ───────────────────
#
# `HybridProcess` runs exact simulation while the outbreak is small
# (chance-dominated, and cheap at that size), then switches, once and for
# all, to a mean-field or linear-noise approximation once enough people are
# infected for the approximation to be both accurate and cheap. No filtering
# code lives here: `HybridProcess` only carries the state across the switch,
# so an external Kalman or particle filter sees one `simulate(model, state;
# max_time)` interface throughout, exactly as it would for either half on
# its own.

# The exact model a `HybridProcess` can drive: a `HomogeneousProcess`, or a
# `ModelSpec` built on one. Its default termination controls are already off
# (`_honours_termination_controls` is false for it), so a run under this
# function's `simulate(exact; max_time, ...)` call stops only for extinction
# or `max_time`, never silently capped by the `max_cases`/`max_generations`
# shortcuts' defaults. Those would otherwise leave a state that
# `simulate(model, state; max_time)` cannot continue (see
# `added-simulate-continuation.md`), stuck for good partway through
# `_advance_exact!`'s loop below.
_unwrap_homogeneous(m::HomogeneousProcess) = m
_unwrap_homogeneous(s::ModelSpec) = _unwrap_homogeneous(process(s))
function _unwrap_homogeneous(m)
    throw(
        ArgumentError(
            "HybridProcess needs its exact model to be a HomogeneousProcess, or a " *
                "ModelSpec built on one, so it can track how many people are " *
                "currently infectious"
        )
    )
end

"""
    HybridProcess(exact, approx; threshold, check_interval = 1.0)

A transmission model that runs as `exact` (a [`HomogeneousProcess`](@ref), or
a [`ModelSpec`](@ref) built on one) while the number of currently infectious
people is below `threshold`, and switches once and for all to `approx` (a
[`MeanFieldProcess`](@ref) or [`LinearNoiseProcess`](@ref) sharing `exact`'s
`population_size`) once that is reached, carrying the compartment counts
across rather than restarting. `check_interval` (days) sets how often the
exact run is checked against `threshold`: shorter catches the crossing
sooner, at the cost of resuming the race more often.

Simulating and continuing both use [`simulate`](@ref), exactly as for the
exact and approximate models on their own, so an external Kalman or particle
filter can score what it returns against data without knowing which half of
the model produced it.

# Example

```julia
using EpiBranch, Distributions
exact = ModelSpec(
    HomogeneousProcess(; transmission_rate = 0.4, population_size = 5000);
    progression = [Transition(:recovered; rate = 0.2, terminal = true)]
)
approx = MeanFieldProcess(;
    population_size = 5000, transmission_rate = 0.4, removal_rate = 0.2
)
model = HybridProcess(exact, approx; threshold = 20)
state = simulate(model; n_initial = 5, max_time = 100.0)
```
"""
struct HybridProcess{E, A}
    exact::E
    approx::A
    threshold::Int
    check_interval::Float64
end

function HybridProcess(exact, approx; threshold::Integer, check_interval::Real = 1.0)
    _unwrap_homogeneous(exact)
    threshold >= 1 || throw(ArgumentError("threshold must be ≥ 1"))
    check_interval > 0 || throw(ArgumentError("check_interval must be positive"))
    population_size(exact) == population_size(approx) || throw(
        ArgumentError(
            "exact and approx must share the same population_size " *
                "($(population_size(exact)) vs $(population_size(approx)))"
        )
    )
    return HybridProcess(exact, approx, Int(threshold), Float64(check_interval))
end

population_size(m::HybridProcess) = population_size(m.exact)

"""
    HybridState

The state of a [`HybridProcess`](@ref): `phase` is `:exact` or `:approx`,
whichever dynamics are currently driving it; `exact` holds the
[`SimulationState`](@ref) while in the exact phase (`nothing` after the
switch); `approx` holds the [`MeanFieldState`](@ref) or
[`LinearNoiseState`](@ref) once the switch has happened (`nothing` before).

For extension authors: `exact_t` is the time the exact phase has been driven
to, kept separately because `exact.max_infection_time` is the time of the
latest actual infection, which lags behind it whenever a stretch passes with
none.
"""
mutable struct HybridState{S}
    phase::Symbol
    exact::Union{Nothing, SimulationState}
    exact_t::Float64
    approx::Union{Nothing, S}
end

function Base.show(io::IO, s::HybridState)
    return print(io, "HybridState(phase=:$(s.phase))")
end

# How many people the exact state `es` has infectious as of time `t` (not
# necessarily `es.max_infection_time`, which lags `t` whenever no infection
# has happened since the last check): infected, and not yet at their
# natural-history outcome. Shared with `Prevalence` (see signals.jl) rather
# than walking `es.individuals` again here.
_n_infectious(es::SimulationState, t::Real) = Prevalence(:infection)(es, (0.0, t))

# The exact state's compartment counts as of time `t`, folded into the
# approximate state `model.approx` switches to: `I` from `_n_infectious`, `S`
# and `R` from the aggregate bookkeeping every exact run already keeps (which,
# unlike `I`, needs no query time: nothing between the last infection and `t`
# changes how many people have ever been infected).
function _compartment_counts(es::SimulationState, t::Real)
    I = Float64(_n_infectious(es, t))
    S = Float64(es.population_size) - Float64(es.cumulative_cases)
    R = Float64(es.cumulative_cases) - I
    return S, I, R
end

_to_approx_state(::MeanFieldProcess, es::SimulationState, t::Real) =
    MeanFieldState(_compartment_counts(es, t)..., t)
_to_approx_state(::LinearNoiseProcess, es::SimulationState, t::Real) =
    LinearNoiseState(_compartment_counts(es, t)..., t, zeros(2, 2))

_approx_state_type(::MeanFieldProcess) = MeanFieldState
_approx_state_type(::LinearNoiseProcess) = LinearNoiseState

"""
    simulate(model::HybridProcess; n_initial = 1, max_time, rng = Random.default_rng())

Simulate `model` from `t = 0`, with `n_initial` people infected, as exact
simulation until the number currently infectious reaches `model.threshold`
(or `max_time`, whichever comes first), then as `model.approx` from there to
`max_time`. Returns a [`HybridState`](@ref).
"""
function simulate(
        model::HybridProcess; n_initial::Integer = 1, max_time::Real,
        rng::AbstractRNG = Random.default_rng()
    )
    exact_state = simulate(model.exact; n_initial, max_time = 0.0, rng)
    state = HybridState{_approx_state_type(model.approx)}(:exact, exact_state, 0.0, nothing)
    return _advance_hybrid!(model, state, max_time)
end

"""
    simulate(model::HybridProcess, state::HybridState; max_time)

Continue `state`, from wherever it stopped, up to `max_time`: as exact
simulation if it had not yet crossed `model.threshold`, switching to
`model.approx` the moment it does, or straight on with `model.approx` if it
already had.
"""
function simulate(model::HybridProcess, state::HybridState; max_time::Real)
    return _advance_hybrid!(model, state, max_time)
end

function _advance_hybrid!(model::HybridProcess, state::HybridState, max_time::Real)
    state.phase === :approx && return _advance_approx!(model, state, max_time)
    return _advance_exact!(model, state, max_time)
end

function _advance_approx!(model::HybridProcess, state::HybridState{S}, max_time::Real) where {S}
    approx = state.approx::S
    max_time > approx.t || return state
    state.approx = simulate(model.approx, approx; max_time)
    return state
end

# Switch `state` to `model.approx` once `es` has crossed `model.threshold` as
# of time `t`: fold `es`'s compartment counts into the approximate state and,
# if there is time left before `max_time`, integrate it the rest of the way.
# Returns whether the switch happened, so `_advance_exact!` knows to stop
# looping.
function _cross_to_approx!(
        model::HybridProcess, state::HybridState, es::SimulationState, t::Real,
        max_time::Real
    )
    _n_infectious(es, t) >= model.threshold || return false
    state.exact = nothing
    state.exact_t = t
    state.phase = :approx
    state.approx = _to_approx_state(model.approx, es, t)
    remaining = max_time - t
    remaining > 0 && (state.approx = simulate(model.approx, state.approx; max_time))
    return true
end

function _advance_exact!(model::HybridProcess, state::HybridState, max_time::Real)
    es = state.exact::SimulationState
    t = state.exact_t
    _cross_to_approx!(model, state, es, t, max_time) && return state
    while t < max_time
        if es.extinct
            state.exact = es
            state.exact_t = t
            return state
        end
        next_check = min(t + model.check_interval, max_time)
        es = simulate(model.exact, es; max_time = next_check)
        t = next_check
        _cross_to_approx!(model, state, es, t, max_time) && return state
    end
    state.exact = es
    state.exact_t = t
    return state
end
