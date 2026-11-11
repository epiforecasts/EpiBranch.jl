# ── Approximate population dynamics ─────────────────────────────────
#
# A closed, homogeneously-mixing SIR population (the dynamics
# `HomogeneousProcess` simulates exactly), approximated as a state-space
# model: `MeanFieldProcess` integrates the deterministic mean-field ODE,
# `LinearNoiseProcess` adds a Gaussian process around that same path. Both
# share `simulate(model, state; max_time)` with the exact simulation, so an
# external Kalman or particle filter, or a `HybridProcess` (see hybrid.jl),
# can treat all three as interchangeable state-space transitions.

"""
    MeanFieldProcess(; population_size, transmission_rate, removal_rate, step = 0.05)

The deterministic SIR mean-field approximation of a closed population of
`population_size` people that mixes homogeneously, the large-population limit
of [`HomogeneousProcess`](@ref) composed with a single exponential removal
step (`Transition(:recovered; rate = removal_rate, terminal = true)`):

```
dS/dt = -βSI/N,  dI/dt = βSI/N - γI,  dR/dt = γI
```

with transmission rate β (`transmission_rate`) and removal rate γ
(`removal_rate`). The compartments are integrated as continuous numbers,
not people, by fourth-order Runge–Kutta with a fixed step of `step` days
(the last step of a call is shortened to end exactly at `max_time`).

Pass the result to [`simulate`](@ref) for a [`MeanFieldState`](@ref), and that
state back to [`simulate`](@ref), with a later `max_time`, to continue the
integration from where it stopped: the same interface the exact
simulation uses. See [`LinearNoiseProcess`](@ref) for the same dynamics with
Gaussian noise around the mean path, and [`HybridProcess`](@ref) for
switching between exact simulation and either one.

# Example

```julia
using EpiBranch
model = MeanFieldProcess(;
    population_size = 10_000, transmission_rate = 0.4, removal_rate = 0.2
)
state = simulate(model; n_initial = 5, max_time = 30.0)
state = simulate(model, state; max_time = 60.0)
```
"""
struct MeanFieldProcess
    population_size::Int
    transmission_rate::Float64
    removal_rate::Float64
    step::Float64
end

function MeanFieldProcess(;
        population_size::Integer, transmission_rate::Real, removal_rate::Real,
        step::Real = 0.05
    )
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    (isfinite(transmission_rate) && transmission_rate >= 0) || throw(
        ArgumentError("transmission_rate must be finite and non-negative")
    )
    (isfinite(removal_rate) && removal_rate >= 0) ||
        throw(ArgumentError("removal_rate must be finite and non-negative"))
    step > 0 || throw(ArgumentError("step must be positive"))
    return MeanFieldProcess(
        Int(population_size), Float64(transmission_rate), Float64(removal_rate), Float64(step)
    )
end

population_size(m::MeanFieldProcess) = m.population_size

function Base.show(io::IO, m::MeanFieldProcess)
    return print(
        io,
        "MeanFieldProcess(population_size=$(m.population_size), β=$(m.transmission_rate), γ=$(m.removal_rate))"
    )
end

"""
    MeanFieldState

The compartment counts of a [`MeanFieldProcess`](@ref) at a point in time:
`S`, `I` and `R`, continuous because the mean-field approximation treats the
population as infinitely divisible rather than as discrete people. `t` is the
time (days since the start of the outbreak) the state has been integrated to.
"""
struct MeanFieldState
    S::Float64
    I::Float64
    R::Float64
    t::Float64
end

function Base.show(io::IO, s::MeanFieldState)
    return print(
        io,
        "MeanFieldState(S=$(round(s.S, digits = 1)), I=$(round(s.I, digits = 1)), " *
            "R=$(round(s.R, digits = 1)), t=$(s.t))"
    )
end

"""
    simulate(model::MeanFieldProcess; n_initial = 1, max_time)

Integrate the mean-field SIR ODE of `model` from `t = 0`, with `n_initial`
people infected and the rest susceptible, up to `max_time`. Returns a
[`MeanFieldState`](@ref); continue it further with
`simulate(model, state; max_time)`.
"""
function simulate(model::MeanFieldProcess; n_initial::Integer = 1, max_time::Real)
    1 <= n_initial <= model.population_size ||
        throw(ArgumentError("n_initial must be between 1 and population_size"))
    state = MeanFieldState(
        Float64(model.population_size - n_initial), Float64(n_initial), 0.0, 0.0
    )
    return simulate(model, state; max_time)
end

"""
    simulate(model::MeanFieldProcess, state::MeanFieldState; max_time)

Continue integrating `state`'s mean-field ODE up to `max_time`, which must be
later than `state.t`.
"""
function simulate(model::MeanFieldProcess, state::MeanFieldState; max_time::Real)
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    S, I = _integrate_sir(model, state.S, state.I, state.t, max_time)
    R = model.population_size - S - I
    return MeanFieldState(S, I, R, Float64(max_time))
end

# The SIR mean-field drift (dS/dt, dI/dt); R is never integrated directly,
# since S + I + R = N holds exactly and R is read off that conservation law
# instead.
_sir_drift(S, I, β, γ, N) = (-β * S * I / N, β * S * I / N - γ * I)

# The drift's Jacobian with respect to (S, I), shared by the linear noise
# approximation below.
function _sir_jacobian(S, I, β, γ, N)
    return [
        -β * I / N -β * S / N
        β * I / N β * S / N - γ
    ]
end

function _rk4_sir_step(S, I, dt, f)
    k1S, k1I = f(S, I)
    k2S, k2I = f(S + dt / 2 * k1S, I + dt / 2 * k1I)
    k3S, k3I = f(S + dt / 2 * k2S, I + dt / 2 * k2I)
    k4S, k4I = f(S + dt * k3S, I + dt * k3I)
    Snew = S + dt / 6 * (k1S + 2k2S + 2k3S + k4S)
    Inew = I + dt / 6 * (k1I + 2k2I + 2k3I + k4I)
    return Snew, Inew
end

# Fixed-step RK4 from `t0` to `max_time`, in steps no longer than
# `model.step` (the last one shortened to end exactly at `max_time`).
function _integrate_sir(model::MeanFieldProcess, S, I, t0, max_time)
    N = model.population_size
    β = model.transmission_rate
    γ = model.removal_rate
    n_steps = max(1, ceil(Int, (max_time - t0) / model.step))
    dt = (max_time - t0) / n_steps
    f = (s, i) -> _sir_drift(s, i, β, γ, N)
    for _ in 1:n_steps
        S, I = _rk4_sir_step(S, I, dt, f)
    end
    # RK4 with a small step overshoots by very little, if at all; clamping
    # keeps the compartments physical (0 to N) rather than passing on a
    # fractional excursion past a boundary to the next call.
    S = clamp(S, 0.0, Float64(N))
    I = clamp(I, 0.0, Float64(N) - S)
    return S, I
end

"""
    LinearNoiseProcess(; population_size, transmission_rate, removal_rate, step = 0.05)

The same closed-population SIR dynamics as [`MeanFieldProcess`](@ref), represented
as a Gaussian process around the deterministic mean path (the linear noise
approximation) rather than the bare mean: the state holds the mean
compartment counts and their covariance, which grows from the demographic
noise of the two reactions (infection, removal) and is damped by the
linearised dynamics around the mean. This is the predictive distribution an
external Kalman filter needs for its prediction step; a particle filter can
draw its own particles from the returned mean and covariance instead of
running the exact simulation.

The Gaussian has no mass on early extinction, and the approximation needs
enough initial cases for extinction to be negligible. From one or a few cases
it describes only the outbreaks that take off, and only roughly: the mean and
variance are off. With around 50 initial cases it matches the exact
simulation well. When early extinction matters, use the exact simulation or
[`HybridProcess`](@ref), which simulates exactly while few people are
infectious.

Accepts the same keywords as [`MeanFieldProcess`](@ref). The mean path
integrates identically, with the addition of the covariance.

# Example

```julia
using EpiBranch
model = LinearNoiseProcess(;
    population_size = 10_000, transmission_rate = 0.4, removal_rate = 0.2
)
state = simulate(model; n_initial = 5, max_time = 30.0)
state.covariance   # 2×2 covariance of (S, I)
```
"""
struct LinearNoiseProcess
    population_size::Int
    transmission_rate::Float64
    removal_rate::Float64
    step::Float64
end

function LinearNoiseProcess(;
        population_size::Integer, transmission_rate::Real, removal_rate::Real,
        step::Real = 0.05
    )
    mean_field = MeanFieldProcess(; population_size, transmission_rate, removal_rate, step)
    return LinearNoiseProcess(
        mean_field.population_size, mean_field.transmission_rate,
        mean_field.removal_rate, mean_field.step
    )
end

population_size(m::LinearNoiseProcess) = m.population_size

function Base.show(io::IO, m::LinearNoiseProcess)
    return print(
        io,
        "LinearNoiseProcess(population_size=$(m.population_size), β=$(m.transmission_rate), γ=$(m.removal_rate))"
    )
end

"""
    LinearNoiseState

The Gaussian state of a [`LinearNoiseProcess`](@ref): mean compartment counts
`S`, `I` and `R`, time `t`, and `covariance`, the 2×2 covariance of `(S, I)`
(`R` has no separate uncertainty because it is `population_size - S - I`).
"""
struct LinearNoiseState
    S::Float64
    I::Float64
    R::Float64
    t::Float64
    covariance::Matrix{Float64}
end

function Base.show(io::IO, s::LinearNoiseState)
    return print(
        io,
        "LinearNoiseState(S=$(round(s.S, digits = 1)), I=$(round(s.I, digits = 1)), " *
            "R=$(round(s.R, digits = 1)), t=$(s.t))"
    )
end

"""
    simulate(model::LinearNoiseProcess; n_initial = 1, max_time)

Integrate the linear noise approximation of `model` from `t = 0`, with
`n_initial` people infected, the rest susceptible, and no uncertainty yet
(a zero covariance), up to `max_time`. The default of one initial case is
only a starting value. The Gaussian approximation ignores early extinction and
from one or a few cases describes only outbreaks that take off, and only
roughly. Use enough initial cases for extinction to be negligible (around 50
is close to the exact simulation). Returns a [`LinearNoiseState`](@ref);
continue it further with `simulate(model, state; max_time)`.
"""
function simulate(model::LinearNoiseProcess; n_initial::Integer = 1, max_time::Real)
    1 <= n_initial <= model.population_size ||
        throw(ArgumentError("n_initial must be between 1 and population_size"))
    state = LinearNoiseState(
        Float64(model.population_size - n_initial), Float64(n_initial), 0.0, 0.0,
        zeros(2, 2)
    )
    return simulate(model, state; max_time)
end

"""
    simulate(model::LinearNoiseProcess, state::LinearNoiseState; max_time)

Continue `state`'s mean path and covariance up to `max_time`, which must be
later than `state.t`.
"""
function simulate(model::LinearNoiseProcess, state::LinearNoiseState; max_time::Real)
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    S, I, Σ = _integrate_lna(model, state.S, state.I, state.covariance, state.t, max_time)
    R = model.population_size - S - I
    return LinearNoiseState(S, I, R, Float64(max_time), Σ)
end

# The outer products of the two reactions' stoichiometry vectors (infection:
# S -> S - 1, I -> I + 1; removal: I -> I - 1), scaling the Chemical Langevin
# diffusion each contributes by its own propensity (`_integrate_lna`).
const _SIR_INFECTION_OUTER = [1.0 -1.0; -1.0 1.0]
const _SIR_REMOVAL_OUTER = [0.0 0.0; 0.0 1.0]

# Mean (RK4, shared with `MeanFieldProcess`) and covariance (explicit Euler,
# evaluating the Jacobian and diffusion at the step's start) from `t0` to
# `max_time`, in steps no longer than `model.step`. A first-order update for
# Σ, against RK4 for the mean, is standard practice for the (linear, given
# the mean path) covariance ODE at a small step, and keeps the two states in
# lockstep without a second, independent step count.
function _integrate_lna(model::LinearNoiseProcess, S, I, Σ, t0, max_time)
    N = model.population_size
    β = model.transmission_rate
    γ = model.removal_rate
    n_steps = max(1, ceil(Int, (max_time - t0) / model.step))
    dt = (max_time - t0) / n_steps
    f = (s, i) -> _sir_drift(s, i, β, γ, N)
    for _ in 1:n_steps
        a1 = β * S * I / N
        a2 = γ * I
        J = _sir_jacobian(S, I, β, γ, N)
        Q = a1 * _SIR_INFECTION_OUTER + a2 * _SIR_REMOVAL_OUTER
        Σ = Σ + dt * (J * Σ + Σ * J' + Q)
        Σ = (Σ + Σ') / 2   # undo floating-point asymmetry from the update above
        S, I = _rk4_sir_step(S, I, dt, f)
    end
    S = clamp(S, 0.0, Float64(N))
    I = clamp(I, 0.0, Float64(N) - S)
    return S, I, Σ
end
