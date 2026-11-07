# ── Approximate dynamics for a MixingProcess ─────────────────────────
#
# The multi-group analogue of `MeanFieldProcess`/`LinearNoiseProcess`
# (population_dynamics.jl): one SIR compartment per mixing group rather than
# one for the whole population, the large-population limit of a
# `MixingProcess` composed with a single exponential removal step. Built
# directly from the `ModelSpec` it approximates, reading the group sizes off
# the population its `attributes` assign and the between-group rate and
# calendar off the `MixingProcess` itself, so the mixing structure is never
# written down twice. Shares `simulate(model, state; max_time)` with the
# scalar approximations and the exact models, so `HybridProcess` (hybrid.jl)
# can switch to either from a `MixingProcess` run.

# A member's mixing group, read off its state the same missing-safe way
# `_MassAction` reads it when the exact race is built (mass_action.jl), so a
# person the attributes left without one of `mixing_by`'s keys falls into a
# `missing` group here too, rather than erroring in one path and not the
# other.
_group_key(ind, mixing_by::Tuple) = Tuple(get(ind.state, k, missing) for k in mixing_by)

# The groups `spec`'s `attributes` assign and how many people land in each,
# realised by actually building the population (`new_state` plus
# `add_individuals!`, with nobody infected and no interventions), so a group
# assignment that depends on values available to `attributes` (such as an
# individual id) is evaluated as it would be in the exact model. A stochastic
# assignment would make the group sizes themselves random on every draw,
# which a fixed-size compartment cannot represent, so `attributes` is
# expected to assign groups deterministically.
function _mixing_group_layout(spec::ModelSpec, mixing_by::Tuple, rng::AbstractRNG)
    bare = process(spec)
    state = new_state(bare, progression(spec), attributes(spec), rng)
    add_individuals!(state, population_size(bare), (); setup = (ind, i) -> nothing)
    keys = Tuple[]
    index = Dict{Any, Int}()
    sizes = Int[]
    for ind in state.individuals
        key = _group_key(ind, mixing_by)
        g = get!(index, key) do
            push!(keys, key)
            push!(sizes, 0)
            length(keys)
        end
        sizes[g] += 1
    end
    return keys, sizes
end

# `spec` must be a `ModelSpec` built on a `MixingProcess`, so there is both a
# `rate`/`calendar` to read and an `attributes` that assigns the groups
# `rate` is defined over. Its interventions are not read at all, since
# nothing here folds a vaccination's effect into the ODE, so a `spec` with
# one would silently approximate the model without it rather than with it;
# refusing the model is safer than a silent mismatch.
function _mixing_process_spec(spec, what)
    spec isa ModelSpec && process(spec) isa MixingProcess || throw(
        ArgumentError(
            "$what needs a ModelSpec built on a MixingProcess, with `attributes` " *
                "assigning the keys its `mixing_by` groups by, so the group sizes " *
                "and the between-group rate can be read from it"
        )
    )
    isempty(interventions(spec)) || throw(
        ArgumentError(
            "$what cannot yet be built from a ModelSpec with interventions: " *
                "vaccination is not yet read from them as a population-level input, " *
                "so the approximation would silently leave it out"
        )
    )
    return spec
end

# `Λ[g, h]`, the rate at which one infective of group `h` contacts one
# member of group `g`: `Λ * I` is then the force of infection on every
# group, for `I` the vector of each group's infectious count, exactly as
# `MixingProcess.rate(from, to)` gives that rate to the exact race.
_contact_matrix(rate, groups) = [rate(h, g) for g in groups, h in groups]

_calendar_value(::Nothing, t::Real) = 1.0
_calendar_value(calendar, t::Real) = calendar_multiplier(calendar, t)

"""
    MixingMeanFieldProcess(spec::ModelSpec; removal_rate, step = 0.05,
                           rng = Random.default_rng())

The deterministic mean-field approximation of a [`MixingProcess`](@ref)
composed with a single exponential removal step, one SIR compartment per
mixing group instead of one for the whole population:

```
dS_g/dt = -S_g Σ_h Λ[g,h] I_h,  dI_g/dt = S_g Σ_h Λ[g,h] I_h - γ I_g,  dR_g/dt = γ I_g
```

`spec` must be a [`ModelSpec`](@ref) built on a `MixingProcess`, with
`attributes` assigning the keys `mixing_by` groups by. The groups, their
sizes, the between-group rate `Λ[g,h] = rate(h, g)` and any `calendar`
scaling it are all read from `spec`, so the same mixing structure drives the
exact and the approximate dynamics rather than being given twice: the group
sizes are read by building the population `spec` describes (with nobody
infected) and counting who `attributes` puts in each group. The groups must
come out the same on every draw (by `ind.id`, say), because a fixed-size
compartment cannot represent a group with a random size.

`removal_rate` (γ) is the exponential removal rate the exact model's
progression gives its terminal step, exactly as [`MeanFieldProcess`](@ref)'s
own `removal_rate` is. The compartments are integrated as continuous
numbers, not people, by fourth-order Runge–Kutta with a fixed step of
`step` days.

Vaccination is not yet read from the interventions in `spec` as a
population-level input. `spec` must have no interventions to avoid
approximating a model that omits their effects.

Pass the result to [`simulate`](@ref) for a [`MixingMeanFieldState`](@ref),
and that state back to [`simulate`](@ref), with a later `max_time`, to continue
the integration from where it stopped. See
[`MixingLinearNoiseProcess`](@ref) for the same dynamics with Gaussian noise
around the mean path, and [`HybridProcess`](@ref) for switching between
exact simulation of `spec` and this approximation.

# Example

```julia
using EpiBranch, Distributions
n = [2_000, 8_000]
band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
spec = ModelSpec(
    MixingProcess(; population_size = sum(n), mixing_by = (:age_band,),
        rate = (from, to) -> (from == to ? 0.3 : 0.1) / n[to[1]]);
    progression = [Transition(:recovered; rate = 0.2, terminal = true)],
    attributes = band)
model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
state = simulate(model; n_initial = 5, max_time = 60.0)
```
"""
struct MixingMeanFieldProcess{G, C}
    population_size::Int
    mixing_by::Tuple
    groups::Vector{G}
    group_sizes::Vector{Int}
    contact_matrix::Matrix{Float64}
    calendar::C
    removal_rate::Float64
    step::Float64
end

function MixingMeanFieldProcess(
        spec; removal_rate::Real, step::Real = 0.05,
        rng::AbstractRNG = Random.default_rng()
    )
    spec = _mixing_process_spec(spec, "MixingMeanFieldProcess")
    bare = process(spec)
    (isfinite(removal_rate) && removal_rate >= 0) ||
        throw(ArgumentError("removal_rate must be finite and non-negative"))
    step > 0 || throw(ArgumentError("step must be positive"))
    groups, sizes = _mixing_group_layout(spec, bare.mixing_by, rng)
    Λ = float.(_contact_matrix(bare.rate, groups))
    all(r -> isfinite(r) && r >= 0, Λ) ||
        throw(ArgumentError("mixing rates must be finite and non-negative"))
    return MixingMeanFieldProcess(
        sum(sizes), bare.mixing_by, groups, sizes, Λ, bare.calendar,
        Float64(removal_rate), Float64(step)
    )
end

population_size(m::MixingMeanFieldProcess) = m.population_size

function Base.show(io::IO, m::MixingMeanFieldProcess)
    return print(
        io,
        "MixingMeanFieldProcess(population_size=$(m.population_size), " *
            "groups=$(length(m.groups)), γ=$(m.removal_rate))"
    )
end

"""
    MixingMeanFieldState

The compartment counts of a [`MixingMeanFieldProcess`](@ref) at a point in
time: `S`, `I` and `R` hold one continuous count per mixing group, in the
same order as `groups`; `t` is the time (days since the start of the
outbreak) the state has been integrated to.
"""
struct MixingMeanFieldState{G}
    groups::Vector{G}
    S::Vector{Float64}
    I::Vector{Float64}
    R::Vector{Float64}
    t::Float64
end

function Base.show(io::IO, s::MixingMeanFieldState)
    return print(
        io,
        "MixingMeanFieldState(S=$(round(sum(s.S), digits = 1)), " *
            "I=$(round(sum(s.I), digits = 1)), R=$(round(sum(s.R), digits = 1)), " *
            "groups=$(length(s.groups)), t=$(s.t))"
    )
end

"""
    simulate(model::MixingMeanFieldProcess; n_initial = 1, max_time)

Integrate the mean-field ODE of `model` from `t = 0`, with `n_initial` people
infected overall, split across the mixing groups in proportion to their
size, up to `max_time`. Returns a [`MixingMeanFieldState`](@ref); continue it
further with `simulate(model, state; max_time)`.
"""
function simulate(model::MixingMeanFieldProcess; n_initial::Integer = 1, max_time::Real)
    1 <= n_initial <= model.population_size ||
        throw(ArgumentError("n_initial must be between 1 and population_size"))
    I0 = n_initial .* (model.group_sizes ./ model.population_size)
    S0 = Float64.(model.group_sizes) .- I0
    G = length(model.groups)
    state = MixingMeanFieldState(model.groups, S0, I0, zeros(G), 0.0)
    return simulate(model, state; max_time)
end

"""
    simulate(model::MixingMeanFieldProcess, state::MixingMeanFieldState; max_time)

Continue integrating `state`'s mean-field ODE up to `max_time`, which must
be later than `state.t`.
"""
function simulate(
        model::MixingMeanFieldProcess, state::MixingMeanFieldState; max_time::Real
    )
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    S, I = _integrate_mixing_sir(model, state.S, state.I, state.t, max_time)
    R = Float64.(model.group_sizes) .- S .- I
    return MixingMeanFieldState(model.groups, S, I, R, Float64(max_time))
end

# The per-group SIR drift: `λ` is each group's force of infection, from the
# contact matrix built when the model was constructed.
function _mixing_drift(S, I, Λ::AbstractMatrix, γ::Real, cal::Real)
    λ = cal .* (Λ * I)
    dS = -S .* λ
    dI = S .* λ .- γ .* I
    return dS, dI
end

# Fourth-order Runge–Kutta for the vector (S, I), with `drift(t, S, I)`
# evaluated at each of the step's four stage times so a calendar scaling the
# contact matrix is read at the time it actually applies to, rather than
# frozen at the step's start.
function _rk4_mixing_step(S, I, t::Real, dt::Real, drift)
    k1S, k1I = drift(t, S, I)
    k2S, k2I = drift(t + dt / 2, S + dt / 2 * k1S, I + dt / 2 * k1I)
    k3S, k3I = drift(t + dt / 2, S + dt / 2 * k2S, I + dt / 2 * k2I)
    k4S, k4I = drift(t + dt, S + dt * k3S, I + dt * k3I)
    Snew = S + dt / 6 * (k1S + 2k2S + 2k3S + k4S)
    Inew = I + dt / 6 * (k1I + 2k2I + 2k3I + k4I)
    return Snew, Inew
end

# Fixed-step RK4 from `t0` to `max_time`, in steps no longer than
# `model.step` (the last one shortened to end exactly at `max_time`),
# mirroring `_integrate_sir` for the scalar case.
function _integrate_mixing_sir(model::MixingMeanFieldProcess, S, I, t0, max_time)
    γ = model.removal_rate
    Λ = model.contact_matrix
    n_steps = max(1, ceil(Int, (max_time - t0) / model.step))
    dt = (max_time - t0) / n_steps
    drift = (t, s, i) -> _mixing_drift(s, i, Λ, γ, _calendar_value(model.calendar, t))
    t = t0
    for _ in 1:n_steps
        S, I = _rk4_mixing_step(S, I, t, dt, drift)
        t += dt
    end
    N = Float64.(model.group_sizes)
    S = clamp.(S, 0.0, N)
    I = clamp.(I, 0.0, N .- S)
    return S, I
end

"""
    MixingLinearNoiseProcess(spec::ModelSpec; removal_rate, step = 0.05,
                             rng = Random.default_rng())

The same multi-group SIR dynamics as [`MixingMeanFieldProcess`](@ref),
represented as a Gaussian process around the deterministic mean path (the
linear noise approximation): the state holds the mean compartment counts and
their covariance, which grows from the demographic noise of the
infection and removal reactions in every group and is damped by the
linearised dynamics around the mean.

Accepts the same keywords as [`MixingMeanFieldProcess`](@ref), built from the
same `ModelSpec`. The mean path integrates identically, with the
addition of the covariance.

# Example

```julia
using EpiBranch, Distributions
n = [2_000, 8_000]
band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
spec = ModelSpec(
    MixingProcess(; population_size = sum(n), mixing_by = (:age_band,),
        rate = (from, to) -> (from == to ? 0.3 : 0.1) / n[to[1]]);
    progression = [Transition(:recovered; rate = 0.2, terminal = true)],
    attributes = band)
model = MixingLinearNoiseProcess(spec; removal_rate = 0.2)
state = simulate(model; n_initial = 5, max_time = 60.0)
state.covariance   # 2|groups|×2|groups| covariance of (S_1, ..., I_1, ...)
```
"""
struct MixingLinearNoiseProcess{G, C}
    population_size::Int
    mixing_by::Tuple
    groups::Vector{G}
    group_sizes::Vector{Int}
    contact_matrix::Matrix{Float64}
    calendar::C
    removal_rate::Float64
    step::Float64
end

function MixingLinearNoiseProcess(
        spec; removal_rate::Real, step::Real = 0.05,
        rng::AbstractRNG = Random.default_rng()
    )
    mean_field = MixingMeanFieldProcess(spec; removal_rate, step, rng)
    return MixingLinearNoiseProcess(
        mean_field.population_size, mean_field.mixing_by, mean_field.groups,
        mean_field.group_sizes, mean_field.contact_matrix, mean_field.calendar,
        mean_field.removal_rate, mean_field.step
    )
end

population_size(m::MixingLinearNoiseProcess) = m.population_size

function Base.show(io::IO, m::MixingLinearNoiseProcess)
    return print(
        io,
        "MixingLinearNoiseProcess(population_size=$(m.population_size), " *
            "groups=$(length(m.groups)), γ=$(m.removal_rate))"
    )
end

"""
    MixingLinearNoiseState

The Gaussian state of a [`MixingLinearNoiseProcess`](@ref): mean compartment
counts `S`, `I` and `R` (one per mixing group), time `t`, and `covariance`,
the `2|groups| × 2|groups|` covariance of `(S_1, ..., S_g, I_1, ..., I_g)`
(`R` has no separate uncertainty, because each group's is
`group_sizes[g] - S[g] - I[g]`).
"""
struct MixingLinearNoiseState{G}
    groups::Vector{G}
    S::Vector{Float64}
    I::Vector{Float64}
    R::Vector{Float64}
    t::Float64
    covariance::Matrix{Float64}
end

function Base.show(io::IO, s::MixingLinearNoiseState)
    return print(
        io,
        "MixingLinearNoiseState(S=$(round(sum(s.S), digits = 1)), " *
            "I=$(round(sum(s.I), digits = 1)), R=$(round(sum(s.R), digits = 1)), " *
            "groups=$(length(s.groups)), t=$(s.t))"
    )
end

"""
    simulate(model::MixingLinearNoiseProcess; n_initial = 1, max_time)

Integrate the linear noise approximation of `model` from `t = 0`, with
`n_initial` people infected overall, split across the mixing groups in
proportion to their size, and no uncertainty yet (a zero covariance), up to
`max_time`. Returns a [`MixingLinearNoiseState`](@ref); continue it further
with `simulate(model, state; max_time)`.
"""
function simulate(model::MixingLinearNoiseProcess; n_initial::Integer = 1, max_time::Real)
    1 <= n_initial <= model.population_size ||
        throw(ArgumentError("n_initial must be between 1 and population_size"))
    I0 = n_initial .* (model.group_sizes ./ model.population_size)
    S0 = Float64.(model.group_sizes) .- I0
    G = length(model.groups)
    state = MixingLinearNoiseState(model.groups, S0, I0, zeros(G), 0.0, zeros(2G, 2G))
    return simulate(model, state; max_time)
end

"""
    simulate(model::MixingLinearNoiseProcess, state::MixingLinearNoiseState; max_time)

Continue `state`'s mean path and covariance up to `max_time`, which must be
later than `state.t`.
"""
function simulate(
        model::MixingLinearNoiseProcess, state::MixingLinearNoiseState; max_time::Real
    )
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    S, I, Σ = _integrate_mixing_lna(model, state.S, state.I, state.covariance, state.t, max_time)
    R = Float64.(model.group_sizes) .- S .- I
    return MixingLinearNoiseState(model.groups, S, I, R, Float64(max_time), Σ)
end

# The drift's Jacobian with respect to (S, I) (each a `G`-vector, so a
# `2G × 2G` block matrix), shared by the covariance update below.
function _mixing_jacobian(S, I, Λ::AbstractMatrix, γ::Real, cal::Real)
    G = length(S)
    λ = cal .* (Λ * I)
    A = cal .* (S .* Λ)   # A[g, h] = cal * S[g] * Λ[g, h]
    J = zeros(2G, 2G)
    J[1:G, 1:G] .= Diagonal(-λ)
    J[1:G, (G + 1):(2G)] .= -A
    J[(G + 1):(2G), 1:G] .= Diagonal(λ)
    J[(G + 1):(2G), (G + 1):(2G)] .= A - Diagonal(fill(γ, G))
    return J
end

# The Chemical Langevin diffusion for the `2G`-dimensional state `(S, I)`:
# every group's infection and removal reactions move only that group's own
# `(S_g, I_g)` pair, so each contributes the same `[1 -1; -1 1]` (infection)
# or `[0 0; 0 1]` (removal) block as the scalar case
# (`_SIR_INFECTION_OUTER`/`_SIR_REMOVAL_OUTER` in population_dynamics.jl), at
# that group's own place in the `2G × 2G` matrix. Group `g`'s infection
# reactions are driven by every group `h`'s infectious count, but they all
# share group `g`'s stoichiometry, so their propensities sum before scaling
# it rather than needing one outer product per `(g, h)` pair.
function _mixing_diffusion(S, I, Λ::AbstractMatrix, γ::Real, cal::Real)
    G = length(S)
    λ = cal .* (Λ * I)
    a = S .* λ   # each group's total infection propensity, summed over h
    b = γ .* I   # each group's removal propensity
    Q = zeros(2G, 2G)
    for g in 1:G
        Q[g, g] += a[g]
        Q[g, G + g] -= a[g]
        Q[G + g, g] -= a[g]
        Q[G + g, G + g] += a[g] + b[g]
    end
    return Q
end

# Mean (RK4) and covariance (explicit Euler, evaluating the Jacobian and
# diffusion at the step's start) from `t0` to `max_time`, in steps no longer
# than `model.step`, mirroring `_integrate_lna` for the scalar case.
function _integrate_mixing_lna(model::MixingLinearNoiseProcess, S, I, Σ, t0, max_time)
    γ = model.removal_rate
    Λ = model.contact_matrix
    n_steps = max(1, ceil(Int, (max_time - t0) / model.step))
    dt = (max_time - t0) / n_steps
    drift = (t, s, i) -> _mixing_drift(s, i, Λ, γ, _calendar_value(model.calendar, t))
    t = t0
    for _ in 1:n_steps
        cal = _calendar_value(model.calendar, t)
        J = _mixing_jacobian(S, I, Λ, γ, cal)
        Q = _mixing_diffusion(S, I, Λ, γ, cal)
        Σ = Σ + dt * (J * Σ + Σ * J' + Q)
        Σ = (Σ + Σ') / 2   # undo floating-point asymmetry from the update above
        S, I = _rk4_mixing_step(S, I, t, dt, drift)
        t += dt
    end
    N = Float64.(model.group_sizes)
    S = clamp.(S, 0.0, N)
    I = clamp.(I, 0.0, N .- S)
    return S, I, Σ
end
