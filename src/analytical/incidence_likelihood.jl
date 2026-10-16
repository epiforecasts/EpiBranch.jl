# ── Incidence time-series likelihood ─────────────────────────────────
#
# A closed-form likelihood for the incidence time series a branching process
# produces, in the spirit of Boudreau and St-Onge's calendar-time branching
# process (arXiv:2610.17138): advance the probability generating function one
# time step at a time, carrying one auxiliary variable per step, then read
# the mean and covariance of the step counts off the cumulant generating
# function (the log of that PGF) and match a multivariate normal or
# lognormal to them.
#
# One individual infected at step k (1 ≤ k ≤ n, the last observed step)
# contributes u_k for itself, plus the PGF of its offspring's own
# contributions, each landing at step k + j with probability w[j] from the
# discretised generation-time distribution:
#
#   Ψ_k(u) = u_k · f(Σ_j w[j] · Ψ_{k+j}(u)),   Ψ_m(u) ≡ 1 for m > n
#
# (`Ψ_m ≡ 1` beyond the horizon marginalises out what is not observed.) The
# `seeds` index cases at step 0 are not themselves observed, so their shared
# PGF drops the `u_0` factor, Ψ_0(u) = f(Σ_j w[j] · Ψ_j(u)), and because they
# act independently the whole series has cumulant generating function
# K(u) = log G(u) = seeds · log Ψ_0(u). Differentiating the composition at
# u = 1 gives the mean and covariance of the step counts:
#
#   mean[t]        = ∂K/∂u_t
#   cov[t, t']     = ∂²K/∂u_t∂u_t' + δ_tt' · mean[t]
#
# the usual bridge from a probability generating function's factorial
# moments to ordinary ones. `ForwardDiff` takes those derivatives; `K` is
# written generically in its argument's element type so a `Dual` offspring
# or generation-time parameter still differentiates (nested automatic
# differentiation), the same contract `_chain_length_ll_pgf` keeps for the
# chain-length likelihood.

# Discretised generation-time pmf over lag 1, 2, ..., binning the continuous
# `generation_time` distribution into day-length (or whatever `data`'s unit
# is) steps: `w[j]` is the probability mass in `(j - 1, j]`. Stops once the
# remaining tail is below `tol`, then renormalises away that negligible
# truncation. Mass at or below 0 is dropped, which assumes (as every
# generation-time distribution used elsewhere in the package does) that it is
# continuous with no atom there.
function _discretize_generation_interval(
        generation_time::Distribution; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    w = Float64[]
    c_prev = cdf(generation_time, 0)
    j = 1
    while true
        c = cdf(generation_time, j)
        push!(w, c - c_prev)
        c_prev = c
        if 1 - c < tol
            break
        elseif j >= max_lag
            @warn "The generation-time distribution has more than $tol of its mass " *
                "above a lag of $max_lag steps, so the discretised pmf is truncated " *
                "there." maxlog = 1
            break
        end
        j += 1
    end
    s = sum(w)
    s > 0 || throw(ArgumentError("generation_time has no mass on positive lags"))
    return w ./ s
end

# The cumulant generating function K(u) = log G(u) of the incidence counts at
# steps 1:n, for `seeds` index cases at step 0, an offspring law read through
# `pgf` and a discretised generation-time pmf `w`. Backward recursion in `k`
# so that `Ψ[m]` for `m > k` is already known. `Toff` is the offspring law's
# own element type (`Distributions.partype`, `Float64` for a
# `GeneratingFunctionOffspring`): `Ψ`'s element type promotes `eltype(u)`
# with it rather than taking `eltype(u)` alone, so a `Dual` offspring
# parameter composes correctly even while `u` is still the plain `Float64`
# ForwardDiff probes before perturbing it (nested automatic differentiation,
# as a `Dual` offspring under an outer AD call over R or k gives).
function _incidence_log_pgf(
        u::AbstractVector{T}, w::AbstractVector{<:Real}, pgf, n::Integer, seeds::Real,
        ::Type{Toff}
    ) where {T <: Real, Toff <: Real}
    S = promote_type(T, Toff)
    Ψ = Vector{S}(undef, n)
    for k in n:-1:1
        a = zero(S)
        for j in 1:length(w)
            m = k + j
            a += w[j] * (m > n ? one(S) : Ψ[m])
        end
        Ψ[k] = u[k] * pgf(a)
    end
    a0 = zero(S)
    for j in 1:length(w)
        a0 += w[j] * (j > n ? one(S) : Ψ[j])
    end
    return seeds * log(pgf(a0))
end

# The offspring law's own numeric element type, promoted with `eltype(u)` in
# `_incidence_log_pgf` above. `Distributions.partype` gives this for any
# `Distribution`; a `GeneratingFunctionOffspring`'s `pgf` closure is opaque, so
# its own parameters (if any are duals under AD) must already return that
# type consistently and `Float64` is assumed here.
_offspring_partype(offspring::Distribution) = Distributions.partype(offspring)
_offspring_partype(::GeneratingFunctionOffspring) = Float64

"""
    incidence_moments(offspring, generation_time::Distribution, n::Integer; seeds = 1)

Mean vector and covariance matrix of the incidence counts a branching process
produces at steps `1:n`, started by `seeds` index cases at step 0, with
offspring law `offspring` (a `Poisson`, `NegativeBinomial`,
[`GeneratingFunctionOffspring`](@ref), or any other discrete distribution
`EpiBranch` knows the generating function of) and a `generation_time`
distribution giving the time from one case's infection to each of its
secondary infections, in the same unit as a step of `n` (so a weekly series
needs a generation time in weeks).

Returns a `(mean, cov)` named tuple: `mean[t]` is the expected count at step
`t`, and `cov` is their `n × n` covariance matrix, both ignoring susceptible
depletion (see [`loglikelihood`](@ref) for the likelihood this feeds and its
limits). Matching a `MvNormal(mean, cov)`
or a moment-matched lognormal to these is what the `NormalApproximation`
and `LognormalApproximation` of that likelihood do; call this directly for
a forecast instead, or to compare against simulated incidence (`seeds`
simulated introductions, summed into steps the same way `data` is).

# Examples

```julia
m = incidence_moments(NegBin(1.3, 0.5), LogNormal(1.6, 0.5), 20; seeds = 3)
m.mean[end]
sqrt(m.cov[end, end])  # standard deviation of the last step's count
```
"""
function incidence_moments(
        offspring::Union{DiscreteUnivariateDistribution, GeneratingFunctionOffspring},
        generation_time::Distribution, n::Integer;
        seeds::Real = 1, tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    seeds > 0 || throw(ArgumentError("seeds must be positive"))
    w = _discretize_generation_interval(generation_time; tol, max_lag)
    Toff = _offspring_partype(offspring)
    K(u) = _incidence_log_pgf(u, w, s -> _pgf(offspring, s), n, seeds, Toff)
    u1 = ones(Float64, n)
    μ = ForwardDiff.gradient(K, u1)
    H = ForwardDiff.hessian(K, u1)
    return (mean = μ, cov = Symmetric(H + Diagonal(μ)))
end

# ── Matching a distribution to the moments ────────────────────────────

"""
    IncidenceApproximation

How [`loglikelihood(::IncidenceCounts, ::Distribution)`](@ref) turns the
mean and covariance from [`incidence_moments`](@ref) into a likelihood:
[`NormalApproximation`](@ref) matches a multivariate normal, and
[`LognormalApproximation`](@ref) a multivariate lognormal. Dispatch on a new
subtype to match something else.
"""
abstract type IncidenceApproximation end

"""
    NormalApproximation()

Match a `MvNormal` with the mean and covariance from [`incidence_moments`](@ref).
Defined for any observed counts, including zeros; see
[`LognormalApproximation`](@ref) for the alternative that respects
non-negativity.
"""
struct NormalApproximation <: IncidenceApproximation end

"""
    LognormalApproximation()

Match a multivariate lognormal with the mean and covariance from
[`incidence_moments`](@ref): the log-counts are matched to a `MvNormal` by
the usual lognormal moment equations, `logpdf` adding the log-Jacobian
`-Σ log(data)` for the change of variables. Needs every observed count to be
strictly positive, since a lognormal puts no mass at 0; use
[`NormalApproximation`](@ref) for data that includes zeros.
"""
struct LognormalApproximation <: IncidenceApproximation end

function _incidence_loglik(::NormalApproximation, data::AbstractVector{<:Real}, μ, Σ)
    return logpdf(MvNormal(μ, Σ), data)
end

function _incidence_loglik(::LognormalApproximation, data::AbstractVector{<:Real}, μ, Σ)
    all(>(0), data) || throw(
        ArgumentError(
            "LognormalApproximation needs every observed count to be strictly positive " *
                "(a lognormal puts no mass at 0); use NormalApproximation for data with zeros."
        )
    )
    n = length(μ)
    σ²_log = [log1p(Σ[t, t] / μ[t]^2) for t in 1:n]
    μ_log = [log(μ[t]) - σ²_log[t] / 2 for t in 1:n]
    Σ_log = Symmetric(
        [
            t == t′ ? σ²_log[t] : log1p(Σ[t, t′] / (μ[t] * μ[t′]))
                for t in 1:n, t′ in 1:n
        ]
    )
    return logpdf(MvNormal(μ_log, Σ_log), log.(data)) - sum(log, data)
end

# ── Unified loglikelihood interface ────────────────────────────────────

"""
    loglikelihood(data::IncidenceCounts, offspring, generation_time::Distribution,
        approximation::IncidenceApproximation = NormalApproximation())

Closed-form, moment-matched log-likelihood of an incidence time series under
a branching process with offspring law `offspring` and a `generation_time`
distribution, both as in [`incidence_moments`](@ref). `approximation`
chooses what [`incidence_moments`](@ref)'s mean and covariance are matched
to; see [`NormalApproximation`](@ref) and [`LognormalApproximation`](@ref).

!!! warning "Limits"
    This ignores susceptible depletion, so it holds only while the
    population is effectively unsaturated. It models no interventions and
    no under-reporting, and the `generation_time` distribution is
    discretised onto `data`'s own time step by binning it at each step's
    right edge: accurate when the step is short relative to how spread out
    `generation_time` is, and biased when it is not (a day step against a
    generation time whose own spread is only a few days, say). The
    likelihood itself is approximate, a moment-matched normal or lognormal
    standing in for the true (unknown in closed form) distribution of the
    counts; prediction intervals built from it can be too narrow,
    particularly once the process is not far short of depletion.

# Examples

```julia
data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18])
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5))
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5), LognormalApproximation())
```
"""
function loglikelihood(
        data::IncidenceCounts,
        offspring::Union{DiscreteUnivariateDistribution, GeneratingFunctionOffspring},
        generation_time::Distribution,
        approximation::IncidenceApproximation = NormalApproximation();
        tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    μ, Σ = incidence_moments(
        offspring, generation_time, length(data.data);
        seeds = data.seeds, tol, max_lag
    )
    return _incidence_loglik(approximation, data.data, μ, Σ)
end

# Refuse under-reporting from either source `simulate` reads it from: an
# observation model, or a `Reporting` progression step (`_reporting_probability`
# is the same helper `_eoo_assert_full_reporting` reads for the end-of-outbreak
# probability). Thinning the moments correctly needs a binomial correction this
# likelihood does not yet apply.
_incidence_assert_full_reporting(::NoObservation, progression) = _incidence_assert_reporting_rate(
    _reporting_probability(progression)
)
function _incidence_assert_full_reporting(obs, progression)
    throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model with $(typeof(obs))) is not " *
                "implemented: thinning the moments for under-reporting is not yet " *
                "supported. Use a model with no observation, or call loglikelihood with " *
                "the offspring and generation-time distributions directly."
        )
    )
end
_incidence_assert_reporting_rate(::Nothing) = nothing
function _incidence_assert_reporting_rate(ρ::Real)
    ρ >= 1.0 && return nothing
    throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model with a Reporting step at probability " *
                "$ρ) is not implemented: thinning the moments for under-reporting is not " *
                "yet supported. Use a model with no Reporting step, or call loglikelihood " *
                "with the offspring and generation-time distributions directly."
        )
    )
end

"""
    loglikelihood(data::IncidenceCounts, model::Union{BranchingProcess, ModelSpec},
        approximation::IncidenceApproximation = NormalApproximation())

Log-likelihood of an incidence time series under a model's own offspring and
generation-time distributions. Only defined for a model with no
interventions, a single infectiousness window and full reporting (every
case counted, whether under-reporting would otherwise come from an
observation model or a `Reporting` progression step); each of those
reshapes transmission, detection or timing in a way this closed form does
not represent, and throws an `ArgumentError` naming the reason instead of a
silent approximation. Simulate [`weekly_incidence`](@ref) (or daily, by
passing a weekly `condition`) and compare empirically for a model outside
these limits.
"""
function loglikelihood(
        data::IncidenceCounts, model::Union{BranchingProcess, ModelSpec},
        approximation::IncidenceApproximation = NormalApproximation(); kwargs...
    )
    isempty(interventions(model)) || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model with interventions) has no closed form: " *
                "interventions reshape transmission over time in ways this likelihood does " *
                "not represent. Simulate instead."
        )
    )
    _incidence_assert_full_reporting(observation(model), _progression(model))
    gt = _single_kernel(model)
    gt isa NoGenerationTime && throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, ...) needs a generation-time distribution; " *
                "this model's kernel is NoGenerationTime(). Build the BranchingProcess with " *
                "one, e.g. BranchingProcess(offspring, LogNormal(1.6, 0.5))."
        )
    )
    gt isa Distribution || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, ...) needs a fixed generation-time distribution, " *
                "not a per-case function; call loglikelihood with the offspring and " *
                "generation-time distributions directly instead."
        )
    )
    return loglikelihood(data, single_type_offspring(model), gt, approximation; kwargs...)
end
