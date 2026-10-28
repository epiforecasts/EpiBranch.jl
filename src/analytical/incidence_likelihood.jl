# ── Incidence time-series likelihood ─────────────────────────────────
#
# A closed-form likelihood for the incidence time series a branching process
# produces, in the spirit of Boudreau and St-Onge's calendar-time branching
# process (arXiv:2610.17138): advance the probability generating function one
# time step at a time, with one auxiliary variable per step, then read
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
# K(u) = log G(u) = seeds · log Ψ_0(u).
#
# Under-reporting replaces the `u_k` an individual contributes for itself
# with its report: none with probability 1 - ρ, otherwise one at step k + l
# with probability d[l] from the discretised infection-to-report delay,
#
#   r_k(u) = 1 - ρ + ρ · Σ_l d[l] · U_{k+l},   U_m = u_m for 1 ≤ m ≤ n, else 1,
#
# so Ψ_k(u) = r_k(u) · f(Σ_j w[j] · Ψ_{k+j}(u)) and Ψ_0 takes the factor
# r_0 too: a seed's own report, delayed into an observed step, is counted
# like anyone else's. With ρ = 1 and no delay, r_k = u_k and r_0 = 1, the
# recursion above. Differentiating the composition at u = 1 gives the mean
# and covariance of the step counts:
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

# Discretised infection-to-report delay pmf over lag 0, 1, 2, ...: `d[1]` is
# the mass at or below 0 (a report in the step of infection), and `d[l + 1]`
# the mass in `(l - 1, l]`, the same binning as the generation interval's.
# `nothing` reports every case in the step it is infected, and a number is a
# fixed delay of that many steps.
_discretize_report_delay(::Nothing; kwargs...) = [1.0]
_discretize_report_delay(delay::Real; kwargs...) = _discretize_report_delay(Dirac(delay); kwargs...)
function _discretize_report_delay(
        delay::Distribution; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    minimum(delay) >= 0 || throw(ArgumentError("reporting_delay must be non-negative"))
    return _bin_delay(t -> cdf(delay, t), tol, max_lag)
end

# The sum of two independent delays, such as an incubation period and the
# delay from onset to report, which a model's `Reporting` step gives.
struct _DelaySum{A <: Distribution, B <: Union{Real, Distribution}}
    first::A
    second::B
end

function _discretize_report_delay(
        delay::_DelaySum; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    a, b = delay.first, delay.second
    minimum(a) >= 0 && (b isa Real ? b >= 0 : minimum(b) >= 0) ||
        throw(ArgumentError("reporting delays must be non-negative"))
    b isa Real && return _bin_delay(t -> cdf(a, t - b), tol, max_lag)
    # P(A + B ≤ t) = ∫ f_A(x) F_B(t - x) dx over x ≤ t, B being non-negative.
    lo = minimum(a)
    F(t) = t <= lo ? 0.0 : first(quadgk(x -> pdf(a, x) * cdf(b, t - x), lo, t))
    return _bin_delay(F, tol, max_lag)
end

# Bin a delay with cumulative distribution `F` onto lags 0, 1, 2, ...: the
# mass at or below 0, then each `(l - 1, l]`, until the remaining tail is
# below `tol`.
function _bin_delay(F, tol::Real, max_lag::Int)
    d = [F(0)]
    l = 1
    while 1 - F(l - 1) >= tol
        if l > max_lag
            @warn "The reporting delay has more than $tol of its mass above a lag of " *
                "$max_lag steps, so the discretised pmf is truncated there." maxlog = 1
            break
        end
        push!(d, F(l) - F(l - 1))
        l += 1
    end
    return d ./ sum(d)
end

# The cumulant generating function K(u) = log G(u) of the reported counts at
# steps 1:n, for `seeds` index cases at step 0, an offspring law read through
# `pgf`, a discretised generation-time pmf `w`, a reporting probability `ρ`
# and a discretised infection-to-report delay pmf `d` (lag 0 first). Backward
# recursion in `k` so that `Ψ[m]` for `m > k` is already known. `Toff` is the
# offspring law's own element type (`Distributions.partype`, `Float64` for a
# `GeneratingFunctionOffspring`): `Ψ`'s element type promotes `eltype(u)`
# with it, and with `ρ`'s, rather than taking `eltype(u)` alone, so a `Dual`
# offspring or reporting parameter composes correctly even while `u` is
# still the plain `Float64` ForwardDiff probes before perturbing it (nested
# automatic differentiation, as a `Dual` offspring under an outer AD call
# over R or k gives).
function _incidence_log_pgf(
        u::AbstractVector{T}, w::AbstractVector{<:Real}, pgf, n::Integer, seeds::Real,
        ::Type{Toff}, ρ::Real = 1, d::AbstractVector{<:Real} = [1.0]
    ) where {T <: Real, Toff <: Real}
    S = promote_type(T, Toff, typeof(ρ))
    U(m) = 1 <= m <= n ? S(u[m]) : one(S)
    function report(k)
        r = zero(S)
        for l in 0:(length(d) - 1)
            r += d[l + 1] * U(k + l)
        end
        return 1 - ρ + ρ * r
    end
    Ψ = Vector{S}(undef, n)
    for k in n:-1:1
        a = zero(S)
        for j in 1:length(w)
            m = k + j
            a += w[j] * (m > n ? one(S) : Ψ[m])
        end
        Ψ[k] = report(k) * pgf(a)
    end
    a0 = zero(S)
    for j in 1:length(w)
        a0 += w[j] * (j > n ? one(S) : Ψ[j])
    end
    return seeds * log(report(0) * pgf(a0))
end

# The offspring law's own numeric element type, promoted with `eltype(u)` in
# `_incidence_log_pgf` above. `Distributions.partype` gives this for any
# `Distribution`; a `GeneratingFunctionOffspring`'s `pgf` closure is opaque, so
# its own parameters (if any are duals under AD) must already return that
# type consistently and `Float64` is assumed here.
_offspring_partype(offspring::Distribution) = Distributions.partype(offspring)
_offspring_partype(::GeneratingFunctionOffspring) = Float64

"""
    incidence_moments(offspring, generation_time::Distribution, n::Integer; seeds = 1,
        reporting_probability = 1, reporting_delay = nothing)

Mean vector and covariance matrix of the incidence counts a branching process
produces at steps `1:n`, started by `seeds` index cases at step 0, with
offspring law `offspring` (a `Poisson`, `NegativeBinomial`,
[`GeneratingFunctionOffspring`](@ref), or any other discrete distribution
`EpiBranch` knows the generating function of) and a `generation_time`
distribution giving the time from one case's infection to each of its
secondary infections, in the same unit as a step of `n` (so a weekly series
needs a generation time in weeks).

With `reporting_probability` below 1, each case is reported independently
with that probability, and the counts are of reported cases. A
`reporting_delay` (a distribution or a fixed number of steps, measured from
infection, in the same unit) dates each report that many steps after the
infection, so the counts are by report date; the seeds' own reports count
too once a delay moves them into an observed step. Both default to every
case counted in the step it is infected. The result is exact for
independent reporting, and `reporting_probability` can be differentiated
like the offspring parameters.

Returns a `(mean, cov)` named tuple: `mean[t]` is the expected count at step
`t`, and `cov` is their `n × n` covariance matrix, both ignoring susceptible
depletion (see `loglikelihood` for the likelihood this feeds and its
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
        seeds::Real = 1, reporting_probability::Real = 1, reporting_delay = nothing,
        tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    seeds > 0 || throw(ArgumentError("seeds must be positive"))
    0 < reporting_probability <= 1 ||
        throw(ArgumentError("reporting_probability must be in (0, 1]"))
    w = _discretize_generation_interval(generation_time; tol, max_lag)
    d = _discretize_report_delay(reporting_delay; tol, max_lag)
    Toff = _offspring_partype(offspring)
    K(u) = _incidence_log_pgf(
        u, w, s -> _pgf(offspring, s), n, seeds, Toff, reporting_probability, d
    )
    u1 = ones(Float64, n)
    μ = ForwardDiff.gradient(K, u1)
    H = ForwardDiff.hessian(K, u1)
    return (mean = μ, cov = Symmetric(H + Diagonal(μ)))
end

# ── Matching a distribution to the moments ────────────────────────────

"""
    IncidenceApproximation

How `loglikelihood(::IncidenceCounts, ::Distribution)` turns the
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

# A count that can only be zero, such as a type with no seeds in the first
# step, or any count before a delayed report can arrive, has zero mean and
# variance, which no approximation can be matched to. It is left out, and the
# data are impossible unless they are zero there too.
function _incidence_loglik_observed(approximation, data, μ, Σ)
    keep = [Σ[t, t] > 0 for t in eachindex(data)]
    all(keep) && return _incidence_loglik(approximation, data, μ, Σ)
    T = float(eltype(μ))
    any(t -> !keep[t] && data[t] != 0, eachindex(data)) && return T(-Inf)
    any(keep) || return zero(T)
    return _incidence_loglik(
        approximation, data[keep], μ[keep], Symmetric(Matrix(Σ)[keep, keep])
    )
end

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
        approximation::IncidenceApproximation = NormalApproximation();
        reporting_probability = 1, reporting_delay = nothing)

Closed-form, moment-matched log-likelihood of an incidence time series under
a branching process with offspring law `offspring` and a `generation_time`
distribution, both as in [`incidence_moments`](@ref), as are
`reporting_probability` and `reporting_delay` for a series of reported
cases. `approximation` chooses what [`incidence_moments`](@ref)'s mean and
covariance are matched to; see [`NormalApproximation`](@ref) and
[`LognormalApproximation`](@ref).

!!! warning "Limits"
    This ignores susceptible depletion, so it holds only while the
    population is effectively unsaturated. It models no interventions, and
    under-reporting only as each case reported independently with the same
    probability. The `generation_time` distribution and `reporting_delay`
    are discretised onto `data`'s own time step by binning them at each step's
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
# 60% of cases reported, on average 3 days after infection
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5);
    reporting_probability = 0.6, reporting_delay = Gamma(2.0, 1.5))
```
"""
function loglikelihood(
        data::IncidenceCounts,
        offspring::Union{DiscreteUnivariateDistribution, GeneratingFunctionOffspring},
        generation_time::Distribution,
        approximation::IncidenceApproximation = NormalApproximation();
        reporting_probability::Real = 1, reporting_delay = nothing,
        tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    y, seeds = _single_type_series(data)
    μ, Σ = incidence_moments(
        offspring, generation_time, length(y);
        seeds, reporting_probability, reporting_delay, tol, max_lag
    )
    return _incidence_loglik_observed(approximation, y, μ, Σ)
end

_single_type_series(data::IncidenceCounts{Vector{Int}, Int}) = (data.data, data.seeds)
function _single_type_series(::IncidenceCounts)
    throw(
        ArgumentError(
            "counts or seeds by type need a multi-type offspring law " *
                "(MultiTypeOffspring); a single-type law takes a vector of counts and " *
                "a single number of seeds"
        )
    )
end

# ── Multi-type processes ──────────────────────────────────────────────
#
# A type-j case draws its total offspring from G_j = dist_fn(R_j) and gives
# each child type i with probability a[i, j] (see `multi_type.jl`), and each
# child independently lands w[l] steps later. Each type and step has its own
# auxiliary variable, so with B_i(k) = Σ_l w[l] · Ψ_{k+l, i},
#
#   Ψ_{k, j}(u) = r_{k, j}(u) · G_j(Σ_i a[i, j] · B_i(k)),
#
# r as in the single-type recursion but reading type j's own variables, and
# K(u) = Σ_j seeds[j] · log Ψ_{0, j}(u). The variables are ordered step
# first, then type, as `vec` orders an n × m matrix.
function _incidence_log_pgf(
        u::AbstractVector{T}, w::AbstractVector{<:Real}, o::MultiTypeOffspring,
        n::Integer, seeds::AbstractVector{<:Real}, ρ::Real, d::AbstractVector{<:Real}
    ) where {T <: Real}
    m = _n_types(o)
    laws = _total_count_laws(o)
    S = promote_type(
        T, eltype(o.alloc_probs), eltype(o.R_by_type), typeof(ρ),
        map(Distributions.partype, laws)...
    )
    U(k, i) = 1 <= k <= n ? S(u[k + (i - 1) * n]) : one(S)
    function report(k, i)
        r = zero(S)
        for l in 0:(length(d) - 1)
            r += d[l + 1] * U(k + l, i)
        end
        return 1 - ρ + ρ * r
    end
    Ψ = Matrix{S}(undef, n, m)
    B = Vector{S}(undef, m)
    function next_generation!(k)
        for i in 1:m
            b = zero(S)
            for l in 1:length(w)
                b += w[l] * (k + l > n ? one(S) : Ψ[k + l, i])
            end
            B[i] = b
        end
        return B
    end
    child(j) = sum(o.alloc_probs[i, j] * B[i] for i in 1:m)
    for k in n:-1:1
        next_generation!(k)
        for j in 1:m
            Ψ[k, j] = report(k, j) * _pgf(laws[j], child(j))
        end
    end
    next_generation!(0)
    K = zero(S)
    for j in 1:m
        iszero(seeds[j]) && continue
        K += seeds[j] * log(report(0, j) * _pgf(laws[j], child(j)))
    end
    return K
end

"""
    incidence_moments(offspring::MultiTypeOffspring, generation_time::Distribution,
        n::Integer; seeds, reporting_probability = 1, reporting_delay = nothing)

Mean and covariance of the incidence counts of each type of a multi-type
branching process at steps `1:n`, started by `seeds[j]` index cases of type
`j` at step 0. `generation_time` is shared by every type, and
`reporting_probability` and `reporting_delay` act as in the single-type
method.

Returns a `(mean, cov)` named tuple: `mean[t, j]` is the expected count of
type `j` at step `t`, and `cov` is the covariance matrix of `vec(mean)`'s
entries, steps first and then types, so `cov[t + (j - 1) * n, t′ + (j′ - 1) * n]`
is the covariance between type `j` at step `t` and type `j′` at step `t′`.

# Examples

```julia
offspring = EpiBranch.MultiTypeOffspring([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5))
m = incidence_moments(offspring, LogNormal(1.6, 0.5), 20; seeds = [0, 1])
m.mean[end, :]  # expected count of each type on the last step
```
"""
function incidence_moments(
        offspring::MultiTypeOffspring, generation_time::Distribution, n::Integer;
        seeds::AbstractVector{<:Real}, reporting_probability::Real = 1,
        reporting_delay = nothing, tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    m = _n_types(offspring)
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    length(seeds) == m || throw(
        ArgumentError("seeds needs one entry per type ($m), got $(length(seeds))")
    )
    all(>=(0), seeds) && sum(seeds) > 0 ||
        throw(ArgumentError("seeds must be ≥ 0 and not all zero"))
    0 < reporting_probability <= 1 ||
        throw(ArgumentError("reporting_probability must be in (0, 1]"))
    w = _discretize_generation_interval(generation_time; tol, max_lag)
    d = _discretize_report_delay(reporting_delay; tol, max_lag)
    K(u) = _incidence_log_pgf(u, w, offspring, n, seeds, reporting_probability, d)
    u1 = ones(Float64, n * m)
    μ = ForwardDiff.gradient(K, u1)
    H = ForwardDiff.hessian(K, u1)
    return (mean = reshape(μ, n, m), cov = Symmetric(H + Diagonal(μ)))
end

"""
    loglikelihood(data::IncidenceCounts, offspring::MultiTypeOffspring,
        generation_time::Distribution,
        approximation::IncidenceApproximation = NormalApproximation();
        reporting_probability = 1, reporting_delay = nothing)

Log-likelihood of an incidence time series under a multi-type branching
process, from the moments of [`incidence_moments`](@ref). `data` holds the
counts by type, one column per type, or their total over types as a vector;
its `seeds` give the index cases of each type. The limits are those of the
single-type method.

# Examples

```julia
offspring = EpiBranch.MultiTypeOffspring([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5))
gt = LogNormal(1.6, 0.5)
loglikelihood(IncidenceCounts([0 1; 1 1; 1 3; 2 4]; seeds = [0, 1]), offspring, gt)
loglikelihood(IncidenceCounts([1, 2, 4, 6]; seeds = [0, 1]), offspring, gt)
```
"""
function loglikelihood(
        data::IncidenceCounts, offspring::MultiTypeOffspring,
        generation_time::Distribution,
        approximation::IncidenceApproximation = NormalApproximation();
        reporting_probability::Real = 1, reporting_delay = nothing,
        tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    data.seeds isa AbstractVector || throw(
        ArgumentError(
            "a multi-type process needs seeds by type: IncidenceCounts(data; seeds = " *
                "[...]) with one entry per type"
        )
    )
    n = size(data.data, 1)
    μ, Σ = incidence_moments(
        offspring, generation_time, n;
        seeds = data.seeds, reporting_probability, reporting_delay, tol, max_lag
    )
    y, μ_obs, Σ_obs = _observed_moments(data.data, μ, Σ)
    return _incidence_loglik_observed(approximation, y, μ_obs, Σ_obs)
end

# The observed vector and its moments: the counts by type as they are, or
# their total over types, whose mean and covariance sum the per-type blocks.
function _observed_moments(data::AbstractMatrix, μ, Σ)
    size(data) == size(μ) || throw(
        ArgumentError(
            "counts by type have $(size(data, 2)) columns, but the process has " *
                "$(size(μ, 2)) types"
        )
    )
    return vec(data), vec(μ), Σ
end
function _observed_moments(data::AbstractVector, μ, Σ)
    n, m = size(μ)
    block(i) = ((i - 1) * n + 1):(i * n)
    Σ_total = sum(Σ[block(i), block(i′)] for i in 1:m, i′ in 1:m)
    return data, vec(sum(μ; dims = 2)), Symmetric(Σ_total)
end

# A model's under-reporting as the direct method's keywords. An observation
# model is refused. A `Reporting` step becomes `reporting_probability` and an
# infection-to-report `reporting_delay`: its own probability and delay when it
# reports from infection, and, from onset, the delay added to the incubation
# period and the probability scaled by the share of cases that reach onset.
# The step must be the model's only one: an outcome or another transition can
# end a case before its report, or set the onset it reports from, and neither
# keeps reporting independent of the rest of the case.
function _incidence_reporting(model)
    obs = observation(model)
    obs isa NoObservation || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model with $(typeof(obs))) is not " *
                "implemented; report cases with a Reporting step instead."
        )
    )
    prog = progression(model)
    _reporting_probability(prog) === nothing && return (;)
    length(prog) == 1 || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) supports a Reporting step only as " *
                "the model's whole progression: another step can end a case before its " *
                "report or set the time it reports from, and the closed form assumes " *
                "each case is reported independently. Call loglikelihood with the " *
                "offspring and generation-time distributions directly, passing " *
                "`reporting_probability` and `reporting_delay` measured from infection."
        )
    )
    r = only(prog)
    r.delay isa Union{Real, Distribution} || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) needs the Reporting delay to be a " *
                "number or a distribution, not a per-case function"
        )
    )
    reached, to_anchor = _infection_to(r.from, attributes(model))
    return (
        reporting_probability = _reporting_probability(prog) * reached,
        reporting_delay = to_anchor === nothing ? r.delay : _DelaySum(to_anchor, r.delay),
    )
end

# The share of cases that reach the time a `Reporting` step measures its delay
# from, and that time's delay after infection (`nothing` for infection itself).
function _infection_to(from::Symbol, attrs)
    from === :infection && return (1.0, nothing)
    _resolved_key(from) === :onset_time && return _infection_to_onset(attrs)
    throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) supports a Reporting step from " *
                "infection or onset, not from `:$from`"
        )
    )
end
function _infection_to(from, attrs)
    throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) supports a Reporting step from " *
                "infection or onset, not from a function"
        )
    )
end

# Onset from the model's `clinical_presentation`, its incubation period and
# asymptomatic share. The builders are run once, in order, on a probe case, as
# `simulate` runs them, to check that no other one records an onset of its own.
function _infection_to_onset(attrs)
    builders = attrs isa NoAttributes ? () :
        attrs isa Union{Tuple, AbstractVector} ? Tuple(attrs) : (attrs,)
    presentations = filter(b -> b isa ClinicalPresentation, builders)
    length(presentations) == 1 || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) reads the onset a Reporting step " *
                "reports from from the model's clinical_presentation, and needs exactly " *
                "one; this model has $(length(presentations))"
        )
    )
    # Fresh copies, so the probe leaves the model's own builders as they were.
    _check_onset_builders(_fresh_attributes(builders))
    cp = only(presentations)
    cp.prob_asymptomatic isa Real || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) needs a fixed prob_asymptomatic, not " *
                "a per-case function"
        )
    )
    return (1 - cp.prob_asymptomatic, cp.incubation_period)
end

function _check_onset_builders(builders)
    probe = Individual(id = 1, infection_time = 0.0)
    rng = Random.Xoshiro(1)
    onset(ind) = (get(ind.state, :onset_time, nothing), get(ind.state, :incubation_period, nothing))
    for b in builders
        before = onset(probe)
        try
            _apply_attributes!(b, rng, probe)
        catch err
            throw(
                ArgumentError(
                    "loglikelihood(IncidenceCounts, model) could not check the model's " *
                        "attribute builders for an onset of their own: running them on " *
                        "a probe case failed with $(sprint(showerror, err))"
                )
            )
        end
        b isa ClinicalPresentation || isequal(onset(probe), before) || throw(
            ArgumentError(
                "loglikelihood(IncidenceCounts, model): an attribute builder other than " *
                    "clinical_presentation records an onset, which this closed form cannot read"
            )
        )
    end
    return nothing
end

"""
    loglikelihood(data::IncidenceCounts, model::Union{BranchingProcess, ModelSpec},
        approximation::IncidenceApproximation = NormalApproximation())

Log-likelihood of an incidence time series under a model's own offspring and
generation-time distributions, single-type or multi-type (a model built from
an offspring matrix, with `data` holding counts and seeds by type or their
total). A [`Reporting`](@ref) step makes `data` a series of reported cases
by report date: a report timed from infection uses the step's own
probability and delay, and one timed from onset (the default) adds the
model's [`clinical_presentation`](@ref) incubation period to the delay and
counts only the cases that are not asymptomatic.

Only defined for a model with no interventions, an unlimited
`population_size`, a single infectiousness window with no `until`,
attributes that leave the offspring law unchanged (see
[`EpiBranch.offspring_law_unchanged`](@ref)), and either no progression or a
`Reporting` step as its only one, with a fixed probability and delay. An
outcome or another transition can end a case before its report, which the
closed form, assuming each case is reported independently, does not
represent. Each of these limits throws an `ArgumentError` naming the reason
instead of a silent approximation. Simulate [`weekly_incidence`](@ref) (or daily, by
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
    offspring_law_unchanged(model) && offspring_law_unchanged(attributes(model)) &&
        all(offspring_law_unchanged, progression(model)) || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) has no closed form: a finite " *
                "population_size, an infectiousness window with `until`, or an " *
                "attribute (such as transmission_traits away from its neutral value) " *
                "reshapes transmission over time in a way this likelihood does not " *
                "represent. Simulate instead."
        )
    )
    reporting = _incidence_reporting(model)
    isempty(reporting) ||
        !any(k -> k in (:reporting_probability, :reporting_delay), keys(kwargs)) || throw(
        ArgumentError(
            "this model's Reporting step already sets the reporting probability and " *
                "delay; drop `reporting_probability` and `reporting_delay`"
        )
    )
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
    return loglikelihood(
        data, _analytic_offspring(model), gt, approximation; reporting..., kwargs...
    )
end
