# ── Moments of an incidence time series ──────────────────────────────
#
# The mean and covariance of the incidence time series a branching process
# produces, in the spirit of Boudreau and St-Onge's calendar-time branching
# process (arXiv:2610.17138): advance the probability generating function one
# time step at a time, with one auxiliary variable per step, then read
# the mean and covariance of the step counts off the cumulant generating
# function (the log of that PGF).
#
# One individual infected at step k (1 ≤ k ≤ n, the last observed step)
# contributes u_k for itself, plus the PGF of its offspring's own
# contributions, each landing at step k + j (j ≥ 0) with probability w_j from
# the discretised generation-time distribution:
#
#   Ψ_k(u) = u_k · f(Σ_{j≥0} w_j · Ψ_{k+j}(u)),   Ψ_m(u) ≡ 1 for m > n
#
# (`Ψ_m ≡ 1` beyond the horizon marginalises out what is not observed.) The
# j = 0 term covers offspring infected in the step of their infector. Ψ_k
# therefore appears on both sides and is found by fixed-point iteration. The
# `seeds` index cases at step 0 are not themselves observed, so their shared
# PGF drops the `u_0` factor, Ψ_0(u) = f(Σ_{j≥0} w_j · Ψ_j(u)), and their
# own offspring follow the same rule. Because they
# act independently the whole series has cumulant generating function
# K(u) = log G(u) = seeds · log Ψ_0(u).
#
# Under-reporting replaces the `u_k` an individual contributes for itself
# with its report: none with probability 1 - ρ, otherwise one at step k + l
# with probability d[l] from the discretised infection-to-report delay,
#
#   r_k(u) = 1 - ρ + ρ · Σ_l d[l] · U_{k+l},   U_m = u_m for 1 ≤ m ≤ n, else 1,
#
# so Ψ_k(u) = r_k(u) · f(Σ_j w_j · Ψ_{k+j}(u)) and Ψ_0 takes the factor
# r_0 too: a seed's own report, delayed into an observed step, is counted
# like anyone else's. With ρ = 1 and no delay, r_k = u_k and r_0 = 1, the
# recursion above. Differentiating the composition at u = 1 gives the mean
# and covariance of the step counts:
#
#   mean[t]        = ∂K/∂u_t
#   cov[t, t']     = ∂²K/∂u_t∂u_t' + δ_tt' · mean[t]
#
# the usual bridge from a probability generating function's factorial
# moments to ordinary ones. `ForwardDiff` takes those derivatives (through
# `_gradient_and_hessian`, defined by the extension that loads with it); `K` is
# written generically in its argument's element type so a `Dual` offspring
# or generation-time parameter still differentiates (nested automatic
# differentiation), the same contract `_chain_length_ll_pgf` keeps for the
# chain-length likelihood.

# Cumulative distribution of the step lag between two events when the first
# falls uniformly within its step and the second is that event plus a delay
# with distribution `delay`: the lag is at most `l` steps with probability
# `cdf(dic, l + 1)`, where `dic` is the delay under double interval
# censoring (a uniform primary event, whole-step secondary interval). Cases
# are infected throughout a step. Binning the delay directly onto
# `(l - 1, l]` would overstate the realised lag by half a step.
# The closed forms of CensoredDistributions.jl and its `Dual`-aware numeric
# fallback keep this differentiable in the parameters of the delay.
function _step_lag_cdf(delay::Distribution)
    dic = double_interval_censored(delay; interval = 1)
    return l -> cdf(dic, l + 1)
end

# A fixed delay is a point mass whose cdf jumps at the step boundaries where
# the censored cdf is evaluated. The lag cdf is therefore written out directly:
# a delay of `c` steps from a uniform start lands in step `floor(c)` or the
# one after it, with the fractional part of `c` the chance of the latter.
_step_lag_cdf(delay::Dirac) = l -> clamp(l + 1 - delay.value, 0.0, 1.0)

# Bin the step-lag cumulative distribution `H` onto the integer lags
# `first_lag`, `first_lag + 1`, ...: the first entry is `H(first_lag) - base`,
# where `base` is the mass below `first_lag` that the lags cannot represent,
# then each `H(l) - H(l - 1)`. Stops once the remaining tail is below `tol`,
# then renormalises away that negligible truncation and the dropped `base`.
# `what` names the distribution in the truncation warning.
function _bin_cdf(
        H, first_lag::Integer, tol::Real, max_lag::Int, what::AbstractString;
        base::Real = 0.0
    )
    lag = first_lag
    c = H(lag)
    p = [c - base]
    while !(1 - c < tol)
        if lag >= max_lag
            @warn "The $what has more than $tol of its mass above a lag of " *
                "$max_lag steps, so the discretised pmf is truncated there." maxlog = 1
            break
        end
        lag += 1
        c_prev = c
        c = H(lag)
        push!(p, c - c_prev)
    end
    s = sum(p)
    s > 0 || throw(ArgumentError("the $what has no mass on the lags binned"))
    return p ./ s
end

# Discretised generation-time pmf over lag 0, 1, 2, ...: `w[l + 1]` is the
# probability that an infectee's step is `l` steps after its infector's, with
# the infector infected uniformly within its step (see `_step_lag_cdf`). Lag 0
# covers infectees in the same step as their infector. It matters when the
# generation time is short compared with a step: dropping it would attribute
# those infections to later steps and overstate R.
function _discretize_generation_interval(
        generation_time::Distribution; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    H = _step_lag_cdf(generation_time)
    return _bin_cdf(H, 0, tol, max_lag, "generation-time distribution")
end

# Discretised infection-to-report delay pmf over lag 0, 1, 2, ...: `d[1]` is
# the probability of a report in the step of infection, and `d[l + 1]` that of
# one `l` steps later, with the infection uniform within its step as for the
# generation interval.
# `nothing` reports every case in the step it is infected, and a number is a
# fixed delay of that many steps.
_discretize_report_delay(::Nothing; kwargs...) = [1.0]
_discretize_report_delay(delay::Real; kwargs...) = _discretize_report_delay(Dirac(delay); kwargs...)
function _discretize_report_delay(
        delay::Distribution; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    minimum(delay) >= 0 || throw(ArgumentError("reporting_delay must be non-negative"))
    return _bin_cdf(_step_lag_cdf(delay), 0, tol, max_lag, "reporting delay")
end

# The offspring law's own numeric element type, promoted with `eltype(u)` in
# `_incidence_log_pgf`. `Distributions.partype` gives this for any
# `Distribution`; a `GeneratingFunctionOffspring`'s `pgf` closure is opaque, so
# its own parameters (if any are duals under AD) must already return that
# type consistently and `Float64` is assumed here.
_offspring_partype(offspring::Distribution) = Distributions.partype(offspring)
_offspring_partype(::GeneratingFunctionOffspring) = Float64

# The size of the change between two iterates of the same-step fixed point,
# partial derivatives included; the `ForwardDiff.Dual` method is in
# `ext/EpiBranchForwardDiffExt.jl`.
_fixed_point_gap(a::Real, b::Real) = abs(a - b)

# The gradient and Hessian of `K` at `u`. The method lives in
# `ext/EpiBranchForwardDiffExt.jl`, so `ForwardDiff` stays optional for users
# who only simulate.
function _gradient_and_hessian(K, u)
    throw(
        ArgumentError(
            "incidence_moments differentiates " *
                "the generating function with ForwardDiff.jl; load it with " *
                "`using ForwardDiff`."
        )
    )
end

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

Offspring infected in the same step as their infector are kept, as are those
of the seeds. A generation time so short against the step, or an offspring
mean so large, that the expected number of infections within a step is
unbounded throws an `ArgumentError` asking for a shorter step.

Requires `using ForwardDiff` for the derivatives of the generating function.

Returns a `(mean, cov)` named tuple: `mean[t]` is the expected count at step
`t`, and `cov` is their `n × n` covariance matrix, both ignoring susceptible
depletion. Use it for a forecast, or to compare against simulated incidence
(`seeds` simulated introductions, summed into steps of the same length).

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
    # The one-type case of the multi-type recursion.
    K(u) = _incidence_log_pgf(
        u, w, [s -> _pgf(offspring, s)], ones(1, 1), n, [seeds], reporting_probability,
        d, Toff
    )
    return _incidence_moments_from(K, n)
end

# `tail[l] = Σ_{l' ≥ l} p[l']`, with `tail[end] = 0`.
function _tail_sums(p::AbstractVector)
    tail = zeros(float(eltype(p)), length(p) + 1)
    for l in length(p):-1:1
        tail[l] = tail[l + 1] + p[l]
    end
    return tail
end

# ── Multi-type processes ──────────────────────────────────────────────
#
# A type-j case draws its total offspring from G_j = dist_fn(R_j) and gives
# each child type i with probability a[i, j] (see `multi_type.jl`), and each
# child independently lands w[l + 1] steps later (`w[1]` is the same step).
# Each type and step has its own auxiliary variable, so with
# B_i(k) = Σ_{l≥0} w[l + 1] · Ψ_{k+l, i},
#
#   Ψ_{k, j}(u) = r_{k, j}(u) · G_j(Σ_i a[i, j] · B_i(k)),
#
# r as in the single-type recursion but reading type j's own variables, and
# K(u) = Σ_j seeds[j] · log Ψ_{0, j}(u). The variables are ordered step
# first, then type, as `vec` orders an n × m matrix.
#
# The single-type process is the one-type case, with `pgfs = [pgf]` and
# `alloc = ones(1, 1)`. `Tpar` is the offspring laws' own numeric element
# type (`Distributions.partype`, `Float64` for a `GeneratingFunctionOffspring`
# because its `pgf` closure is opaque): `Ψ`'s element type promotes `eltype(u)` with
# it, and with `ρ`'s, rather than taking `eltype(u)` alone, so a `Dual`
# offspring or reporting parameter composes correctly even while `u` is still
# the plain `Float64` ForwardDiff probes before perturbing it (nested
# automatic differentiation, as a `Dual` offspring under an outer AD call over
# R or k gives).
function _incidence_log_pgf(
        u::AbstractVector{T}, w::AbstractVector{<:Real}, pgfs::AbstractVector,
        alloc::AbstractMatrix{<:Real}, n::Integer, seeds::AbstractVector{<:Real},
        ρ::Real, d::AbstractVector{<:Real}, ::Type{Tpar};
        max_iter::Int = 100_000
    ) where {T <: Real, Tpar <: Real}
    m = length(pgfs)
    S = promote_type(T, eltype(w), eltype(d), eltype(alloc), Tpar, typeof(ρ))
    U(k, i) = 1 <= k <= n ? S(u[k + (i - 1) * n]) : one(S)
    # Lags beyond step `n` contribute 1. Only the lags inside the horizon are
    # summed, and a precomputed tail sum, `wtail[l] = Σ_{l' ≥ l} w[l']`, adds
    # the mass beyond them.
    wtail = _tail_sums(w)
    dtail = _tail_sums(d)
    function report(k, i)
        r = zero(S)
        L = min(length(d), n - k + 1)
        for l in 0:(L - 1)
            r += d[l + 1] * U(k + l, i)
        end
        r += dtail[L + 1]
        return 1 - ρ + ρ * r
    end
    Ψ = Matrix{S}(undef, n, m)
    # What the later steps contribute to a step-`k` case's offspring. Lags
    # beyond the horizon contribute 1 through the precomputed tail sum.
    function later(k, i)
        b = zero(S)
        L = min(length(w) - 1, n - k)
        for l in 1:L
            b += w[l + 1] * Ψ[k + l, i]
        end
        return b + wtail[L + 2]
    end
    # Ψ_k depends on itself through the offspring infected in the same step
    # (weight w[1]), so it solves Ψ_{k,j} = r_{k,j} · G_j(Σ_i a[i,j] (w[1] Ψ_{k,i} + b_i))
    # by iterating from the value without same-step offspring. The iteration
    # converges, derivatives included, when the expected same-step
    # offspring of a case are below one.
    function solve_step(k)
        b = [later(k, i) for i in 1:m]
        r = [report(k, j) for j in 1:m]
        x = [r[j] * pgfs[j](sum(alloc[i, j] * b[i] for i in 1:m)) for j in 1:m]
        w[1] > 0 || return x
        converged = false
        for _ in 1:max_iter
            y = [
                r[j] * pgfs[j](sum(alloc[i, j] * (w[1] * x[i] + b[i]) for i in 1:m))
                    for j in 1:m
            ]
            gap = maximum(_fixed_point_gap(y[j], x[j]) for j in 1:m)
            x = y
            gap <= 1.0e-13 && (converged = true)
            converged && break
            # Derivatives that grow instead of settling.
            gap < 1.0e12 || break
        end
        # The moments are read at u = 1, where the PGF of a finite count is 1.
        # A value below 1 means the iteration reached the extinction root of a
        # same-step cascade that is itself supercritical.
        if !converged || any(xj -> abs(_plain(xj) - 1) > 1.0e-6, x)
            throw(
                ArgumentError(
                    "the same-step offspring of a case do not converge: the " *
                        "generation time puts too much mass within a step for the " *
                        "offspring law, so the expected number of infections " *
                        "within a step is unbounded; use a shorter step."
                )
            )
        end
        return x
    end
    for k in n:-1:1
        Ψ[k, :] = solve_step(k)
    end
    K = zero(S)
    ψ0 = solve_step(0)
    for j in 1:m
        iszero(seeds[j]) && continue
        K += seeds[j] * log(ψ0[j])
    end
    return K
end

function _incidence_log_pgf(
        u::AbstractVector, w::AbstractVector{<:Real}, o::MultiTypeOffspring,
        n::Integer, seeds::AbstractVector{<:Real}, ρ::Real, d::AbstractVector{<:Real}
    )
    laws = _total_count_laws(o)
    Tpar = promote_type(
        eltype(o.R_by_type), map(Distributions.partype, laws)...
    )
    pgfs = [s -> _pgf(law, s) for law in laws]
    return _incidence_log_pgf(u, w, pgfs, o.alloc_probs, n, seeds, ρ, d, Tpar)
end

# Mean and covariance from the cumulant generating function `K` of `N` step
# variables, by the factorial-to-ordinary moment bridge described above.
function _incidence_moments_from(K, N::Integer)
    u1 = ones(Float64, N)
    μ, H = _gradient_and_hessian(K, u1)
    return (mean = μ, cov = Symmetric(H + Diagonal(μ)))
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
    μ, cov = _incidence_moments_from(K, n * m)
    return (mean = reshape(μ, n, m), cov = cov)
end
