# ── Incidence time-series likelihood ─────────────────────────────────
#
# A closed-form likelihood for the incidence time series a branching process
# produces. `incidence_moments` describes the model, in the spirit of
# Boudreau and St-Onge's calendar-time branching process (arXiv:2610.17138):
# advance the probability generating function one time step at a time, with
# one auxiliary variable per step, then read the mean and covariance of the
# step counts off the cumulant generating function (the log of that PGF). The
# likelihood itself is the renewal count likelihood further down.
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

# Cumulative distribution of the step lag between two events when the first
# falls uniformly within its step (double interval censoring) and the second
# is that event plus a delay with cumulative distribution `F`: the lag is at
# most `l` steps with probability ∫_l^{l+1} F(x) dx. Cases are infected
# throughout a step, so binning the delay directly onto `(l - 1, l]` would
# overstate the realised lag by half a step.
_censored_cdf(F) = l -> first(quadgk(F, l, l + 1))

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

# Discretised generation-time pmf over lag 1, 2, ...: `w[j]` is the
# probability that an infectee's step is `j` steps after its infector's, with
# the infector infected uniformly within its step (see `_censored_cdf`). A
# lag of 0 (both infected in the same step) has no place in the recursion.
# That mass is dropped and the rest renormalised, which changes little when
# the generation time is long compared with a step. The
# distribution is taken to be continuous with no atom at or below 0; the
# integral over each step avoids evaluating the cdf at 0 itself, which a `Dual`
# parameter would send through `log(0) = -Inf` for a distribution like
# `LogNormal`.
function _discretize_generation_interval(
        generation_time::Distribution; tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    H = _censored_cdf(t -> cdf(generation_time, t))
    return _bin_cdf(
        H, 1, tol, max_lag, "generation-time distribution"; base = H(0)
    )
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
    return _bin_cdf(_censored_cdf(t -> cdf(delay, t)), 0, tol, max_lag, "reporting delay")
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
    b isa Real && return _bin_cdf(
        _censored_cdf(t -> cdf(a, t - b)), 0, tol, max_lag, "reporting delay"
    )
    # A fixed first delay has no density for the quadrature to sample. It
    # shifts the second delay instead.
    if a isa Dirac
        shift = a.value
        return _bin_cdf(
            _censored_cdf(t -> cdf(b, t - shift)), 0, tol, max_lag, "reporting delay"
        )
    end
    # A fixed second delay shifts the first in the same way.
    if b isa Dirac
        shift = b.value
        return _bin_cdf(
            _censored_cdf(t -> cdf(a, t - shift)), 0, tol, max_lag, "reporting delay"
        )
    end
    # P(A + B ≤ t) = ∫ f_A(x) F_B(t - x) dx over x ≤ t, B being non-negative.
    lo = minimum(a)
    F(t) = t <= lo ? 0.0 : first(quadgk(x -> pdf(a, x) * cdf(b, t - x), lo, t))
    return _bin_cdf(_censored_cdf(F), 0, tol, max_lag, "reporting delay")
end

# The offspring law's own numeric element type, promoted with `eltype(u)` in
# `_incidence_log_pgf`. `Distributions.partype` gives this for any
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
limits). Use it for a forecast, or to compare against simulated incidence
(`seeds` simulated introductions, summed into steps the same way `data` is).

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

# ── Renewal count likelihood ──────────────────────────────────────────
#
# The counts are scored with the model that `incidence_moments` describes, in
# its count form. The cases infected at step `s` form a cohort `C_s`. With
# negative binomial offspring the combined offspring mean of a cohort is
# `V_s ~ Gamma(k * C_s, R / k)`, because gammas with a shared scale add, and
# the count at step `t` is `Poisson(Σ_s V_s * w[t - s])`. Given the cohorts,
# the count at step `t` is a Poisson mixture with mean
#
#   μ_t = R * Σ_s w[t - s] * C_s
#
# and extra variance
#
#   e_t = (σ² - R) * Σ_s w[t - s]² * C_s,
#
# where σ² is the variance of the offspring law (R²/k for the negative
# binomial, for which the mean-variance relation is exact). Any other law is
# covered the same way: the offspring of a case landing `j` steps later are
# its offspring thinned by `w[j]`. A negative binomial with that mean and
# variance, `μ_t + e_t`, scores the count, with a Poisson where `e_t = 0`.
# Each step is scored given the observed counts before it, and the likelihood
# is the product over steps. This treats the steps as independent given the
# past and drops the correlation between the offspring of one infector spread
# over several steps.
#
# With under-reporting the cohorts are not observed. The reports `Z_s` stand
# in for them: a report probability `ρ` and delay `d` commute with the
# generation-time convolution, so `R * Σ_s w[t - s] * Z_s` plus the seeds' own
# reports is the expected count of reports at step `t`. Thinning scales the
# variance of a Poisson mixture by `ρ²` and the cohort by `1 / ρ`, so `e_t`
# has a factor `ρ`. The seeds' own reports, a binomial draw, are scored as
# if Poisson. Both are approximations that are exact without under-reporting.

# Variance of an offspring law, with the same fallback as `_law_mean` for a
# law that has no `var` method.
function _law_var(d::DiscreteUnivariateDistribution)
    try
        return var(d)
    catch err
        err isa MethodError || rethrow()
    end
    μ = _law_mean(d)
    lo, hi = _series_range(d)
    return sum((x - μ)^2 * pdf(d, x) for x in lo:hi)
end
function _law_var(o::GeneratingFunctionOffspring)
    # G''(1) + G'(1) - G'(1)^2, the variance from the factorial moments.
    g2 = ForwardDiff.derivative(s -> ForwardDiff.derivative(o.pgf, s), 1.0)
    return g2 + o.mean - o.mean^2
end

# The excess of each law's variance over its mean, which is what separates its
# counts from Poisson ones. A law below its mean is not a Poisson mixture.
function _offspring_excess(laws)
    return map(laws) do law
        R, σ² = _law_mean(law), _law_var(law)
        σ² >= R - 1.0e-8 * max(one(R), R) || throw(
            ArgumentError(
                "the incidence likelihood is a count likelihood for offspring laws " *
                    "at least as dispersed as Poisson; this one has variance $σ² below " *
                    "its mean $R"
            )
        )
        return (mean = R, excess = max(σ² - R, zero(σ²)))
    end
end

# Conditional mean and extra variance of the count of each observed category
# (rows of `P`, a map from types to categories: the identity for counts by
# type, a row of ones for their total) at each step 1:n, given `Z`, the
# cohorts (rows `s + 1` for step `s`, columns types) in place of the
# unobserved infections. `alloc` is the child-type allocation.
function _incidence_conditional(
        Z::AbstractMatrix, P::AbstractMatrix, alloc::AbstractMatrix, laws,
        seeds::AbstractVector, w::AbstractVector, d::AbstractVector, ρ::Real, n::Integer
    )
    stats = _offspring_excess(laws)
    A = P * alloc
    q, m = size(A)
    T = promote_type(
        eltype(Z), eltype(A), eltype(w), eltype(d), typeof(ρ), eltype(seeds),
        map(s -> typeof(s.mean), stats)..., map(s -> typeof(s.excess), stats)...
    )
    μ = zeros(T, n, q)
    e = zeros(T, n, q)
    for t in 1:n, c in 1:q
        direct = t + 1 <= length(d) ? ρ * d[t + 1] * sum(P[c, j] * seeds[j] for j in 1:m) : zero(T)
        mean_t = direct
        extra_t = zero(T)
        for j in 1:m
            a = A[c, j]
            iszero(a) && continue
            s1 = zero(T)
            s2 = zero(T)
            for s in max(0, t - length(w)):(t - 1)
                z = Z[s + 1, j]
                s1 += w[t - s] * z
                s2 += w[t - s]^2 * z
            end
            mean_t += a * stats[j].mean * s1
            extra_t += ρ * a^2 * stats[j].excess * s2
        end
        μ[t, c] = mean_t
        e[t, c] = extra_t
    end
    return μ, e
end

# Log-likelihood of counts `y` (steps by categories) given their conditional
# mean `μ` and extra variance `e`: negative binomial with variance `μ + e`,
# Poisson without extra variance. A count whose mean is 0, such as one before a
# delayed report can arrive or of a type with no cohort yet, can only be zero.
function _incidence_count_loglik(y::AbstractMatrix{<:Integer}, μ, e)
    T = float(eltype(μ))
    ll = zero(T)
    for idx in eachindex(y)
        yi, μi, ei = y[idx], μ[idx], e[idx]
        if !(μi > 0)
            yi == 0 || return T(-Inf)
        elseif !(ei > 0)
            ll += logpdf(Poisson(μi), yi)
        else
            r = μi^2 / ei
            ll += logpdf(NegativeBinomial(r, r / (r + μi)), yi)
        end
    end
    return ll
end

# The cohorts of every step as a matrix: the reports of step `s` by type in
# row `s + 1`, and the seeds' own expected reports in row 1.
function _incidence_cohorts(Y::AbstractMatrix, seeds, ρ, d)
    return vcat(reshape(ρ * d[1] .* seeds, 1, :), Y)
end

# The cohorts when only the total over types is observed: each step's total is
# split by the type shares of the expected counts of the model at that step.
function _incidence_cohorts(
        y::AbstractVector, expected::AbstractMatrix, seeds, ρ, d
    )
    n, m = size(expected)
    Y = zeros(promote_type(Float64, eltype(expected)), n, m)
    for s in 1:n
        tot = sum(expected[s, :])
        shares = tot > 0 ? expected[s, :] ./ tot : seeds ./ sum(seeds)
        Y[s, :] = y[s] .* shares
    end
    return _incidence_cohorts(Y, seeds, ρ, d)
end

# ── Unified loglikelihood interface ────────────────────────────────────

"""
    loglikelihood(data::IncidenceCounts, offspring, generation_time::Distribution;
        reporting_probability = 1, reporting_delay = nothing)

Log-likelihood of an incidence time series under a branching process with
offspring law `offspring` and a `generation_time` distribution, both as in
[`incidence_moments`](@ref), as are `reporting_probability` and
`reporting_delay` for a series of reported cases.

It is a renewal count likelihood: given the counts before it, the count at
step `t` is negative binomial with mean `R * Σ_s w[s] * C[t - s]`, `w` being
the discretised generation time and `C` the cases infected at each step (the
`seeds` at step 0). The dispersion comes from the variance `σ²` of the
offspring law, giving the count a variance of
`μ + (σ² - R) * Σ_s w[s]² * C[t - s]`. For negative binomial offspring with
mean `R` and dispersion `k` this is the variance of the count
`Poisson(Σ_s V_s * w[t - s])`, where `V_s` is the combined offspring mean of
the cohort `C_s`, `Gamma(k * C_s, R / k)`. A Poisson offspring law gives a
Poisson count, the limit as `k` grows. Counts of zero are scored like any
other.

With `reporting_probability` below 1, or a `reporting_delay`, the counts are
of reports and stand in for the unobserved infections. The mean is the
expected reports given the earlier ones and the extra variance is scaled by
the reporting probability. This is exact for the mean and approximate for the
variance.

!!! warning "Limits"
    The steps are scored one at a time given the past, which drops the
    correlation between the offspring of one infector spread over several
    steps. The model ignores susceptible depletion and holds only while the
    population is effectively unsaturated. It models no interventions, and
    under-reporting only as each case reported independently with the same
    probability. The `generation_time` distribution and `reporting_delay` are
    discretised onto the time step of `data`, with each infection uniform
    within its step.
    The recursion has no lag of 0, so generation times shorter than a step
    lose their same-step mass and the estimate is biased when the step is long
    compared with the generation time. An offspring law with a variance below
    its mean is not a Poisson mixture and throws an `ArgumentError`.

# Examples

```julia
data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18])
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5))
# 60% of cases reported, on average 3 days after infection
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5);
    reporting_probability = 0.6, reporting_delay = Gamma(2.0, 1.5))
```
"""
function loglikelihood(
        data::IncidenceCounts,
        offspring::Union{DiscreteUnivariateDistribution, GeneratingFunctionOffspring},
        generation_time::Distribution;
        reporting_probability::Real = 1, reporting_delay = nothing,
        tol::Real = 1.0e-8, max_lag::Int = 10_000
    )
    y, seeds = _single_type_series(data)
    n = length(y)
    0 < reporting_probability <= 1 ||
        throw(ArgumentError("reporting_probability must be in (0, 1]"))
    w = _discretize_generation_interval(generation_time; tol, max_lag)
    d = _discretize_report_delay(reporting_delay; tol, max_lag)
    Y = reshape(y, n, 1)
    Z = _incidence_cohorts(Y, [seeds], reporting_probability, d)
    μ, e = _incidence_conditional(
        Z, ones(1, 1), ones(1, 1), [offspring], [seeds], w, d, reporting_probability, n
    )
    return _incidence_count_loglik(Y, μ, e)
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
        ρ::Real, d::AbstractVector{<:Real}, ::Type{Tpar}
    ) where {T <: Real, Tpar <: Real}
    m = length(pgfs)
    S = promote_type(T, eltype(w), eltype(d), eltype(alloc), Tpar, typeof(ρ))
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
    child(j) = sum(alloc[i, j] * B[i] for i in 1:m)
    for k in n:-1:1
        next_generation!(k)
        for j in 1:m
            Ψ[k, j] = report(k, j) * pgfs[j](child(j))
        end
    end
    next_generation!(0)
    K = zero(S)
    for j in 1:m
        iszero(seeds[j]) && continue
        K += seeds[j] * log(report(0, j) * pgfs[j](child(j)))
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
    μ = ForwardDiff.gradient(K, u1)
    H = ForwardDiff.hessian(K, u1)
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

"""
    loglikelihood(data::IncidenceCounts, offspring::MultiTypeOffspring,
        generation_time::Distribution;
        reporting_probability = 1, reporting_delay = nothing)

Log-likelihood of an incidence time series under a multi-type branching
process: the renewal count likelihood of the single-type method, with a count
for each type at each step. The offspring of a type-`j` cohort land in type `i`
with probability `a[i, j]`, giving the count of type `i` a mean of
`Σ_j a[i, j] * R_j * Σ_s w[s] * C_j[t - s]` and the dispersion of the type-`j`
offspring law. The counts of different types at one step are scored as
independent given the past, which drops their correlation through the shared
parent.

`data` holds the counts by type, one column per type, or their total over
types as a vector; its `seeds` give the index cases of each type. The cohorts
of a total are split by type in the proportions of the expected counts of the
model at each step, because the observed counts have no split. The limits are
those of the single-type method.

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
        generation_time::Distribution;
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
    m = _n_types(offspring)
    data.data isa AbstractMatrix && size(data.data, 2) != m && throw(
        ArgumentError(
            "counts by type have $(size(data.data, 2)) columns, but the process has " *
                "$m types"
        )
    )
    length(data.seeds) == m || throw(
        ArgumentError("seeds needs one entry per type ($m), got $(length(data.seeds))")
    )
    0 < reporting_probability <= 1 ||
        throw(ArgumentError("reporting_probability must be in (0, 1]"))
    w = _discretize_generation_interval(generation_time; tol, max_lag)
    d = _discretize_report_delay(reporting_delay; tol, max_lag)
    seeds = data.seeds
    ρ = reporting_probability
    laws = _total_count_laws(offspring)
    if data.data isa AbstractMatrix
        Y = data.data
        Z = _incidence_cohorts(Y, seeds, ρ, d)
        P = Matrix{Float64}(I, m, m)
    else
        Y = reshape(data.data, n, 1)
        expected = incidence_moments(
            offspring, generation_time, n; seeds, reporting_probability, reporting_delay,
            tol, max_lag
        ).mean
        Z = _incidence_cohorts(data.data, expected, seeds, ρ, d)
        P = ones(1, m)
    end
    μ, e = _incidence_conditional(Z, P, offspring.alloc_probs, laws, seeds, w, d, ρ, n)
    return _incidence_count_loglik(Y, μ, e)
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
    hasproperty(r, :delay) || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) reads the delay of a " *
                "reporting step from its `delay` field; $(typeof(r)) has none"
        )
    )
    r.delay isa Union{Real, Distribution} || throw(
        ArgumentError(
            "loglikelihood(IncidenceCounts, model) needs the Reporting delay to be a " *
                "number or a distribution, not a per-case function"
        )
    )
    reached, to_anchor = _infection_to(reporting_from(r), attributes(model))
    anchor = to_anchor isa Real ? Dirac(float(to_anchor)) : to_anchor
    return (
        reporting_probability = _reporting_probability(prog) * reached,
        reporting_delay = anchor === nothing ? r.delay : _DelaySum(anchor, r.delay),
    )
end

# The share of cases that reach the time a `Reporting` step measures its delay
# from, and that time's delay after infection (`nothing` for infection itself).
function _infection_to(from::Symbol, attrs)
    from === :infection && return (1.0, nothing)
    _resolved_key(from) === :onset_time && return _infection_to_onset(attrs)
    throw(
        ArgumentError(
            "The closed form supports a Reporting step from " *
                "infection or onset, not from `:$from`"
        )
    )
end
function _infection_to(from, attrs)
    throw(
        ArgumentError(
            "The closed form supports a Reporting step from " *
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
    laws = filter(!isnothing, map(onset_law, builders))
    length(laws) == 1 || throw(
        ArgumentError(
            "The closed form reads the onset a Reporting step " *
                "reports from from the model's clinical_presentation (or another " *
                "attribute that defines `onset_law`), and needs exactly " *
                "one; this model has $(length(laws))"
        )
    )
    # Fresh copies, so the probe leaves the model's own builders as they were.
    _check_onset_builders(_fresh_attributes(builders))
    cp = only(laws)
    cp.prob_asymptomatic isa Real || throw(
        ArgumentError(
            "The closed form needs a fixed prob_asymptomatic, not " *
                "a per-case function"
        )
    )
    return (1 - cp.prob_asymptomatic, _closed_form_incubation(cp, "The closed form"))
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
                    "The closed form could not check the model's " *
                        "attribute builders for an onset of their own: running them on " *
                        "a probe case failed with $(sprint(showerror, err))"
                )
            )
        end
        onset_law(b) !== nothing || isequal(onset(probe), before) || throw(
            ArgumentError(
                "the closed form: an attribute builder that does not define `onset_law` records an onset, which this closed form cannot read"
            )
        )
    end
    return nothing
end

"""
    loglikelihood(data::IncidenceCounts, model::Union{BranchingProcess, ModelSpec})

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
        data::IncidenceCounts, model::Union{BranchingProcess, ModelSpec}; kwargs...
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
        data, _analytic_offspring(model), gt; reporting..., kwargs...
    )
end
