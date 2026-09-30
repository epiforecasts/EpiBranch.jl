# ── Maximum-likelihood fitting with profile-likelihood intervals ────
#
# `fit` extends `Distributions.fit` (via StatsAPI) with methods that take one
# of the data wrappers first, mirroring the `loglikelihood(data, dist)`
# argument order already used throughout the package. The explicit `import`
# attaches new methods to the same generic function that `using Distributions`
# brought into scope, rather than shadowing it with an unrelated `fit` local
# to this module — the two would otherwise be ambiguous wherever a caller has
# both packages loaded.
#
# No dependency on Optim.jl: golden-section search covers the bounded 1D
# profiles this module needs, and a small Nelder-Mead simplex the joint
# (R, k) NegBin fit. Both work directly against the same `loglikelihood`
# methods used for the analytical and simulation-based likelihoods
# elsewhere, so a `fit` on data under interventions or a `ModelSpec` would
# need only a `loglikelihood` method for that combination — none is added
# here since interventions bring no closed-form profile to search.
import Distributions: fit

"""
    MLEFit

Result of a maximum-likelihood [`fit`](@ref): point estimate, the
maximised log-likelihood, and a profile-likelihood confidence interval
per parameter. A side of an interval that the search never bounds — most
often the upper side of `k`, where the Negative Binomial likelihood keeps
improving towards the Poisson limit as `k → ∞` — is reported as `Inf`
rather than a value the search had to give up on.

Fields:

- `estimate::NamedTuple` — the MLE, `(R,)` for a Poisson fit or `(R, k)`
  for a Negative Binomial fit.
- `loglikelihood::Float64` — the maximised log-likelihood.
- `ci::NamedTuple` — profile-likelihood interval per parameter, each a
  `(lower, upper)` tuple, at `level`.
- `level::Float64` — the confidence level the intervals target.
- `bootstrap_ci` — percentile interval from a parametric bootstrap, the
  same shape as `ci`, or `nothing` if `fit` was not asked for one.
"""
struct MLEFit{E <: NamedTuple, C <: NamedTuple, B}
    estimate::E
    loglikelihood::Float64
    ci::C
    level::Float64
    bootstrap_ci::B
end

function Base.show(io::IO, f::MLEFit)
    params = join(
        ("$name = $(round(v, digits = 3))" for (name, v) in pairs(f.estimate)), ", "
    )
    return print(io, "MLEFit($params; loglikelihood = $(round(f.loglikelihood, digits = 2)))")
end

# ── Generic derivative-free optimisation ────────────────────────────

const _GOLDEN_RATIO = (sqrt(5.0) - 1.0) / 2.0

"""Maximise a unimodal `f` on `[lo, hi]` by golden-section search."""
function _golden_max(f, lo::Float64, hi::Float64; iters::Int = 200)
    a, b = lo, hi
    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:iters
        if fc > fd
            b, d, fd = d, c, fc
            c = b - _GOLDEN_RATIO * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + _GOLDEN_RATIO * (b - a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

"""
Minimise `f: Vector{Float64} -> Float64` with the Nelder-Mead simplex
algorithm. A dependency-free stand-in for a 2-parameter optimiser: the
joint (log R, log k) NegBin likelihood has a ridge that block-coordinate
golden section follows only slowly, so a proper simplex search seeds the
coordinate-wise polish in the `NegativeBinomial` method of [`fit`](@ref).
"""
function _nelder_mead(f, x0::Vector{Float64}; iters::Int = 400, step::Float64 = 0.5)
    n = length(x0)
    simplex = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        simplex[i + 1][i] += step
    end
    fvals = [f(x) for x in simplex]
    for _ in 1:iters
        order = sortperm(fvals)
        simplex, fvals = simplex[order], fvals[order]
        centroid = sum(simplex[1:(end - 1)]) / n
        worst = simplex[end]
        xr = centroid + (centroid - worst)
        fr = f(xr)
        if fr < fvals[1]
            xe = centroid + 2 * (centroid - worst)
            fe = f(xe)
            simplex[end], fvals[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < fvals[end - 1]
            simplex[end], fvals[end] = xr, fr
        else
            xc = centroid + 0.5 * (worst - centroid)
            fc = f(xc)
            if fc < fvals[end]
                simplex[end], fvals[end] = xc, fc
            else
                for i in 2:(n + 1)
                    simplex[i] = simplex[1] + 0.5 * (simplex[i] - simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    best = argmin(fvals)
    return simplex[best], fvals[best]
end

"""
Bisect `g` (assumed to change sign once) between `lo` and `hi` for its
root.
"""
function _bisect(g, lo::Float64, hi::Float64; tol::Float64 = 1.0e-6, maxiter::Int = 100)
    glo = g(lo)
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        gmid = g(mid)
        if sign(gmid) == sign(glo)
            lo, glo = mid, gmid
        else
            hi = mid
        end
        (hi - lo) < tol * max(1.0, abs(mid)) && break
    end
    return (lo + hi) / 2
end

"""
One side (`direction = ±1`) of a profile-likelihood interval: expand
outward from the MLE `θ̂` until the profile `f(θ)` drops below `target`,
then bisect for the crossing. A side that reaches `bound` (the search's
numerical stand-in for infinity, or the model's own domain edge for
`lo_bound`) without crossing is reported as `Inf` on the upper side, or
`lo_bound` on the lower.
"""
function _profile_bound(
        f, θ̂::Float64, target::Float64, direction::Int;
        lo_bound::Float64, hi_bound::Float64,
        factor::Float64 = 1.5, max_expand::Int = 80, tol::Float64 = 1.0e-6
    )
    prev = θ̂
    for _ in 1:max_expand
        raw = direction == 1 ? prev * factor : prev / factor
        at_bound = direction == 1 ? raw >= hi_bound : raw <= lo_bound
        cand = direction == 1 ? min(raw, hi_bound) : max(raw, lo_bound)
        if f(cand) < target
            lo, hi = direction == 1 ? (prev, cand) : (cand, prev)
            return _bisect(θ -> f(θ) - target, lo, hi; tol)
        end
        at_bound && return direction == 1 ? Inf : lo_bound
        prev = cand
    end
    return direction == 1 ? Inf : lo_bound
end

"""Two-sided profile-likelihood interval around the MLE `θ̂`."""
function _profile_interval(
        f, θ̂::Float64, target::Float64; lo_bound::Float64 = 0.0, hi_bound::Float64
    )
    lower = _profile_bound(f, θ̂, target, -1; lo_bound, hi_bound)
    upper = _profile_bound(f, θ̂, target, 1; lo_bound, hi_bound)
    return (lower, upper)
end

# `ChainLengths` are only defined for a subcritical process (R < 1); the other
# data types allow any positive R, so search up to a generously high numerical
# cap rather than a true bound.
_r_search_bound(::Union{OffspringCounts, ChainSizes}) = 1.0e6
_r_search_bound(::ChainLengths) = 1.0 - 1.0e-9

const _K_SEARCH_BOUND = 1.0e6

_percentile_interval(v::AbstractVector, level::Real) =
    (quantile(v, (1 - level) / 2), quantile(v, 1 - (1 - level) / 2))

# Draw one bootstrap replicate of the same size (and, for `ChainSizes`, the
# same per-cluster seed counts) as `data`, from the fitted `offspring`.
function _resample(data::OffspringCounts, rng::AbstractRNG, offspring::Distribution)
    return OffspringCounts(rand(rng, offspring, length(data.data)))
end

function _resample(data::ChainSizes, rng::AbstractRNG, offspring::Distribution)
    law = chain_size_distribution(offspring)
    sizes = [sum(rand(rng, law) for _ in 1:s) for s in data.seeds]
    return ChainSizes(sizes; seeds = data.seeds)
end

function _resample(data::ChainLengths, rng::AbstractRNG, offspring::Distribution)
    states = simulate(BranchingProcess(offspring), length(data.data); rng)
    lengths = Int[]
    for s in states
        append!(lengths, chain_statistics(s).length)
    end
    return ChainLengths(lengths)
end

# Bind a fixed parameter through a function argument rather than a captured
# variable that the caller later reassigns: `R -> ll(R, k̂)` inside a loop
# that reassigns `k̂` boxes the capture (and confuses JET's inference), while
# `_fix_k(ll, k̂)` passes the current value once, giving a closure over an
# ordinary, never-reassigned parameter.
_fix_k(ll, k) = R -> ll(R, k)
_fix_R(ll, R) = k -> ll(R, k)

function _bootstrap_ci(
        data, family::Type{Poisson}, θ̂::NamedTuple,
        n_boot::Int, level::Real, rng::AbstractRNG
    )
    Rs = [fit(_resample(data, rng, Poisson(θ̂.R)), family).estimate.R for _ in 1:n_boot]
    return (R = _percentile_interval(Rs, level),)
end

function _bootstrap_ci(
        data, family::Type{NegativeBinomial}, θ̂::NamedTuple,
        n_boot::Int, level::Real, rng::AbstractRNG
    )
    fits = [fit(_resample(data, rng, NegBin(θ̂.R, θ̂.k)), family) for _ in 1:n_boot]
    Rs = [f.estimate.R for f in fits]
    ks = [f.estimate.k for f in fits]
    return (R = _percentile_interval(Rs, level), k = _percentile_interval(ks, level))
end

"""
    fit(data, ::Type{Poisson}; level=0.95, bootstrap=0, rng=Random.default_rng())
    fit(data, ::Type{NegativeBinomial}; level=0.95, bootstrap=0, rng=Random.default_rng())

Maximum-likelihood fit of `data` — an [`OffspringCounts`](@ref),
[`ChainSizes`](@ref), or [`ChainLengths`](@ref) — to a `Poisson(R)` or
`NegBin(R, k)` offspring distribution, via the same [`loglikelihood`](@ref)
methods used throughout the package for simulation and inference.

Returns an [`MLEFit`](@ref): the point estimate, the maximised
log-likelihood, and a profile-likelihood confidence interval per
parameter at `level` (default 0.95). With `bootstrap > 0`, that many
parametric-bootstrap replicates are simulated from the fitted
distribution and refit; `bootstrap_ci` on the result holds the resulting
percentile interval alongside the profile one.

`ChainLengths` are only defined for a subcritical process, so `R` is
searched over `(0, 1)`; `OffspringCounts` and `ChainSizes` allow any
positive `R`, including the supercritical region.

```julia
data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 1, 4])
f = fit(data, NegativeBinomial)
f.estimate      # (R = ..., k = ...)
f.ci.k          # (lower, upper) — upper is Inf if k is not identified
```
"""
function fit(
        data::Union{OffspringCounts, ChainSizes, ChainLengths}, ::Type{Poisson};
        level::Real = 0.95, bootstrap::Int = 0,
        rng::AbstractRNG = Random.default_rng()
    )
    r_bound = _r_search_bound(data)
    ll(R) = loglikelihood(data, Poisson(R))
    R̂, ll_max = _golden_max(ll, 1.0e-8, r_bound)
    target = ll_max - quantile(Chisq(1), level) / 2
    ci = (R = _profile_interval(ll, R̂, target; hi_bound = r_bound),)
    boot = bootstrap > 0 ?
        _bootstrap_ci(data, Poisson, (R = R̂,), bootstrap, level, rng) : nothing
    return MLEFit((R = R̂,), ll_max, ci, Float64(level), boot)
end

function fit(
        data::Union{OffspringCounts, ChainSizes, ChainLengths}, ::Type{NegativeBinomial};
        level::Real = 0.95, bootstrap::Int = 0,
        rng::AbstractRNG = Random.default_rng()
    )
    r_bound = _r_search_bound(data)
    ll(R, k) = loglikelihood(data, NegBin(R, k))

    # Seed the simplex from the Poisson fit (k large) rather than an arbitrary
    # guess, then polish with a couple of coordinate-wise golden-section
    # passes — the simplex alone is not tuned for high precision.
    R0 = _golden_max(R -> ll(R, 1.0), 1.0e-8, r_bound)[1]
    # The simplex's reflection/expansion steps routinely propose points outside
    # the domain `loglikelihood` is defined on (e.g. R >= 1 for `ChainLengths`,
    # which is only defined for a subcritical process); reject those with a
    # bad score instead of letting the domain check inside `ll` throw.
    function neg_ll(θ)
        R, k = exp(θ[1]), exp(θ[2])
        # The lower bound matches the one used for the golden-section passes
        # below: below it, `NegBin`'s `p = k / (k + R)` rounds to `1.0` and
        # its `mean` collapses to exactly `0.0`, which fails `GammaBorel`'s
        # domain check just as surely as `R`/`k` underflowing to `0.0` would.
        (R <= 1.0e-8 || k <= 1.0e-8 || R >= r_bound || k >= _K_SEARCH_BOUND) && return Inf
        return -ll(R, k)
    end
    θ̂, _ = _nelder_mead(neg_ll, [log(R0), 0.0])
    R̂, k̂ = exp(θ̂[1]), exp(θ̂[2])
    for _ in 1:3
        R̂ = _golden_max(_fix_k(ll, k̂), 1.0e-8, r_bound)[1]
        k̂ = _golden_max(_fix_R(ll, R̂), 1.0e-8, _K_SEARCH_BOUND)[1]
    end
    ll_max = ll(R̂, k̂)

    target = ll_max - quantile(Chisq(1), level) / 2
    profile_over_k(R) = _golden_max(_fix_R(ll, R), 1.0e-8, _K_SEARCH_BOUND)[2]
    profile_over_r(k) = _golden_max(_fix_k(ll, k), 1.0e-8, r_bound)[2]
    ci = (
        R = _profile_interval(profile_over_k, R̂, target; lo_bound = 1.0e-8, hi_bound = r_bound),
        k = _profile_interval(
            profile_over_r, k̂, target; lo_bound = 1.0e-8, hi_bound = _K_SEARCH_BOUND
        ),
    )
    boot = bootstrap > 0 ?
        _bootstrap_ci(data, NegativeBinomial, (R = R̂, k = k̂), bootstrap, level, rng) :
        nothing
    return MLEFit((R = R̂, k = k̂), ll_max, ci, Float64(level), boot)
end
