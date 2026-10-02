"""
    Borel(μ)

The Borel distribution with parameter `μ > 0`.

P(X = n) = (μn)^(n-1) * exp(-μn) / n!  for n = 1, 2, ...

This is the chain size distribution for a Poisson(μ) branching process.
For `μ > 1` (supercritical) the PMF is still valid at each `n`, but its
total mass is less than 1: chains are infinite with positive probability.
We keep the PMF defined in the supercritical region so that integrating
chain size PMFs over a mixing distribution that spans both sides of 1
works pointwise.
"""
struct Borel{T <: Real} <: DiscreteUnivariateDistribution
    μ::T

    function Borel(μ::Real)
        0.0 < μ || throw(ArgumentError("μ must be positive, got $μ"))
        return new{typeof(μ)}(μ)
    end
end

Distributions.params(d::Borel) = (d.μ,)

"""
Log-PDF of the Borel distribution. Accepts any numeric type for μ
(AD-compatible). With `s > 1`, this is the Borel-Tanner generalisation
for the total chain size starting from `s` independent index cases:
`P(X = x | s, μ) = (s/x) * (xμ)^(x-s) * exp(-xμ) / (x-s)!`.

The single-seed case (`s = 1`) is a port of the Borel chain-size
log-likelihood (`dborel` / `.pois_size_ll`) in epichains (Azam, Funk &
Finger, https://github.com/epiverse-trace/epichains, MIT).
"""
function _borel_logpdf(μ, x::Integer, s::Integer = 1)
    (s < 1 || x < s) && return oftype(float(μ), -Inf)
    return log(s) - log(x) + (x - s) * log(x * μ) - x * μ - logabsgamma(x - s + 1)[1]
end

Distributions.logpdf(d::Borel, n::Integer) = _borel_logpdf(d.μ, n)

Distributions.pdf(d::Borel, n::Integer) = exp(logpdf(d, n))
Distributions.minimum(::Borel) = 1
Distributions.maximum(::Borel) = Inf
Distributions.insupport(::Borel, n::Integer) = n >= 1

function Distributions.mean(d::Borel)
    d.μ >= 1.0 && return Inf
    return 1.0 / (1.0 - d.μ)
end

function Base.rand(rng::AbstractRNG, d::Borel)
    d.μ >= 1.0 && throw(
        ArgumentError(
            "rand is not defined for supercritical Borel (μ ≥ 1): total mass is < 1 and the chain is infinite with positive probability"
        )
    )
    return _inverse_cdf_rand(rng, d, "Borel")
end

"""
Sample an integer from a discrete distribution on `1, 2, …` by walking
the inverse CDF. Used by the chain size distributions defined in this
file, which do not have faster dedicated samplers. Warns and returns
`10_000` if the cumulative mass doesn't reach `u` within 10,000 terms.
"""
function _inverse_cdf_rand(rng::AbstractRNG, d, name::AbstractString)
    u = rand(rng)
    cumprob = 0.0
    for n in 1:10_000
        cumprob += pdf(d, n)
        u <= cumprob && return n
    end
    @warn "$name inverse CDF did not converge in 10,000 terms, returning 10,000"
    return 10_000
end

"""
    GammaBorel(k, R)

Chain size distribution for a NegativeBinomial(k, R) branching process,
derived via Lagrange inversion.

For `R > 1` (supercritical) the PMF is still valid at each `n`, but its
total mass is less than 1: chains are infinite with positive
probability.
"""
struct GammaBorel{T <: Real} <: DiscreteUnivariateDistribution
    k::T
    R::T

    function GammaBorel(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        T = promote_type(typeof(k), typeof(R))
        return new{T}(T(k), T(R))
    end
end

Distributions.params(d::GammaBorel) = (d.k, d.R)

"""
Log-PDF of the GammaBorel distribution. Accepts any numeric type for
k, R (AD-compatible). With `s > 1`, this is the multi-seed
generalisation: the PGF factors as `T(z)^s` and Lagrange inversion
gives `P(X = x | s) = (s/x) * C(kx + x - s - 1, x - s) *
k^(kx) * (k+R)^(s - kx - x) * R^(x - s)`.

The single-seed case (`s = 1`) is a port of `.nbinom_size_ll` in
epichains (Azam, Funk & Finger,
https://github.com/epiverse-trace/epichains, MIT).
"""
function _gammaborel_logpdf(k, R, x::Integer, s::Integer = 1)
    (s < 1 || x < s) && return oftype(float(k), -Inf)
    return (
        log(s) - log(x)
            + logabsgamma(k * x + x - s)[1]
            - logabsgamma(k * x)[1]
            -
            logabsgamma(x - s + 1)[1]
            + k * x * log(k / (k + R))
            + (x - s) * log(R / (k + R))
    )
end

Distributions.logpdf(d::GammaBorel, n::Integer) = _gammaborel_logpdf(d.k, d.R, n)

Distributions.pdf(d::GammaBorel, n::Integer) = exp(logpdf(d, n))
Distributions.minimum(::GammaBorel) = 1
Distributions.maximum(::GammaBorel) = Inf
Distributions.insupport(::GammaBorel, n::Integer) = n >= 1

# Expected total progeny of a subcritical branching process depends only on
# the mean offspring number R, so this matches `mean(::Borel)`.
function Distributions.mean(d::GammaBorel)
    d.R >= 1.0 && return Inf
    return 1.0 / (1.0 - d.R)
end

function Base.rand(rng::AbstractRNG, d::GammaBorel)
    d.R >= 1.0 && throw(
        ArgumentError(
            "rand is not defined for supercritical GammaBorel (R ≥ 1): total mass is < 1 and the chain is infinite with positive probability"
        )
    )
    return _inverse_cdf_rand(rng, d, "GammaBorel")
end

# Alias documenting that `GammaBorel` is the chain size law of NegativeBinomial
# offspring (individual-level Gamma-Poisson mixing). Not exported: construct via
# `chain_size_distribution(::NegativeBinomial)`. `typeof` still prints `GammaBorel`.
const NegativeBinomialChainSize = GammaBorel

"""
    PoissonGammaChainSize(k, R)

Chain size distribution when the per-chain offspring distribution is
`Poisson(λ)` with `λ ~ Gamma(shape = k, mean = R)`. This corresponds
to rate heterogeneity at the chain (cluster) level rather than the
individual level, and matches the `gborel` likelihood in `epichains`.

Note: this is different from `GammaBorel`, which is the chain size
distribution for `NegativeBinomial` offspring (Gamma-Poisson mixing
at the individual level).
"""
struct PoissonGammaChainSize{T <: Real} <: DiscreteUnivariateDistribution
    k::T
    R::T

    function PoissonGammaChainSize(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        T = promote_type(typeof(k), typeof(R))
        return new{T}(T(k), T(R))
    end
end

Distributions.params(d::PoissonGammaChainSize) = (d.k, d.R)

"""
Log-PDF of the PoissonGammaChainSize distribution (AD-compatible).
With `s > 1`, integrates the multi-seed Borel-Tanner PMF over the
Gamma mixing rate: the Borel-Tanner kernel gives a Gamma density in λ,
which integrates in closed form.

The single-seed case (`s = 1`) is a port of `.gborel_size_ll` in
epichains (Azam, Funk & Finger,
https://github.com/epiverse-trace/epichains, MIT).
"""
function _poisson_gamma_logpdf(k, R, x::Integer, s::Integer = 1)
    (s < 1 || x < s) && return oftype(float(k), -Inf)
    return (
        log(s) - log(x)
            + (x - s) * log(x)
            - logabsgamma(x - s + 1)[1]
            +
            logabsgamma(k + x - s)[1]
            -
            logabsgamma(k)[1]
            -
            k * log(R / k)
            -
            (k + x - s) * log(x + k / R)
    )
end

function Distributions.logpdf(d::PoissonGammaChainSize, n::Integer)
    return _poisson_gamma_logpdf(d.k, d.R, n)
end
Distributions.pdf(d::PoissonGammaChainSize, n::Integer) = exp(logpdf(d, n))
Distributions.minimum(::PoissonGammaChainSize) = 1
Distributions.maximum(::PoissonGammaChainSize) = Inf
Distributions.insupport(::PoissonGammaChainSize, n::Integer) = n >= 1

# The mean is infinite for all parameters: the Gamma rate always places density
# at and above 1, where the conditional chain size `1/(1-λ)` diverges, so the
# marginal expectation `E[1/(1-λ)]` does not converge.
Distributions.mean(::PoissonGammaChainSize) = Inf

function Base.rand(::AbstractRNG, ::PoissonGammaChainSize)
    throw(
        ArgumentError(
            "rand is not defined for PoissonGammaChainSize: the chain-size law has positive infinite-chain mass for all parameter values (the Gamma rate always places mass above 1), so finite-chain sampling is ill-defined"
        )
    )
end

"""
    _chain_size_logpdf(d, x, s)

Internal: multi-seed log-PMF of a chain size distribution `d` starting
from `s` independent index cases. Falls back to the single-seed
`logpdf(d, x)` when `s == 1`; otherwise requires a dedicated method
on `d`. Used by `loglikelihood(::ChainSizes, ...)` when the data
carry non-default seed counts.
"""
function _chain_size_logpdf(d, x::Integer, s::Integer)
    s == 1 && return logpdf(d, x)
    throw(
        ArgumentError(
            "multi-seed chain size likelihood not defined for $(typeof(d))"
        )
    )
end

_chain_size_logpdf(d::Borel, x::Integer, s::Integer) = _borel_logpdf(d.μ, x, s)
function _chain_size_logpdf(d::GammaBorel, x::Integer, s::Integer)
    return _gammaborel_logpdf(d.k, d.R, x, s)
end
function _chain_size_logpdf(d::PoissonGammaChainSize, x::Integer, s::Integer)
    return _poisson_gamma_logpdf(d.k, d.R, x, s)
end

"""
    _chain_size_right_tail_logprob(d, x, s)

Internal: multi-seed log-survival of a chain size distribution `d`,
`log P(X ≥ x | s)`, computed as `log(1 − Σ_{j=s}^{x-1} P(X = j | s))`.
For super-critical processes the missing mass `1 − Σ_{j ≥ s} P(X = j)`
is the "chain takes off" probability and is included automatically.

Returns `0.0` when `x ≤ s` (trivially `P(X ≥ s) = 1`) and `-Inf` when
the partial sum reaches or exceeds 1 (numerical underflow of the tail).
Used by the real-time mixture likelihood in
`loglikelihood(::ChainSizes, ...)`.
"""
function _chain_size_right_tail_logprob(d, x::Integer, s::Integer)
    (s < 1 || x < 1) && return -Inf
    x <= s && return zero(_chain_size_logpdf(d, s, s))
    cum = zero(exp(_chain_size_logpdf(d, s, s)))
    for j in s:(x - 1)
        cum += exp(_chain_size_logpdf(d, j, s))
        cum >= one(cum) && return oftype(cum, -Inf)
    end
    p = one(cum) - cum
    return p > zero(p) ? log(p) : oftype(cum, -Inf)
end

"""
    chain_size_distribution(offspring::Poisson)

Analytical chain size distribution for Poisson offspring.
"""
chain_size_distribution(d::Poisson) = Borel(mean(d))

"""
    chain_size_distribution(offspring::NegativeBinomial)

Analytical chain size distribution for NegativeBinomial offspring.
"""
chain_size_distribution(d::NegativeBinomial) = GammaBorel(d.r, mean(d))

# `chain_size_distribution(model::TransmissionModel; ...)` lives in
# `src/likelihood_dists.jl` so the kwargs-bearing wrapper path and the
# analytical fallback share a single definition.

"""
    IndexChainSize(index_offspring, offspring)

Chain-size distribution for a chain whose index case draws its number of
secondary cases from `index_offspring`, while every later case draws from
`offspring`. Useful when the index case's opportunity to transmit differs
from that of a locally infected case — for example a chain seeded by an
introduced case who arrives part-way through their infectious period or is
quarantined on arrival.

    P(N = n) = P(J = 0) 1{n = 1} + Σ_{j ≥ 1} P(J = j) P(chains from j seeds have n - 1 cases)

where `J ~ index_offspring` is the index case's secondary case count and the
`j`-seed term is the multi-seed chain-size law built from `offspring` via
[`chain_size_distribution`](@ref) (the same closed form used for
multi-seed [`ChainSizes`](@ref)). This requires `chain_size_distribution(offspring)`
to have a multi-seed closed form: `Poisson`, `NegativeBinomial`, and
`ClusterMixed(Poisson, ::Gamma)` all resolve to one, but a general
[`ClusterMixed`](@ref) without a closed form resolves to
[`ChainSizeMixture`](@ref), which has none, and `logpdf`/`pdf` throw once
`n` is large enough that the sum reaches a `j ≥ 2` term.

Only single-index-case chains are supported (`seeds == 1` in
[`ChainSizes`](@ref)): the multi-seed formula for a cluster with several
independently introduced cases is not defined here.

# Examples

```julia
d = IndexChainSize(Poisson(0.3), NegBin(0.8, 0.5))
pdf(d, 5)
loglikelihood(ChainSizes([1, 2, 5, 1]), d)
```
"""
struct IndexChainSize{J <: DiscreteUnivariateDistribution, S <: DiscreteUnivariateDistribution} <:
    DiscreteUnivariateDistribution
    index_offspring::J
    dist::S

    function IndexChainSize(index_offspring::DiscreteUnivariateDistribution, offspring)
        dist = chain_size_distribution(offspring)
        return new{typeof(index_offspring), typeof(dist)}(index_offspring, dist)
    end
end

Distributions.minimum(::IndexChainSize) = 1
Distributions.maximum(::IndexChainSize) = Inf
Distributions.insupport(::IndexChainSize, n::Integer) = n >= 1

"""
Log-PDF of [`IndexChainSize`](@ref). The sum over the index case's
secondary-case count `j` only needs `j` up to `n - 1` (a chain of `j`
non-index seeds has at least `j` cases), so it is a finite loop rather
than a truncated infinite series.
"""
function Distributions.logpdf(d::IndexChainSize, n::Integer)
    n < 1 && return -Inf
    p0 = pdf(d.index_offspring, 0)
    total = n == 1 ? float(p0) : zero(float(p0))
    for j in 1:(n - 1)
        pj = pdf(d.index_offspring, j)
        pj == 0 && continue
        total += pj * exp(_chain_size_logpdf(d.dist, n - 1, j))
    end
    return total > zero(total) ? log(total) : oftype(total, -Inf)
end

Distributions.pdf(d::IndexChainSize, n::Integer) = exp(logpdf(d, n))

# E[N] = P(J=0)·1 + Σ_{j≥1} P(J=j)·E[1 + size(j seeds)]
#      = 1 + E[J]·E[dist], since a j-seed chain's expected size is j·E[dist].
# An index case with E[J] = 0 never seeds later cases, so N ≡ 1 regardless
# of E[dist]; guard this so 0 * Inf does not turn into NaN.
function Distributions.mean(d::IndexChainSize)
    m = mean(d.index_offspring)
    m == 0 && return oftype(float(m), 1)
    return 1 + m * mean(d.dist)
end

"""
Sample a chain size: draw the index case's secondary-case count `j`, then
sum `j` independent draws from the non-index chain-size law (the total size
of `j` independent chains is the sum of `j` iid single-seed chain sizes).
Throws if `dist` is supercritical, as for `rand` on the underlying laws.
"""
function Base.rand(rng::AbstractRNG, d::IndexChainSize)
    j = rand(rng, d.index_offspring)
    j == 0 && return 1
    return 1 + sum(rand(rng, d.dist) for _ in 1:j)
end

"""
    TruncatedChainSize(base, min_size)

Chain-size law of a cluster recorded only once it reaches `min_size` cases:
`P(N = n | N ≥ min_size) = P(N = n) / P(N ≥ min_size)` under `base`, and zero
density below `min_size`. [`observe`](@ref) builds it from a
[`MinimumSize`](@ref) observation, and the conditioning follows each cluster's
own seed count through the multi-seed helpers.
"""
struct TruncatedChainSize{D <: DiscreteUnivariateDistribution} <:
    DiscreteUnivariateDistribution
    base::D
    min_size::Int
end

Distributions.minimum(d::TruncatedChainSize) = d.min_size
Distributions.maximum(::TruncatedChainSize) = Inf
Distributions.insupport(d::TruncatedChainSize, n::Integer) = n >= d.min_size
Distributions.logpdf(d::TruncatedChainSize, n::Integer) = _chain_size_logpdf(d, n, 1)

function _chain_size_logpdf(d::TruncatedChainSize, x::Integer, s::Integer)
    x >= d.min_size || return oftype(_chain_size_logpdf(d.base, max(x, 1), 1), -Inf)
    denom = _chain_size_right_tail_logprob(d.base, d.min_size, s)
    isfinite(denom) || return oftype(denom, -Inf)
    return _chain_size_logpdf(d.base, x, s) - denom
end

# `log P(X ≥ x | X ≥ min_size)`: the base's survival at whichever of the two
# bounds binds, against the survival the conditioning divides by. When the
# denominator itself has underflowed to 0 (`-Inf` on the log scale, per
# `_chain_size_right_tail_logprob`'s own underflow convention), the
# conditioning event is numerically indistinguishable from impossible, so
# fall back to `-Inf` rather than letting `-Inf - (-Inf)` produce `NaN`.
function _chain_size_right_tail_logprob(d::TruncatedChainSize, x::Integer, s::Integer)
    denom = _chain_size_right_tail_logprob(d.base, d.min_size, s)
    x <= d.min_size && return zero(denom)
    isfinite(denom) || return oftype(denom, -Inf)
    return _chain_size_right_tail_logprob(d.base, max(x, d.min_size), s) - denom
end
