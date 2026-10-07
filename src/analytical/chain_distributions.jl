"""
    Borel(μ)

Chain size distribution when each case infects a Poisson number of others with
mean `μ`: the probability that a chain started by one case has `n` cases in
total,

    P(X = n) = (μn)^(n-1) * exp(-μn) / n!  for n = 1, 2, ...

[`chain_size_distribution`](@ref)`(Poisson(μ))` returns it.

When `μ > 1` some chains grow without end, so the probabilities of the finite
sizes add up to less than 1; the missing probability is that of a major
outbreak. The formula stays valid for each `n`, which lets chain-size
probabilities be averaged over values of `μ` either side of 1.
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

Chain size distribution when each case infects a negative binomial number of
others with dispersion `k` and mean `R`: the probability that a chain started
by one case has `n` cases in total. [`chain_size_distribution`](@ref)`(NegBin(R, k))`
returns it. It matches `.nbinom_size_ll` in epichains.

!!! warning "Argument order is (k, R)"
    `GammaBorel(k, R)` takes the dispersion first, the reverse of
    [`NegBin`](@ref)`(R, k)`. `GammaBorel(0.5, 0.8)` is the chain size
    distribution for `NegBin(0.8, 0.5)`. Building it through
    `chain_size_distribution(NegBin(R, k))` avoids the mix-up.

When `R > 1` some chains grow without end, so the probabilities of the finite
sizes add up to less than 1; the missing probability is that of a major
outbreak.
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

Chain size distribution when the reproduction number varies between chains
rather than between cases: each chain draws its own rate `λ` from a Gamma
distribution with shape `k` and mean `R`, and every case in that chain infects
a Poisson(`λ`) number of others. It matches the `gborel` likelihood in
epichains. Like [`GammaBorel`](@ref EpiBranch.GammaBorel), it takes `k` first.

This differs from `GammaBorel`, where each case has its own reproduction
number (negative binomial offspring). [`chain_size_distribution`](@ref)
returns it for `ClusterMixed(Poisson, Gamma(...))`.
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

Exact chain size distribution, the probability that a chain started by one
case has `n` cases in total, when each case infects a Poisson number of
others: a [`Borel`](@ref) distribution. When R > 1 the probabilities of
finite sizes add up to less than 1, the rest being the probability of a major
outbreak.

# Examples

```julia
d = chain_size_distribution(Poisson(0.8))
pdf(d, 3)   # probability a chain has exactly 3 cases
```
"""
chain_size_distribution(d::Poisson) = Borel(mean(d))

"""
    chain_size_distribution(offspring::NegativeBinomial)

Exact chain size distribution, the probability that a chain started by one
case has `n` cases in total, when each case infects a negative binomial
number of others (a [`GammaBorel`](@ref EpiBranch.GammaBorel) distribution).
When R > 1 the probabilities of finite sizes add up to less than 1, the rest
being the probability of a major outbreak.

# Examples

```julia
d = chain_size_distribution(NegBin(0.8, 0.5))
pdf(d, 3)                   # probability a chain has exactly 3 cases
sum(pdf(d, n) for n in 1:10)  # probability a chain has at most 10 cases
```
"""
chain_size_distribution(d::NegativeBinomial) = GammaBorel(d.r, mean(d))

# `chain_size_distribution(model::TransmissionModel; ...)` lives in
# `src/likelihood_dists.jl` so the kwargs-bearing wrapper path and the
# analytical fallback share a single definition.

"""
    IndexChainSize(index_offspring, offspring)

Chain size distribution when the index case has a different offspring
distribution from later cases: the index case infects a number drawn from
`index_offspring`, and every later case a number drawn from `offspring`. Use
it when an introduced case transmits less than locally infected cases, for
example because they arrive part-way through their infectious period or are
quarantined on arrival.

Only chains started by a single index case are supported (`seeds == 1` in
[`ChainSizes`](@ref)).

!!! note
    The probability of a chain of `n` cases sums over the number `j` the index
    case infected, each starting its own chain:

        P(N = n) = P(J = 0) 1{n = 1} + Σ_{j ≥ 1} P(J = j) P(chains from j cases have n - 1 cases)

    This needs the chain size distribution of `offspring` from several
    starting cases, which exists for `Poisson`, `NegativeBinomial` and
    `ClusterMixed(Poisson, ::Gamma)`. For other [`ClusterMixed`](@ref)
    offspring, `pdf` and `logpdf` give an error once `n` is large enough to
    need it.

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
sum `j` independent draws from the non-index chain size distribution (the total size
of `j` independent chains is the sum of `j` iid single-seed chain sizes).
Throws if `dist` is supercritical, as for `rand` on the underlying distributions.
"""
function Base.rand(rng::AbstractRNG, d::IndexChainSize)
    j = rand(rng, d.index_offspring)
    j == 0 && return 1
    return 1 + sum(rand(rng, d.dist) for _ in 1:j)
end

"""
    TruncatedChainSize(base, min_size)

Chain size distribution for clusters recorded only once they reach
`min_size` cases (for example when only clusters of two or more cases are
investigated): the distribution `base`, conditioned on the size being at
least `min_size`,

    P(N = n | N ≥ min_size) = P(N = n) / P(N ≥ min_size)

and zero below `min_size`. [`observe`](@ref) builds it from a
[`MinimumSize`](@ref) observation model. Clusters with several index cases are
conditioned using their own number of index cases.
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
# denominator has underflowed to 0 (`-Inf` on the log scale, following
# `_chain_size_right_tail_logprob`'s own underflow convention), the
# conditioning event has effectively zero probability, so return `-Inf`
# directly rather than let `-Inf - (-Inf)` give `NaN`.
function _chain_size_right_tail_logprob(d::TruncatedChainSize, x::Integer, s::Integer)
    denom = _chain_size_right_tail_logprob(d.base, d.min_size, s)
    x <= d.min_size && return zero(denom)
    isfinite(denom) || return oftype(denom, -Inf)
    return _chain_size_right_tail_logprob(d.base, max(x, d.min_size), s) - denom
end
