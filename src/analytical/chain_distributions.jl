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
        new{typeof(μ)}(μ)
    end
end

Distributions.params(d::Borel) = (d.μ,)

"""
Log-PDF of the Borel distribution. Accepts any numeric type for μ
(AD-compatible). With `s > 1`, this is the Borel-Tanner generalisation
for the total chain size starting from `s` independent index cases:
`P(X = x | s, μ) = (s/x) * (xμ)^(x-s) * exp(-xμ) / (x-s)!`.
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
    d.μ >= 1.0 && throw(ArgumentError(
        "rand is not defined for supercritical Borel (μ ≥ 1): total mass is < 1 and the chain is infinite with positive probability"))
    _inverse_cdf_rand(rng, d, "Borel")
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
        new{T}(T(k), T(R))
    end
end

Distributions.params(d::GammaBorel) = (d.k, d.R)

"""
Log-PDF of the GammaBorel distribution. Accepts any numeric type for
k, R (AD-compatible). With `s > 1`, this is the multi-seed
generalisation: the PGF factors as `T(z)^s` and Lagrange inversion
gives `P(X = x | s) = (s/x) * C(kx + x - s - 1, x - s) *
k^(kx) * (k+R)^(s - kx - x) * R^(x - s)`.
"""
function _gammaborel_logpdf(k, R, x::Integer, s::Integer = 1)
    (s < 1 || x < s) && return oftype(float(k), -Inf)
    return (log(s) - log(x)
            + logabsgamma(k * x + x - s)[1]
            - logabsgamma(k * x)[1]
            -
            logabsgamma(x - s + 1)[1]
            + k * x * log(k / (k + R))
            + (x - s) * log(R / (k + R)))
end

Distributions.logpdf(d::GammaBorel, n::Integer) = _gammaborel_logpdf(d.k, d.R, n)

Distributions.pdf(d::GammaBorel, n::Integer) = exp(logpdf(d, n))
Distributions.minimum(::GammaBorel) = 1
Distributions.maximum(::GammaBorel) = Inf
Distributions.insupport(::GammaBorel, n::Integer) = n >= 1

function Base.rand(rng::AbstractRNG, d::GammaBorel)
    d.R >= 1.0 && throw(ArgumentError(
        "rand is not defined for supercritical GammaBorel (R ≥ 1): total mass is < 1 and the chain is infinite with positive probability"))
    _inverse_cdf_rand(rng, d, "GammaBorel")
end

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
        new{T}(T(k), T(R))
    end
end

Distributions.params(d::PoissonGammaChainSize) = (d.k, d.R)

"""
Log-PDF of the PoissonGammaChainSize distribution (AD-compatible).
With `s > 1`, integrates the multi-seed Borel-Tanner PMF over the
Gamma mixing rate: the Borel-Tanner kernel gives a Gamma density in λ,
which integrates in closed form.
"""
function _poisson_gamma_logpdf(k, R, x::Integer, s::Integer = 1)
    (s < 1 || x < s) && return oftype(float(k), -Inf)
    return (log(s) - log(x)
            + (x - s) * log(x)
            - logabsgamma(x - s + 1)[1]
            +
            logabsgamma(k + x - s)[1]
            -
            logabsgamma(k)[1]
            -
            k * log(R / k)
            -
            (k + x - s) * log(x + k / R))
end

function Distributions.logpdf(d::PoissonGammaChainSize, n::Integer)
    _poisson_gamma_logpdf(d.k, d.R, n)
end
Distributions.pdf(d::PoissonGammaChainSize, n::Integer) = exp(logpdf(d, n))
Distributions.minimum(::PoissonGammaChainSize) = 1
Distributions.maximum(::PoissonGammaChainSize) = Inf
Distributions.insupport(::PoissonGammaChainSize, n::Integer) = n >= 1

function Base.rand(rng::AbstractRNG, d::PoissonGammaChainSize)
    d.R >= 1.0 && throw(ArgumentError(
        "rand is not defined for PoissonGammaChainSize with mean R ≥ 1: too much Gamma mass sits above 1 (supercritical rates) and chains are infinite with positive probability"))
    _inverse_cdf_rand(rng, d, "PoissonGammaChainSize")
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
    throw(ArgumentError(
        "multi-seed chain size likelihood not defined for $(typeof(d))"))
end

_chain_size_logpdf(d::Borel, x::Integer, s::Integer) = _borel_logpdf(d.μ, x, s)
function _chain_size_logpdf(d::GammaBorel, x::Integer, s::Integer)
    _gammaborel_logpdf(d.k, d.R, x, s)
end
function _chain_size_logpdf(d::PoissonGammaChainSize, x::Integer, s::Integer)
    _poisson_gamma_logpdf(d.k, d.R, x, s)
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

"""
    chain_size_distribution(model::TransmissionModel)

Analytical chain size distribution extracted from the model's offspring
specification via `_single_type_offspring`. Works for `BranchingProcess`
and any wrapper that delegates that accessor (e.g. `PartiallyObserved`
delegates before applying its own transformed distribution).
"""
function chain_size_distribution(model::TransmissionModel)
    return chain_size_distribution(_single_type_offspring(model))
end
