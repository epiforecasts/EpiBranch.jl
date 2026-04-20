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
struct Borel{T <: AbstractFloat} <: DiscreteUnivariateDistribution
    μ::T

    function Borel(μ::Real)
        0.0 < μ || throw(ArgumentError("μ must be positive, got $μ"))
        new{typeof(float(μ))}(float(μ))
    end
end

Distributions.params(d::Borel) = (d.μ,)

"""Log-PDF of the Borel distribution. Accepts any numeric type for μ (AD-compatible)."""
function _borel_logpdf(μ, n::Integer)
    n < 1 ? oftype(float(μ), -Inf) : (n - 1) * log(μ * n) - μ * n - logabsgamma(n + 1)[1]
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
struct GammaBorel{T <: AbstractFloat} <: DiscreteUnivariateDistribution
    k::T
    R::T

    function GammaBorel(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        T = float(promote_type(typeof(k), typeof(R)))
        new{T}(T(k), T(R))
    end
end

Distributions.params(d::GammaBorel) = (d.k, d.R)

"""Log-PDF of the GammaBorel distribution. Accepts any numeric type for k, R (AD-compatible)."""
function _gammaborel_logpdf(k, R, n::Integer)
    n < 1 && return oftype(float(k), -Inf)
    return (logabsgamma(k * n + n - 1)[1]
            - logabsgamma(k * n)[1]
            - logabsgamma(n + 1)[1]
            + k * n * log(k / (k + R))
            + (n - 1) * log(R / (k + R)))
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
struct PoissonGammaChainSize{T <: AbstractFloat} <: DiscreteUnivariateDistribution
    k::T
    R::T

    function PoissonGammaChainSize(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        T = float(promote_type(typeof(k), typeof(R)))
        new{T}(T(k), T(R))
    end
end

Distributions.params(d::PoissonGammaChainSize) = (d.k, d.R)

"""Log-PDF of the PoissonGammaChainSize distribution (AD-compatible)."""
function _poisson_gamma_logpdf(k, R, n::Integer)
    n < 1 && return oftype(float(k), -Inf)
    return (logabsgamma(k + n - 1)[1]
            - logabsgamma(n + 1)[1]
            - logabsgamma(k)[1]
            -
            k * log(R / k)
            +
            (n - 1) * log(n)
            -
            (k + n - 1) * log(n + k / R))
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
    chain_size_distribution(model::BranchingProcess)

Analytical chain size distribution extracted from the model's offspring
distribution (single-type only).
"""
function chain_size_distribution(model::BranchingProcess)
    return chain_size_distribution(_single_type_offspring(model))
end
