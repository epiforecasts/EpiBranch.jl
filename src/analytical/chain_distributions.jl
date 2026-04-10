"""
    Borel(μ)

The Borel distribution with parameter `μ` (0 < μ ≤ 1).

P(X = n) = (μn)^(n-1) * exp(-μn) / n!  for n = 1, 2, ...

This is the chain size distribution for a Poisson(μ) branching process.
μ = 0 is excluded because the distribution is degenerate (always 1)
and would cause log(0) in the PMF.
"""
struct Borel{T <: AbstractFloat} <: DiscreteUnivariateDistribution
    μ::T

    function Borel(μ::Real)
        0.0 < μ || throw(ArgumentError("μ must be positive, got $μ"))
        μ <= 1.0 || throw(ArgumentError("μ must be ≤ 1, got $μ"))
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
    u = rand(rng)
    cumprob = 0.0
    for n in 1:10_000
        cumprob += pdf(d, n)
        u <= cumprob && return n
    end
    return 10_000  # truncation fallback
end

"""
    GammaBorel(k, R)

Chain size distribution for a NegativeBinomial(k, R) branching process,
derived via Lagrange inversion.
"""
struct GammaBorel{T <: AbstractFloat} <: DiscreteUnivariateDistribution
    k::T
    R::T

    function GammaBorel(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        R <= 1.0 ||
            @warn "GammaBorel with R > 1 (supercritical): PMF does not sum to 1, chain size is infinite with positive probability"
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
    u = rand(rng)
    cumprob = 0.0
    for n in 1:10_000
        cumprob += pdf(d, n)
        u <= cumprob && return n
    end
    return 10_000
end

"""
    chain_size_distribution(offspring::Poisson)

Analytical chain size distribution for Poisson offspring.
"""
chain_size_distribution(d::Poisson) = Borel(min(mean(d), 1.0))

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
