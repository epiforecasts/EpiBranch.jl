"""
    Borel(μ)

The Borel distribution with parameter `μ` (0 < μ ≤ 1).

P(X = n) = (μn)^(n-1) * exp(-μn) / n!  for n = 1, 2, ...

This is the chain size distribution for a Poisson(μ) branching process.
Implemented as a subtype of Distributions.DiscreteUnivariateDistribution.
"""
struct Borel <: DiscreteUnivariateDistribution
    μ::Float64

    function Borel(μ::Real)
        0.0 < μ || throw(ArgumentError("μ must be positive, got $μ"))
        μ <= 1.0 || throw(ArgumentError("μ must be ≤ 1, got $μ"))
        new(Float64(μ))
    end
end

Distributions.params(d::Borel) = (d.μ,)

function Distributions.logpdf(d::Borel, n::Integer)
    n < 1 && return -Inf
    μ = d.μ
    return (n - 1) * log(μ * n) - μ * n - logabsgamma(n + 1)[1]
end

Distributions.pdf(d::Borel, n::Integer) = exp(logpdf(d, n))

Distributions.minimum(::Borel) = 1
Distributions.maximum(::Borel) = Inf
Distributions.insupport(::Borel, n::Integer) = n >= 1

function Distributions.mean(d::Borel)
    d.μ >= 1.0 && return Inf
    return 1.0 / (1.0 - d.μ)
end

"""
    GammaBorel(k, R)

The Gamma-Borel mixture distribution (chain size distribution for
a NegativeBinomial(k, R) branching process).

This is a Gamma mixture of Borel distributions where μ ~ Gamma(k, R/k).
"""
struct GammaBorel <: DiscreteUnivariateDistribution
    k::Float64
    R::Float64

    function GammaBorel(k::Real, R::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        new(Float64(k), Float64(R))
    end
end

Distributions.params(d::GammaBorel) = (d.k, d.R)

function Distributions.logpdf(d::GammaBorel, n::Integer)
    n < 1 && return -Inf
    k, R = d.k, d.R
    # Chain size distribution for NegBin(k, R) offspring:
    # P(X = n) = Γ(kn + n - 1) / (Γ(kn) * Γ(n + 1)) * (k/(k+R))^(kn) * (R/(k+R))^(n-1)
    # Derived from Lagrange inversion / Gamma mixture of Borel
    return (logabsgamma(k * n + n - 1)[1]
            - logabsgamma(k * n)[1]
            - logabsgamma(n + 1)[1]
            + k * n * log(k / (k + R))
            + (n - 1) * log(R / (k + R)))
end

Distributions.pdf(d::GammaBorel, n::Integer) = exp(logpdf(d, n))

Distributions.minimum(::GammaBorel) = 1
Distributions.maximum(::GammaBorel) = Inf
Distributions.insupport(::GammaBorel, n::Integer) = n >= 1

"""
    chain_size_distribution(offspring::Poisson)

Return the analytical chain size distribution for a Poisson offspring distribution.
"""
chain_size_distribution(d::Poisson) = Borel(min(mean(d), 1.0))

"""
    chain_size_distribution(offspring::NegativeBinomial)

Return the analytical chain size distribution for a NegativeBinomial offspring distribution.
"""
function chain_size_distribution(d::NegativeBinomial)
    k = d.r
    R = mean(d)
    GammaBorel(k, R)
end

"""
    chain_size_distribution(model::BranchingProcess)

Return the analytical chain size distribution for a single-type branching
process model, extracted from its offspring distribution.
"""
function chain_size_distribution(model::BranchingProcess)
    model.offspring isa Distribution || throw(ArgumentError(
        "Analytical chain size distribution only available for single-type models"))
    return chain_size_distribution(model.offspring)
end
