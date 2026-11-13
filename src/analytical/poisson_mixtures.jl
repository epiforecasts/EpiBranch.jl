# ── Poisson mixtures with a heavy-tailed rate ─────────────────────────
# Offspring laws where each case draws its own rate from a lognormal or
# Weibull distribution and then a Poisson count at that rate: alternatives to
# the negative binomial (a Poisson-gamma mixture) for individual variation in
# transmission. Neither has a closed-form mass function, so quantities are
# integrals over the rate, taken on the probability scale of the mixing law.

"""
    PoissonLognormal(μ, σ)

Offspring distribution in which a case's rate is lognormal, with mean `μ` and
standard deviation `σ` of the logarithm (as in `LogNormal(μ, σ)`), and its
number of secondary cases is Poisson at that rate. The mean is
`exp(μ + σ² / 2)`. It is an alternative to the negative binomial
([`NegBin`](@ref)) with a heavier right tail.

The mass function, cumulative distribution and generating function are
evaluated by numerical integration over the rate. Chain sizes and extinction
use the generating function, so they work as for any discrete law
([`chain_size_distribution`](@ref), [`extinction_probability`](@ref)).

# Examples

```julia
d = PoissonLognormal(-0.5, 1.0)
pdf(d, 3)
loglikelihood(ChainSizes([1, 2, 1, 5]), d)
```
"""
struct PoissonLognormal{T <: Real} <: DiscreteUnivariateDistribution
    μ::T
    σ::T
    function PoissonLognormal{T}(μ::T, σ::T) where {T <: Real}
        isfinite(μ) || throw(ArgumentError("μ must be finite, got $μ"))
        (isfinite(σ) && σ > 0) || throw(ArgumentError("σ must be positive, got $σ"))
        return new{T}(μ, σ)
    end
end
PoissonLognormal(μ::Real, σ::Real) = PoissonLognormal{float(promote_type(typeof(μ), typeof(σ)))}(μ, σ)

"""
    PoissonWeibull(k, λ)

Offspring distribution in which a case's rate is Weibull with shape `k` and
scale `λ` (as in `Weibull(k, λ)`), and its number of secondary cases is
Poisson at that rate. The mean is `λ Γ(1 + 1 / k)`. A shape `k` below 1 gives
a heavier tail than the exponential, and so more superspreading.

Like [`PoissonLognormal`](@ref), it is evaluated by numerical integration over
the rate.

# Examples

```julia
d = PoissonWeibull(0.7, 0.5)
pdf(d, 3)
extinction_probability(PoissonWeibull(0.7, 2.0))
```
"""
struct PoissonWeibull{T <: Real} <: DiscreteUnivariateDistribution
    k::T
    λ::T
    function PoissonWeibull{T}(k::T, λ::T) where {T <: Real}
        (isfinite(k) && k > 0) || throw(ArgumentError("k must be positive, got $k"))
        (isfinite(λ) && λ > 0) || throw(ArgumentError("λ must be positive, got $λ"))
        return new{T}(k, λ)
    end
end
PoissonWeibull(k::Real, λ::Real) = PoissonWeibull{float(promote_type(typeof(k), typeof(λ)))}(k, λ)

const _PoissonMixture = Union{PoissonLognormal, PoissonWeibull}

# The distribution of the Poisson rate.
_rate_law(d::PoissonLognormal) = LogNormal(d.μ, d.σ)
_rate_law(d::PoissonWeibull) = Weibull(d.k, d.λ)

Distributions.params(d::PoissonLognormal) = (d.μ, d.σ)
Distributions.params(d::PoissonWeibull) = (d.k, d.λ)
Distributions.partype(::PoissonLognormal{T}) where {T} = T
Distributions.partype(::PoissonWeibull{T}) where {T} = T

Distributions.minimum(::_PoissonMixture) = 0
Distributions.maximum(::_PoissonMixture) = Inf
Distributions.insupport(::_PoissonMixture, x::Integer) = x >= 0
Distributions.mean(d::_PoissonMixture) = mean(_rate_law(d))
Distributions.var(d::_PoissonMixture) = mean(d) + var(_rate_law(d))

# Expectation of `f(Poisson(λ), x)` over the rate, integrated over the
# probability scale of the rate law. The integrand is sharply peaked where the
# rate is near `x`, so the interval is split there.
function _poisson_mixture_expectation(f, d::_PoissonMixture, x::Real)
    rate = _rate_law(d)
    integrand = u -> f(Poisson(quantile(rate, u)), x)
    split = clamp(cdf(rate, x), 1.0e-12, 1 - 1.0e-12)
    value, _ = quadgk(integrand, 0, split, 1; rtol = 1.0e-10, atol = 1.0e-300)
    return value
end

function Distributions.pdf(d::_PoissonMixture, x::Integer)
    x < 0 && return 0.0
    return _poisson_mixture_expectation(pdf, d, x)
end
function Distributions.logpdf(d::_PoissonMixture, x::Integer)
    p = pdf(d, x)
    return p > 0 ? log(p) : oftype(p, -Inf)
end
function Distributions.cdf(d::_PoissonMixture, x::Integer)
    x < 0 && return 0.0
    return min(_poisson_mixture_expectation(cdf, d, x), 1.0)
end
function Distributions.ccdf(d::_PoissonMixture, x::Integer)
    x < 0 && return 1.0
    return _poisson_mixture_expectation(ccdf, d, x)
end

function Base.rand(rng::AbstractRNG, d::_PoissonMixture)
    return rand(rng, Poisson(rand(rng, _rate_law(d))))
end

# Generating function E[exp(λ(s - 1))] and its derivative E[λ exp(λ(s - 1))]
# over the rate, for `s` in [0, 1].
function _pgf(d::_PoissonMixture, s)
    rate = _rate_law(d)
    value, _ = quadgk(
        u -> (λ = quantile(rate, u); exp(λ * (s - 1))), 0, 1;
        rtol = 1.0e-10, atol = 1.0e-300
    )
    return value
end
function _pgf_derivative(d::_PoissonMixture, s)
    rate = _rate_law(d)
    value, _ = quadgk(
        u -> (λ = quantile(rate, u); λ * exp(λ * (s - 1))), 0, 1;
        rtol = 1.0e-10, atol = 1.0e-300
    )
    return value
end
