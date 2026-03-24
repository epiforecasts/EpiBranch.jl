"""
    NegBin(R, k)

Convenience constructor for a Negative Binomial offspring distribution
parameterised by mean reproduction number `R` and dispersion parameter `k`.

A `NegativeBinomial` from Distributions.jl is returned, with mean `R` and
variance `R + R²/k`.

_Note:_ `NegativeBinomial(r, p)` from Distributions.jl uses a different
parameterisation (number of successes and success probability). Using it
directly as an offspring distribution will produce silently wrong results.
Always use `NegBin(R, k)` for epidemiological parameterisation.
"""
function NegBin(R::Real, k::Real)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    p = k / (k + R)
    return NegativeBinomial(k, p)
end

"""
    ringbp_generation_time(; presymptomatic_fraction=0.3, omega=2.0)

Return a function suitable for the `generation_time` field of a
`BranchingProcess`, implementing ringbp's incubation-linked generation
time model.

The returned function takes an incubation period (Float64) and produces a
truncated skew-normal distribution SN(ξ, ω, α), where ξ = incubation
period, and α is chosen so that the fraction of generation times shorter
than the incubation period equals `presymptomatic_fraction`. This matches
the generation time model in ringbp (Hellewell et al. 2020).

Usage:
```julia
model = BranchingProcess(
    NegBin(2.5, 0.16),
    ringbp_generation_time(presymptomatic_fraction=0.3)
)
```
"""
function ringbp_generation_time(; presymptomatic_fraction::Real=0.3,
                                  omega::Real=2.0)
    0.0 < presymptomatic_fraction < 1.0 || throw(ArgumentError(
        "presymptomatic_fraction must be in (0, 1), got $presymptomatic_fraction"))
    omega > 0.0 || throw(ArgumentError("omega must be positive, got $omega"))

    # Compute skew-normal alpha from presymptomatic fraction
    # For SN(xi, omega, alpha): P(X < xi) = 0.5 - arctan(alpha)/π
    # => alpha = tan(π(0.5 - presymp_frac))
    alpha = tan(Float64(π) * (0.5 - Float64(presymptomatic_fraction)))
    om = Float64(omega)

    return function (inc_period::Float64)
        _TruncatedSkewNormal(inc_period, om, alpha)
    end
end

"""Skew-normal truncated to [0, ∞) via rejection sampling (cdf not available)."""
struct _TruncatedSkewNormal <: ContinuousUnivariateDistribution
    ξ::Float64
    ω::Float64
    α::Float64
    inner::SkewNormal{Float64}
    _TruncatedSkewNormal(ξ, ω, α) = new(ξ, ω, α, SkewNormal(ξ, ω, α))
end

function Base.rand(rng::AbstractRNG, d::_TruncatedSkewNormal)
    for _ in 1:10_000
        x = rand(rng, d.inner)
        x >= 0.0 && return x
    end
    return 0.0  # fallback
end

Distributions.logpdf(d::_TruncatedSkewNormal, x::Real) =
    x < 0.0 ? -Inf : logpdf(d.inner, x)
