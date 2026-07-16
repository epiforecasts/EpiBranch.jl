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
    incubation_linked_generation_time(; presymptomatic_fraction=0.3, omega=2.0)

Return a function suitable for the `generation_time` field of a
`BranchingProcess`, in which each individual's generation time is linked
to their own incubation period.

The returned function takes an `Individual` and produces a truncated
skew-normal distribution SN(ξ, ω, α), where ξ is the individual's
incubation period and α is chosen so that the fraction of generation
times shorter than the incubation period equals `presymptomatic_fraction`.
The inversion `α = tan(π(0.5 − presymptomatic_fraction))` is exact for the
untruncated skew-normal; after truncation to `[0, ∞)` the realised fraction is
approximate, holding closely when the incubation period is large relative to
`omega` and diverging for short incubation periods.
This is the generation time model used in Hellewell et al. (2020).
Individuals with no usable incubation period (for example asymptomatic
cases) fall back to a 5-day centre.

Usage:
```julia
model = BranchingProcess(
    NegBin(2.5, 0.16),
    incubation_linked_generation_time(presymptomatic_fraction=0.3)
)
```
"""
function incubation_linked_generation_time(; presymptomatic_fraction::Real = 0.3,
        omega::Real = 2.0)
    0.0 < presymptomatic_fraction < 1.0 || throw(ArgumentError(
        "presymptomatic_fraction must be in (0, 1), got $presymptomatic_fraction"))
    omega > 0.0 || throw(ArgumentError("omega must be positive, got $omega"))

    # Compute skew-normal alpha from presymptomatic fraction
    # For SN(xi, omega, alpha): P(X < xi) = 0.5 - arctan(alpha)/π
    # => alpha = tan(π(0.5 - presymp_frac))
    alpha = tan(float(π) * (0.5 - float(presymptomatic_fraction)))
    om = float(omega)

    return function (individual)
        inc_period = incubation_period(individual)
        if isnan(inc_period) || inc_period <= 0.0
            @debug "Missing or non-positive incubation period (e.g. asymptomatic individual); using 5.0 days" maxlog=1
            inc_period = 5.0
        end
        _TruncatedSkewNormal(inc_period, om, alpha)
    end
end

"""Skew-normal truncated to [0, ∞) via rejection sampling (cdf not available).
Carries `logZ = log P(inner ≥ 0)` so `logpdf` is properly normalised over the
truncated support."""
struct _TruncatedSkewNormal{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T
    inner::SkewNormal{T}
    logZ::T
    function _TruncatedSkewNormal(ξ::Real, ω::Real, α::Real)
        T = float(promote_type(typeof(ξ), typeof(ω), typeof(α)))
        inner = SkewNormal(T(ξ), T(ω), T(α))
        # SkewNormal has no cdf in this Distributions.jl version, so integrate
        # the retained mass on [0, ∞) numerically for the truncation constant.
        Z = first(quadgk(x -> pdf(inner, x), zero(T), T(Inf)))
        new{T}(T(ξ), T(ω), T(α), inner, T(log(Z)))
    end
end

function Base.rand(rng::AbstractRNG, d::_TruncatedSkewNormal)
    for _ in 1:10_000
        x = rand(rng, d.inner)
        x >= 0.0 && return x
    end
    @warn "rejection sampling for TruncatedSkewNormal failed after 10,000 attempts, returning 0.0"
    return 0.0
end

# Normalised over [0, ∞): subtract the retained-mass constant so the density
# integrates to 1 (the bare inner density does not on the truncated support).
function Distributions.logpdf(d::_TruncatedSkewNormal, x::Real)
    x < 0.0 ? oftype(float(x), -Inf) : logpdf(d.inner, x) - d.logZ
end

"""
    _sample_value(x, rng, args...) -> Float64

Resolve a value that can be a `Real`, a `Distribution`, or a callable.
The callable is invoked as `f(rng, args...)` — callers choose the
signature by what they pass after `rng`. The return is always
converted to `Float64`.

Used throughout the package for parameters that accept the same
"scalar | distribution | function" trio: attribute builders
([`transmission_traits`](@ref), [`clinical_presentation`](@ref)),
intervention parameters (vaccination eligibility, isolation delays),
and competing-risk fields ([`Risk`](@ref)).
"""
_sample_value(x::Real, rng, args...) = float(x)
_sample_value(d::Distribution, rng, args...) = float(rand(rng, d))
_sample_value(f, rng, args...) = float(f(rng, args...))
