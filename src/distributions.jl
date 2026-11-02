"""
    NegBin(R, k)

A negative binomial offspring distribution with mean `R` (the reproduction
number) and dispersion `k`, so the variance is `R + R²/k`. Smaller `k` means
more superspreading: `k` below 1 is strong superspreading, and as `k` grows
the distribution approaches a Poisson. Returns a `NegativeBinomial` from
Distributions.jl.

!!! warning
    `NegativeBinomial(r, p)` from Distributions.jl takes a number of
    successes and a success probability, not a mean and dispersion.
    `NegativeBinomial(R, k)` therefore gives a different distribution,
    without any error when `k` is at most 1. Use `NegBin(R, k)`.

# Examples
```julia
NegBin(2.5, 0.16)   # R = 2.5, k = 0.16 (k as estimated for SARS)
```
"""
function NegBin(R::Real, k::Real)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    p = k / (k + R)
    return NegativeBinomial(k, p)
end

"""
    incubation_linked_generation_time(; presymptomatic_fraction=0.3, omega=2.0)

A generation time linked to each case's own incubation period, as in
Hellewell et al. (2020), for use as the `generation_time` of a
[`BranchingProcess`](@ref). Incubation periods come from
[`clinical_presentation`](@ref).

- `presymptomatic_fraction`: the share of transmission that happens before
  the infector's symptom onset.
- `omega`: the spread, in days, of the generation time around the
  incubation period.

Each case's generation time is drawn from a skew-normal distribution centred
on its incubation period, with scale `omega` and the skew chosen so that
`presymptomatic_fraction` of generation times are shorter than the incubation
period. Negative values are excluded, which makes the realised presymptomatic
share approximate: close when the incubation period is long compared with
`omega`, less so for short ones. Cases without an incubation period (for
example asymptomatic cases) are centred on 5 days.

# Examples
```julia
model = BranchingProcess(
    NegBin(2.5, 0.16),
    incubation_linked_generation_time(presymptomatic_fraction=0.3)
)
```
"""
function incubation_linked_generation_time(;
        presymptomatic_fraction::Real = 0.3,
        omega::Real = 2.0
    )
    0.0 < presymptomatic_fraction < 1.0 || throw(
        ArgumentError(
            "presymptomatic_fraction must be in (0, 1), got $presymptomatic_fraction"
        )
    )
    omega > 0.0 || throw(ArgumentError("omega must be positive, got $omega"))

    # Compute skew-normal alpha from presymptomatic fraction
    # For SN(xi, omega, alpha): P(X < xi) = 0.5 - arctan(alpha)/π
    # => alpha = tan(π(0.5 - presymp_frac))
    alpha = tan(float(π) * (0.5 - float(presymptomatic_fraction)))
    om = float(omega)

    return function (individual)
        inc_period = incubation_period(individual)
        if isnan(inc_period) || inc_period <= 0.0
            @debug "Missing or non-positive incubation period (e.g. asymptomatic individual); using 5.0 days" maxlog = 1
            inc_period = 5.0
        end
        return _TruncatedSkewNormal(inc_period, om, alpha)
    end
end

"""Skew-normal truncated to [0, ∞) via rejection sampling (cdf not available).
`logpdf` subtracts the retained-mass constant `log P(inner ≥ 0)`, computed
lazily on first use so the simulation path (which only samples) never pays for
the numerical integral."""
struct _TruncatedSkewNormal{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T
    inner::SkewNormal{T}
    logZ::Base.RefValue{T}   # log P(inner ≥ 0); NaN until first computed
    function _TruncatedSkewNormal(ξ::Real, ω::Real, α::Real)
        T = float(promote_type(typeof(ξ), typeof(ω), typeof(α)))
        inner = SkewNormal(T(ξ), T(ω), T(α))
        return new{T}(T(ξ), T(ω), T(α), inner, Ref(T(NaN)))
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

# The retained-mass constant `log P(inner ≥ 0)`, integrated lazily and cached on
# first request — `rand` (the simulation path) never triggers it. SkewNormal has
# no cdf in this Distributions.jl version, hence the numerical integral.
function _trunc_logZ(d::_TruncatedSkewNormal{T}) where {T}
    isnan(d.logZ[]) || return d.logZ[]
    Z = first(quadgk(x -> pdf(d.inner, x), zero(T), T(Inf)))
    return d.logZ[] = T(log(Z))
end

# Normalised over [0, ∞): subtract the retained-mass constant so the density
# integrates to 1 (the bare inner density does not on the truncated support).
function Distributions.logpdf(d::_TruncatedSkewNormal, x::Real)
    return x < 0.0 ? oftype(float(x), -Inf) : logpdf(d.inner, x) - _trunc_logZ(d)
end

"""
    Conditional(dist)

A parameter with a distribution that depends on the case: wherever a delay or a
probability accepts a number, a `Distribution` or a function, it also
accepts `Conditional(dist)`, where `dist(args...)` returns the
`Distribution` for the case given the arguments a function parameter would
take after `rng` (typically just the case's `Individual`).

Unlike a plain `(rng, args...) -> value` function, the distribution behind a
`Conditional` can be read back through [`resolve`](@ref EpiBranch.resolve)
without sampling it. That lets a likelihood over timing data, such as a
serial interval, use the case's delay distribution directly instead of only
ever drawing from it.

# Examples
```julia
latent_period = Conditional(ind -> Gamma(5.95, 1 / 0.77))
```
"""
struct Conditional{F}
    dist::F
end

"""
    resolve(x, args...) -> Distribution

Turn a parameter given as a number, a `Distribution` or a [`Conditional`](@ref)
into the `Distribution` it stands for: `Dirac(x)` for a number, the
distribution unchanged, or `c.dist(args...)` for a `Conditional` `c`. A plain
`(rng, args...) -> value` function resolves to an adapter that samples by
calling the function; it has no density, so it only ever supports
drawing a value, not reading one back.

[`sample_value`](@ref) draws from the result, as every built-in
delay and probability does. A likelihood that needs the density itself,
rather than a draw from it, calls `resolve` directly.
"""
resolve(x::Real, args...) = Dirac(x)
resolve(d::Distribution, args...) = d
resolve(c::Conditional, args...) = c.dist(args...)
resolve(f, args...) = _FunctionSampler(f, args)

"""A plain `(rng, args...) -> value` function wrapped as a `Sampleable`, so
`resolve` can return it alongside the `Distribution`s the other parameter
forms resolve to. It has no density: only `rand` is defined, for simulation."""
struct _FunctionSampler{F, A <: Tuple} <: Sampleable{Univariate, Continuous}
    f::F
    args::A
end
Base.rand(rng::AbstractRNG, s::_FunctionSampler) = s.f(rng, s.args...)

"""
    sample_value(x, rng, args...) -> Float64

Turn a parameter given as a number, a distribution, a [`Conditional`](@ref)
or a function into a number: `rand(rng, resolve(x, args...))`, converted to
`Float64`. Used wherever a parameter accepts any of these, such as
[`transmission_traits`](@ref), [`clinical_presentation`](@ref), isolation
delays, vaccination parameters and [`Risk`](@ref) fields.

`public` in Julia's sense: an extension can read its own such parameter
through this function instead of writing the dispatch again, with the same
float conversion and function arguments as the built-ins.
"""
sample_value(x, rng, args...) = float(rand(rng, resolve(x, args...)))
