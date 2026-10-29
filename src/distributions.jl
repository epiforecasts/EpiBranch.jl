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
    HazardKernel(hazard; upper = Inf)

A contact-interval distribution built directly from an instantaneous hazard
rather than from a named family: `hazard(t)` is the rate of transmission at
time `t` since the window opened, for `t` from 0 up to `upper` (unbounded by
default). Use it as the `kernel` of an [`Infectiousness`](@ref) window, built
fresh for each case from a function of the individual, `kernel = ind -> ...`,
when transmission should follow something computed on the fly rather than a
named distribution: most usefully a per-case within-host trajectory, drawn
once at infection and stored on the individual, read through a link that
turns it into a rate.

`hazard` is called many times to integrate and invert the cumulative hazard
by quadrature, so it should be cheap and non-negative throughout `(0, upper)`.
One that falls to zero and stays there leaves a case uninfectious from then
on: draws pile up at `upper`, or run to arbitrarily late times when `upper`
is left unbounded, exactly as `Weibull`'s or `Gamma`'s own tails do.

# Examples

A constant hazard gives the same contact-interval law as an exponential with
the reciprocal rate:

```julia
using EpiBranch, Distributions
HazardKernel(t -> 0.5)   # the same law as Exponential(2.0)
```

A case's infectiousness follows a within-host pathogen-load trajectory,
drawn once per case (an ordinary attribute, see [`ModelSpec`](@ref)) and read
through a link turning load into a transmission rate:

```julia
# user-supplied within-host model: load rises to a peak, then clears
function load_curve(rng)
    peak_time = rand(rng, Gamma(3.0, 1.5))
    peak_height = rand(rng, LogNormal(0.0, 0.3))
    clearance = rand(rng, Gamma(4.0, 0.5))
    return t -> t < peak_time ? peak_height * t / peak_time :
        peak_height * exp(-clearance * (t - peak_time))
end

attributes = (rng, ind) -> ind.state[:trajectory] = load_curve(rng)
community = Infectiousness(
    NegBin(1.2, 0.5);
    kernel = ind -> HazardKernel(t -> 0.1 * ind.state[:trajectory](t))
)
model = ModelSpec(BranchingProcess(community); attributes)
```

Generation time, incubation period and serial interval are then whatever the
trajectory and its links to each clinical event imply, instead of separate
inputs drawn independently of one another.
"""
struct HazardKernel{F, T <: AbstractFloat}
    hazard::F
    upper::T
end
function HazardKernel(hazard; upper::Real = Inf)
    upper > 0 || throw(ArgumentError("upper must be positive, got $upper"))
    return HazardKernel(hazard, float(upper))
end

SurvivalDistributions.hazard(k::HazardKernel, t::Real) = k.hazard(t)
SurvivalDistributions.loghazard(k::HazardKernel, t::Real) = log(k.hazard(t))
function SurvivalDistributions.cumhazard(k::HazardKernel, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be non-negative, got $τ"))
    iszero(τ) && return zero(float(τ))
    return _hazard_integral(k.hazard, 0.0, min(float(τ), k.upper))
end

function Distributions.logccdf(k::HazardKernel, τ::Real)
    τ < 0 && return zero(float(τ))
    τ >= k.upper && return oftype(float(τ), -Inf)
    return -cumhazard(k, τ)
end

function Distributions.invlogccdf(k::HazardKernel, lp::Real)
    isinf(lp) && lp < 0 && return oftype(float(lp), k.upper)
    lp <= 0 || throw(ArgumentError("invlogccdf needs a non-positive log-survival, got $lp"))
    target = -lp
    iszero(target) && return 0.0
    return _hazard_invert(k, float(target))
end

# Define `cdf`/`logcdf` so a `HazardKernel` passed as a bare `contact_interval`
# (rather than built per case through `kernel`) works wherever `implied_R` and
# `transmission_timing`'s `offspring` form call `cdf`/`truncated` on it
# directly, the same as a named distribution.
Distributions.cdf(k::HazardKernel, τ::Real) = -expm1(logccdf(k, τ))
Distributions.logcdf(k::HazardKernel, τ::Real) = _log1mexp(logccdf(k, τ))

Base.rand(rng::AbstractRNG, k::HazardKernel) = invlogccdf(k, log(rand(rng)))
Distributions.quantile(k::HazardKernel, p::Real) = invlogccdf(k, log1p(-p))
Base.minimum(::HazardKernel) = 0.0
Base.maximum(k::HazardKernel) = k.upper
Distributions.partype(::HazardKernel{F, T}) where {F, T} = T

# `k.upper` caps the bracket `_invert_hazard` grows, so a `target` beyond the
# total hazard there is met by returning `upper` rather than doubling forever.
# Because `upper` is a cap here, not a certain event, it is not treated as the
# survival's own end the way a calendar-scaled profile's own bounded support
# does.
function _hazard_invert(k::HazardKernel, target::Real)
    t, t_acc = _invert_hazard(k.hazard, k.upper, false, true, target)
    return t < k.upper ? _implicit_hazard_step(k.hazard, t, t_acc, target) : t
end

"""
A [`HazardKernel`](@ref) conditioned on falling at or before `hi`: what
`transmission_timing`'s `offspring` form needs so every draw is realised
rather than thinned away by the case's own recovery. The lower bound is
always 0, since a `HazardKernel`'s own support already starts there.

This is `Distributions.truncated`'s usual meaning, a rescaled conditional
distribution, not `HazardKernel`'s own censoring at `upper` (which piles
draws up there instead of excluding them).
"""
struct _TruncatedHazardKernel{K, T <: AbstractFloat}
    kernel::K
    hi::T
    logF_hi::T
end
function Distributions.truncated(k::HazardKernel, lo::Real, hi::Real)
    iszero(lo) || throw(
        ArgumentError("truncated(::HazardKernel, lo, hi) needs lo = 0")
    )
    hi = float(hi)
    return _TruncatedHazardKernel(k, hi, logcdf(k, hi))
end

function Distributions.logcdf(k::_TruncatedHazardKernel, τ::Real)
    return τ >= k.hi ? zero(k.logF_hi) : logcdf(k.kernel, τ) - k.logF_hi
end
Distributions.cdf(k::_TruncatedHazardKernel, τ::Real) = exp(logcdf(k, τ))
Distributions.logccdf(k::_TruncatedHazardKernel, τ::Real) = _log1mexp(logcdf(k, τ))
function Distributions.quantile(k::_TruncatedHazardKernel, p::Real)
    return quantile(k.kernel, exp(k.logF_hi + log(p)))
end
Base.rand(rng::AbstractRNG, k::_TruncatedHazardKernel) = quantile(k, rand(rng))
Base.minimum(::_TruncatedHazardKernel) = 0.0
Base.maximum(k::_TruncatedHazardKernel) = k.hi

"""
    sample_value(x, rng, args...) -> Float64

Turn a parameter given as a number, a distribution or a function into a
number: the number itself, a draw from the distribution, or `f(rng, args...)`.
Used wherever a parameter accepts any of the three, such as
[`transmission_traits`](@ref), [`clinical_presentation`](@ref), isolation
delays, vaccination parameters and [`Risk`](@ref) fields.

`public` in Julia's sense: an extension can read its own such parameter
through this function instead of writing the three methods again, with the
same float conversion and function arguments as the built-ins.
"""
sample_value(x::Real, rng, args...) = float(x)
sample_value(d::Distribution, rng, args...) = float(rand(rng, d))
sample_value(f, rng, args...) = float(f(rng, args...))
