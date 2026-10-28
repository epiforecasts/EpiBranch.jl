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
struct HazardKernel{F, T <: AbstractFloat} <: ContinuousUnivariateDistribution
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
    return first(quadgk(k.hazard, 0.0, min(float(τ), k.upper)))
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
# (rather than built per case through `kernel`) meets the same `Distribution`
# contract as a named distribution: `truncated` calls `cdf` on construction,
# and `implied_R` calls it directly.
Distributions.cdf(k::HazardKernel, τ::Real) = -expm1(logccdf(k, τ))
Distributions.logcdf(k::HazardKernel, τ::Real) = _log1mexp(logccdf(k, τ))

Base.rand(rng::AbstractRNG, k::HazardKernel) = invlogccdf(k, log(rand(rng)))

# `Distributions.rand(::Truncated)` dispatches to `quantile`, not `invlogccdf`,
# whenever the truncation probability isn't tiny; without this, truncating a
# `HazardKernel` (as `_conditioned_on_recovery` does) throws on sampling.
Distributions.quantile(k::HazardKernel, p::Real) = invlogccdf(k, log1p(-p))
Base.minimum(::HazardKernel) = 0.0
Base.maximum(k::HazardKernel) = k.upper
Distributions.partype(::HazardKernel{F, T}) where {F, T} = T

# Grow a bracket from 0 until the integrated hazard reaches `target`, doubling
# each step; `k.upper` caps the bracket, so a `target` beyond the total hazard
# there is met by returning `upper` rather than doubling forever. With `upper`
# left unbounded, a window adding nothing to `hi_acc` is ambiguous on its own:
# it could mean the hazard has converged to a finite total below `target` (a
# case that clears and never becomes infectious again), or just that the
# window hasn't yet reached a later onset (a hazard that switches on after a
# flat zero run-up). Telling them apart needs the tail: quadgk run to `Inf`
# returns a value together with an error estimate, and that estimate is small
# only when the tail converges; a hazard still to switch on makes it diverge
# instead, because the quadrature then spans both the flat run-up and
# whatever comes after. A stalled window returns `upper` only once that tail
# check confirms there is no more hazard left to integrate.
function _hazard_invert(k::HazardKernel, target::Real)
    upper = k.upper
    lo = zero(target)
    acc = zero(target)
    width = one(target)
    while true
        hi = min(lo + width, upper)
        hi_acc = acc + first(quadgk(k.hazard, lo, hi))
        (hi_acc >= target || hi == upper) &&
            return _bisect_hazard(k, lo, hi, acc, hi_acc, target)
        if isinf(upper) && isapprox(hi_acc, acc; rtol = 1.0e-9, atol = 1.0e-12)
            tail, tail_err = quadgk(k.hazard, hi, Inf)
            tail_err < 1.0e-6 * max(hi_acc + tail, 1.0) && hi_acc + tail < target &&
                return upper
        end
        lo, acc = hi, hi_acc
        width *= 2
    end
    return
end

# Bisect `[lo, hi]` down to adjacent floats; `acc` and `hi_acc`, the
# cumulative hazards from 0 at its two ends, bracket `target`. `hi == k.upper`
# short of `target` never reaches it, so the loop returns `hi` there instead
# of bisecting further.
function _bisect_hazard(k::HazardKernel, lo::Real, hi::Real, acc::Real, hi_acc::Real, target::Real)
    hi_acc >= target || return hi
    while true
        mid = lo + (hi - lo) / 2
        (lo < mid < hi) || return hi
        mid_acc = acc + first(quadgk(k.hazard, lo, mid))
        if mid_acc >= target
            hi = mid
        else
            lo, acc = mid, mid_acc
        end
    end
    return
end

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
