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
    incubation_linked_generation_time(; presymptomatic_fraction=0.3, relative_spread=0.5)

A generation time linked to each case's own incubation period, in the manner
of Hellewell et al. (2020), for use as the `generation_time` of a
[`BranchingProcess`](@ref). Incubation periods come from
[`clinical_presentation`](@ref).

- `presymptomatic_fraction`: the share of transmission that happens before
  the infector's symptom onset.
- `relative_spread`: the spread of the generation time around the incubation
  period, as a fraction of that incubation period.

Each case's generation time is drawn from a skew-normal distribution centred
on its incubation period, with scale `relative_spread` times the incubation
period, truncated at zero. The skew is chosen for the truncated distribution:
the share of generation times shorter than the incubation period equals
`presymptomatic_fraction` whatever the incubation period. The spread scales
with the incubation period, so the skew is the same for every case. An
asymptomatic case has no onset but still draws an incubation period, and its
generation time is centred on that. A case with no incubation period at all is
an error.

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
        relative_spread::Real = 0.5
    )
    0.0 < presymptomatic_fraction < 1.0 || throw(
        ArgumentError(
            "presymptomatic_fraction must be in (0, 1), got $presymptomatic_fraction"
        )
    )
    relative_spread > 0.0 || throw(
        ArgumentError("relative_spread must be positive, got $relative_spread")
    )

    rel = float(relative_spread)
    alpha = _truncated_skewness(float(presymptomatic_fraction), rel)

    return function (individual)
        inc_period = _latent_incubation_period(individual)
        (isnan(inc_period) || inc_period <= 0.0) && throw(
            ArgumentError(
                "incubation_linked_generation_time needs a positive incubation " *
                    "period for every case; provide attributes = clinical_presentation(...)"
            )
        )
        return _TruncatedSkewNormal(inc_period, rel * inc_period, alpha)
    end
end

# The skew `α` for which a skew-normal centred on `ξ` with scale `rel * ξ`,
# truncated at zero, puts `fraction` of its mass below `ξ`. Truncation removes
# mass from the lower side. The untruncated relation
# `P(X < ξ) = 0.5 - atan(α) / π` therefore understates the skew needed once
# `rel` is not small. The problem is scale-free and is solved once in units of
# `ξ`. Bisection on `atan(α)` finds the root because the truncated share
# falls from 1 to 0 as `α` runs from `-Inf` to `Inf`.
function _truncated_skewness(fraction::Float64, rel::Float64)
    centre = 1 / rel
    function share(θ)
        d = SkewNormal(centre, 1.0, tan(θ))
        below = first(quadgk(x -> pdf(d, x), 0.0, centre))
        total = below + first(quadgk(x -> pdf(d, x), centre, centre + 10.0, centre + 50.0))
        return below / total
    end
    lo, hi = -π / 2 + 1.0e-9, π / 2 - 1.0e-9
    for _ in 1:80
        mid = (lo + hi) / 2
        share(mid) > fraction ? (lo = mid) : (hi = mid)
    end
    return tan((lo + hi) / 2)
end

"""Skew-normal truncated to [0, ∞) via rejection sampling (cdf not available).
`logpdf` subtracts the retained-mass constant `log P(inner ≥ 0)`, computed
lazily on first use so the simulation path (which only samples) never pays for
the numerical integral."""
struct _TruncatedSkewNormal{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    inner::SkewNormal{T}
    logZ::Base.RefValue{T}   # log P(inner ≥ 0); NaN until first computed
    function _TruncatedSkewNormal(ξ::Real, ω::Real, α::Real)
        T = float(promote_type(typeof(ξ), typeof(ω), typeof(α)))
        return new{T}(SkewNormal(T(ξ), T(ω), T(α)), Ref(T(NaN)))
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

# Breakpoints bracketing the density's peak, which sits within a few `ω` of
# `ξ` whatever `α` is: without them, `quadgk`'s adaptive refinement can step
# clean over a peak much narrower than the integration range (small `ω`
# relative to `ξ`, or to `x - ξ`) and return a spuriously small mass. Outside
# the bracket the density is a smooth, featureless tail that plain adaptive
# quadrature tracks fine regardless of how wide. Shared by `_trunc_logZ`
# (integrating to `Inf`) and `logcdf` (integrating to `x`).
function _peak_breakpoints(d::_TruncatedSkewNormal, upper::Real)
    (; ξ, ω) = d.inner
    margin = 10 * float(ω)
    inner = (float(ξ) - margin, float(ξ) + margin)
    lo = zero(float(upper))
    return (lo, filter(p -> lo < p < upper, inner)..., upper)
end

# The retained-mass constant `log P(inner ≥ 0)`, integrated lazily and cached on
# first request — `rand` (the simulation path) never triggers it. SkewNormal has
# no cdf in this Distributions.jl version, hence the numerical integral.
function _trunc_logZ(d::_TruncatedSkewNormal{T}) where {T}
    isnan(d.logZ[]) || return d.logZ[]
    points = _peak_breakpoints(d, T(Inf))
    Z = first(quadgk(x -> pdf(d.inner, x), points...))
    return d.logZ[] = T(log(Z))
end

# Normalised over [0, ∞): subtract the retained-mass constant so the density
# integrates to 1 (the bare inner density does not on the truncated support).
function Distributions.logpdf(d::_TruncatedSkewNormal, x::Real)
    return x < 0.0 ? oftype(float(x), -Inf) : logpdf(d.inner, x) - _trunc_logZ(d)
end

# The mass below `x`, normalised the same way as `logpdf`: the integral of
# the inner density up to `x`, over the retained mass `_trunc_logZ` already
# caches. Needed because SkewNormal has no `cdf` in this version of
# Distributions.jl; without it, neither the engine's own `cumhazard`
# (derived from `cdf` by SurvivalDistributions) nor `truncated` (which
# Distributions builds from `logcdf`) can reach it.
function Distributions.logcdf(d::_TruncatedSkewNormal, x::Real)
    x <= 0.0 && return oftype(float(x), -Inf)
    fx = float(x)
    points = _peak_breakpoints(d, fx)
    mass = first(quadgk(t -> pdf(d.inner, t), points...))
    # Clamped at 0: `mass` and `_trunc_logZ` come from separate quadrature
    # calls, so their ratio can round fractionally past 1 once `x` reaches
    # into the far tail, where the true cdf is 1 already.
    return min(log(mass) - _trunc_logZ(d), zero(float(x)))
end
Distributions.cdf(d::_TruncatedSkewNormal, x::Real) = exp(logcdf(d, x))
Distributions.logccdf(d::_TruncatedSkewNormal, x::Real) = _log1mexp(logcdf(d, x))

Base.minimum(::_TruncatedSkewNormal) = 0.0
Base.maximum(::_TruncatedSkewNormal) = Inf
Distributions.partype(::_TruncatedSkewNormal{T}) where {T} = T

# Quantile by Newton's method on `cdf`, which has no closed form here
# either, bracketed by bisection as a safeguard. `truncated` needs it to
# draw the offspring form's generation time, and the race's `invlogccdf`
# reaches it through Distributions' own fallback `quantile(d, -expm1(lp))`,
# both on a path run once per case or pair: bisecting `cdf` itself down to
# full `Float64` precision took around fifty of its quadrature calls per
# quantile, so this uses the closed-form `pdf` for the Newton step, which
# converges in a handful.
function Distributions.quantile(d::_TruncatedSkewNormal, p::Real)
    0.0 <= p <= 1.0 || throw(DomainError(p, "quantile needs p in [0, 1]"))
    iszero(p) && return 0.0
    isone(p) && return Inf
    lo, hi = 0.0, max(abs(d.inner.ξ), d.inner.ω, one(d.inner.ξ))
    while cdf(d, hi) < p
        hi *= 2
    end
    x = (lo + hi) / 2
    for _ in 1:100
        fx = cdf(d, x) - p
        abs(fx) < 1.0e-12 && return x
        fx < 0 ? (lo = x) : (hi = x)
        step = x - fx / pdf(d, x)
        x = lo < step < hi ? step : lo + (hi - lo) / 2
    end
    return x
end

"""
    HazardKernel(hazard; upper = Inf, onset = 0.0)

A contact-interval distribution defined by the instantaneous hazard
`hazard(t)`, the rate of transmission at
time `t` since the window opened, for `t` from `onset` up to `upper`
(unbounded by default). Use it as the `kernel` of an [`Infectiousness`](@ref)
window, built for each case from a function of the individual,
`kernel = ind -> ...`. For example, transmission can depend on a
within-host trajectory drawn once at infection and stored on the
individual, through a link that converts it to a transmission rate.

The hazard is zero before `onset`, which defaults to 0, matching the
start of the hazard from a named distribution. A trajectory with a later
start should specify its onset explicitly instead of encoding a flat run-up
in `hazard`. The kernel treats `hazard` as
zero before `onset` regardless of what it returns there, and rejects an
`onset` still followed by a flat stretch of zero, because nothing would then
distinguish a trajectory still to switch on from one that has permanently
cleared.

Integrating and inverting the cumulative hazard by quadrature requires many
calls to `hazard`. It should be inexpensive to evaluate and non-negative throughout
`(onset, upper)`. `upper` is where the integration stops: the hazard counts
as zero beyond it, and a draw the total hazard up to there cannot reach is
`Inf`, a contact that never happens. Give a hazard that clears an `upper`
past which it is negligible. Such a draw is then settled at `upper`, where
with `upper` left unbounded the search for it has to double out to the
largest float first, about a thousand quadratures of a zero hazard for every
contact that never happens. A hazard that falls to zero and stays there
leaves a case uninfectious from then on either way.

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
# The load has cleared long before 100 days, so the integration stops there.
community = Infectiousness(
    NegBin(1.2, 0.5);
    kernel = ind -> HazardKernel(t -> 0.1 * ind.state[:trajectory](t); upper = 100.0)
)
model = ModelSpec(BranchingProcess(community); attributes)
```

The trajectory and its links to each clinical event determine the generation
time, incubation period and serial interval. These quantities are no longer
separate, independent draws.
"""
struct HazardKernel{F, T <: Real} <: ContinuousUnivariateDistribution
    hazard::F
    onset::T
    upper::T
end
function HazardKernel(hazard; upper::Real = Inf, onset::Real = 0.0)
    upper > 0 || throw(ArgumentError("upper must be positive, got $upper"))
    onset >= 0 || throw(ArgumentError("onset must be non-negative, got $onset"))
    onset < upper || throw(
        ArgumentError("onset must be below upper, got onset = $onset, upper = $upper")
    )
    T = float(promote_type(typeof(onset), typeof(upper)))
    onset, upper = T(onset), T(upper)
    # A point evaluation at `onset` cannot tell a hazard that switches on
    # smoothly from zero there (as a named distribution's own hazard often
    # does) from one still flat for a stretch beyond it, so this integrates
    # the same first step the inverter's own bracket would grow.
    window = _hazard_integral_from(hazard, onset, min(onset + one(T), upper))
    window > 0 || throw(
        ArgumentError(
            "hazard is still zero just after onset = $onset; " *
                "give the time it actually switches on"
        )
    )
    return HazardKernel(hazard, onset, upper)
end

# Zero outside `(onset, upper)` whatever `hazard` returns there, as
# `cumhazard` already treats it: a calendar-scaled window integrates these
# directly, and the pairwise likelihood reads them at observed times.
function SurvivalDistributions.hazard(k::HazardKernel, t::Real)
    return k.onset < t < k.upper ? k.hazard(t) : zero(float(t))
end
function SurvivalDistributions.loghazard(k::HazardKernel, t::Real)
    return k.onset < t < k.upper ? log(k.hazard(t)) : oftype(float(t), -Inf)
end
function SurvivalDistributions.cumhazard(k::HazardKernel, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be non-negative, got $τ"))
    τ <= k.onset && return zero(float(τ))
    hi = min(float(τ), k.upper)
    isfinite(hi) && return _hazard_integral_from(k.hazard, k.onset, hi)
    # Over an unbounded horizon quadgk hands back a number even where the
    # integral diverges, with an error estimate of the same order as the
    # value; the total is infinite then, which the race's guard against
    # redrawing from a window that never closes needs to see.
    val, err = quadgk(s -> k.hazard(k.onset + s), 0.0, Inf)
    return err <= sqrt(eps(one(val))) * max(val, one(val)) ? val : oftype(val, Inf)
end

function Distributions.logccdf(k::HazardKernel, τ::Real)
    τ < 0 && return zero(float(τ))
    return -cumhazard(k, τ)
end

function Distributions.invlogccdf(k::HazardKernel, lp::Real)
    isinf(lp) && lp < 0 && return oftype(float(lp), Inf)
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
# The density is the hazard times the survival, which a clinical transition's
# likelihood reads where the kernel times a delay.
Distributions.logpdf(k::HazardKernel, x::Real) = loghazard(k, x) + logccdf(k, x)
Distributions.pdf(k::HazardKernel, x::Real) = exp(logpdf(k, x))

Base.rand(rng::AbstractRNG, k::HazardKernel) = invlogccdf(k, log(rand(rng)))
Distributions.quantile(k::HazardKernel, p::Real) = invlogccdf(k, log1p(-p))
Base.minimum(::HazardKernel) = 0.0
# A draw beyond the total hazard is `Inf`, so the support is unbounded even
# where `upper` cuts the integration short.
Base.maximum(::HazardKernel) = Inf
Distributions.partype(::HazardKernel{F, T}) where {F, T} = T

# `k.upper` caps the bracket `_invert_hazard` grows. A `target` beyond the
# total hazard up to there is a contact that never happens, returned as `Inf`
# rather than doubled towards forever; `upper` only bounds the integration,
# so it is not treated as the survival's own end the way a calendar-scaled
# profile's bounded support is. The bracket starts at `k.onset`, where the
# hazard is already known to be positive (checked once, at construction), so
# it grows through territory that integrates cleanly rather than risking the
# stall a flat run-up before it would cause.
function _hazard_invert(k::HazardKernel, target::Real)
    t, t_acc = _invert_hazard(k.hazard, k.onset, k.upper, false, target)
    t_acc < target && return oftype(t, Inf)
    return t < k.upper ? _implicit_hazard_step(k.hazard, t, t_acc, target) : t
end

"""
A [`HazardKernel`](@ref) conditioned on falling at or before `hi`: what
`transmission_timing`'s `offspring` form needs so every draw is realised
rather than thinned away by the case's own recovery. The lower bound is
always 0, because a `HazardKernel`'s own support already starts there.

This is the usual meaning of `Distributions.truncated`, a rescaled
conditional distribution; the kernel's own `upper` only bounds its
integration and excludes nothing.

An internal wrapper rather than Distributions' own `Truncated`, which draws
through the base `quantile` and cannot say that a window closing before the
kernel's onset has no contact to draw; a `Distribution` subtype all the same,
so a user who caps a kernel with `truncated` reaches every seam a named
distribution does.
"""
struct _TruncatedHazardKernel{K, T <: Real} <: ContinuousUnivariateDistribution
    kernel::K
    hi::T
    logF_hi::T
end
function Distributions.truncated(k::HazardKernel, lo::Real, hi::Real)
    iszero(lo) || throw(
        ArgumentError("truncated(::HazardKernel, lo, hi) needs lo = 0")
    )
    hi, logF_hi = promote(float(hi), logcdf(k, float(hi)))
    return _TruncatedHazardKernel(k, hi, logF_hi)
end
# The keyword spellings, `truncated(k; upper = hi)` and `truncated(k; lower =
# 0)`, reach Distributions' own `Truncated` otherwise, which knows nothing of
# a window with no mass.
Distributions.truncated(k::HazardKernel, ::Nothing, hi::Real) = truncated(k, 0.0, hi)
function Distributions.truncated(k::HazardKernel, lo::Real, ::Nothing)
    iszero(lo) || throw(
        ArgumentError("truncated(::HazardKernel, lo, hi) needs lo = 0")
    )
    return k
end

# A window that closes at or before the kernel's onset has no mass to
# condition on (`logF_hi == -Inf`): every draw is then `Inf`, a contact that
# never happens, the counterpart of a named distribution's degenerate
# truncation, which the window's `until` censors.
_no_mass(k::_TruncatedHazardKernel) = isinf(k.logF_hi)

function Distributions.logcdf(k::_TruncatedHazardKernel, τ::Real)
    _no_mass(k) && return oftype(k.logF_hi, -Inf)
    return τ >= k.hi ? zero(k.logF_hi) : logcdf(k.kernel, τ) - k.logF_hi
end
Distributions.cdf(k::_TruncatedHazardKernel, τ::Real) = exp(logcdf(k, τ))
Distributions.logccdf(k::_TruncatedHazardKernel, τ::Real) = _log1mexp(logcdf(k, τ))
function Distributions.quantile(k::_TruncatedHazardKernel, p::Real)
    _no_mass(k) && return oftype(k.hi, Inf)
    return quantile(k.kernel, exp(k.logF_hi + log(p)))
end
function Distributions.logpdf(k::_TruncatedHazardKernel, τ::Real)
    (_no_mass(k) || τ >= k.hi) && return oftype(k.logF_hi, -Inf)
    return logpdf(k.kernel, τ) - k.logF_hi
end
Distributions.pdf(k::_TruncatedHazardKernel, τ::Real) = exp(logpdf(k, τ))
# The hazard of the conditional law, which a calendar-scaled window reads
# directly, differs from the base kernel's: it blows up towards `hi`.
function SurvivalDistributions.loghazard(k::_TruncatedHazardKernel, τ::Real)
    lp = logpdf(k, τ)
    # Past the base kernel's `upper` but below the cap both density and
    # survival are gone; the hazard is zero there, not their ratio.
    isfinite(lp) || return lp
    return lp - logccdf(k, τ)
end
SurvivalDistributions.hazard(k::_TruncatedHazardKernel, τ::Real) = exp(loghazard(k, τ))
SurvivalDistributions.cumhazard(k::_TruncatedHazardKernel, τ::Real) = -logccdf(k, τ)
Base.rand(rng::AbstractRNG, k::_TruncatedHazardKernel) = quantile(k, rand(rng))
Base.minimum(::_TruncatedHazardKernel) = 0.0
Base.maximum(k::_TruncatedHazardKernel) = k.hi
Distributions.partype(k::_TruncatedHazardKernel) = Distributions.partype(k.kernel)

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
drawing from it.

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
calling the function; it has no density, so it only supports
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
