"""
    RenewalOffspring(R, generation_time; k = Inf, R_max = nothing)

An offspring function for a time-varying instantaneous reproduction number
`R(t)`, to give to an [`Infectiousness`](@ref) window. A case infected at
calendar time `s` infects people at rate `ν R(t) g(t - s)` at time `t`, as in
the renewal equation, where `g` is the generation-time density and `ν` is the
case's own infectiousness. It returns an [`OffspringTimes`](@ref), so the
window's `kernel` is not used and the model has a single type.

`Infectiousness`'s `calendar` scales the hazard of the contact kernel, which
changes when contacts happen but not how many there are, so it cannot express
`R(t)`.

- `R`: the reproduction number as a function of calendar time `t`, in days
  since the start of the outbreak, or a constant. The case's infection time is
  the calendar time its window opens at, so use the default `from = :infection`.
- `generation_time`: either a continuous `Distribution` of the time from
  infection to each infection caused, or a vector of weights over daily lags
  `1, 2, …`, which is normalised to sum to one. A distribution gives
  continuous infection times: contacts are drawn at the rate `R_max` times the
  density and each is kept with probability `R(t) / R_max`. A vector places
  infections on whole-day lags, each drawn as a Poisson count with mean
  `ν R(s + lag) g(lag)`.
- `k`: the dispersion of individual infectiousness, `ν ~ Gamma(k, 1/k)` with
  mean one. With a constant `R` the number of people a case infects is then
  `NegBin(R, k)`. The default `Inf` gives `ν = 1`, so the number is
  `Poisson(R)`. One case's infections on different days share `ν`, so they
  are correlated.
- `R_max`: an upper bound on `R(t)`, required with a continuous
  `generation_time` and unused with a vector. A draw of `R(t)` above it throws
  an `ArgumentError`. The bound need not be tight, but the cost grows with it.

With a vector `generation_time` and `R(t)` read at the infection time plus
the lag, the model is the renewal equation with infections counted at the end
of each day's lag.

# Examples
```julia
using EpiBranch, Distributions

R(t) = t < 60 ? 1.3 : 0.9
offspring = RenewalOffspring(R, Gamma(5.0, 1.1); k = 1.0, R_max = 1.3)
model = BranchingProcess(Infectiousness(offspring))
state = simulate(model; n_initial = 100, max_cases = nothing, max_time = 120.0)
```
"""
struct RenewalOffspring{RT, G, K <: Real, B}
    R::RT
    generation_time::G
    k::K
    R_max::B
end

function RenewalOffspring(R, generation_time::Distribution; k::Real = Inf, R_max = nothing)
    Distributions.minimum(generation_time) >= 0 ||
        throw(ArgumentError("`generation_time` must have no support below zero"))
    R_max === nothing &&
        throw(ArgumentError("a continuous `generation_time` needs `R_max`, an upper bound on `R(t)`"))
    R_max > 0 || throw(ArgumentError("`R_max` must be positive"))
    _check_renewal_dispersion(k)
    return RenewalOffspring(_as_function(R), generation_time, k, R_max)
end

function RenewalOffspring(R, generation_time::AbstractVector{<:Real}; k::Real = Inf, R_max = nothing)
    isempty(generation_time) && throw(ArgumentError("`generation_time` has no lags"))
    all(>=(0), generation_time) && sum(generation_time) > 0 ||
        throw(ArgumentError("`generation_time` weights must be non-negative and not all zero"))
    _check_renewal_dispersion(k)
    g = Float64.(generation_time) ./ sum(generation_time)
    return RenewalOffspring(_as_function(R), g, k, nothing)
end

_as_function(R::Real) = _ConstantR(Float64(R))
_as_function(R) = R

struct _ConstantR
    value::Float64
end
(c::_ConstantR)(t) = c.value

function _check_renewal_dispersion(k)
    k > 0 || throw(ArgumentError("`k` must be positive (use `Inf` for no individual variation)"))
    return nothing
end

# Individual infectiousness, mean one.
_infectiousness(rng, k) = isfinite(k) ? rand(rng, Gamma(k, 1 / k)) : 1.0

function _renewal_R(o::RenewalOffspring, t)
    r = o.R(t)
    r >= 0 || throw(ArgumentError("`R(t)` is negative at t = $t"))
    return r
end

function (o::RenewalOffspring{RT, <:Distribution})(rng::AbstractRNG, ind) where {RT}
    t = ind.infection_time
    n = rand(rng, Poisson(_infectiousness(rng, o.k) * o.R_max))
    times = Float64[]
    for _ in 1:n
        d = rand(rng, o.generation_time)
        r = _renewal_R(o, t + d)
        r <= o.R_max ||
            throw(ArgumentError("`R(t)` = $r at t = $(t + d) exceeds `R_max` = $(o.R_max)"))
        rand(rng) * o.R_max < r && push!(times, d)
    end
    return OffspringTimes(times)
end

function (o::RenewalOffspring{RT, <:AbstractVector})(rng::AbstractRNG, ind) where {RT}
    t = ind.infection_time
    ν = _infectiousness(rng, o.k)
    times = Float64[]
    for (lag, w) in enumerate(o.generation_time)
        w > 0 || continue
        for _ in 1:rand(rng, Poisson(ν * _renewal_R(o, t + lag) * w))
            push!(times, lag)
        end
    end
    return OffspringTimes(times)
end
