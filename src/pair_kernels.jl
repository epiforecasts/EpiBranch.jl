"""
    PairContext(infector, susceptible, infector_infection_time)

What a [`PairKernel`](@ref) function knows about one infector and the person
they may infect, in simulation and in the likelihood alike: the two people's
numbers in the population (`infector`, `susceptible`) and the infector's
infection time (days since the start of the outbreak).

The susceptible person's own infection time is left out because it is not yet
known when their contact interval is drawn. To use fixed characteristics such
as age or household, look them up by person number in a table the function
refers to.
"""
struct PairContext{T <: Real}
    infector::Int
    susceptible::Int
    infector_infection_time::T
end

"""
    Steps(breaks, values)

A change in transmission on given calendar days, such as a lockdown or school
closure that scales everyone's rate of infecting others. `breaks` are the days
(since the start of the outbreak) on which the multiplier changes and `values`
the multipliers: `values[1]` before `breaks[1]`, `values[k+1]` from `breaks[k]`
(inclusive) up to `breaks[k+1]`, and `values[end]` from `breaks[end]` onwards.
`breaks` must be finite and strictly increasing, and every value non-negative;
a value of zero stops transmission until the next break, or for good if it
is the last value.

Pass it as the `calendar` of a [`PairKernel`](@ref), which multiplies the
contact-interval hazard by the value in force on each calendar day.

# Example

Transmission halves from day 30:

```julia
using EpiBranch, Distributions
lockdown = Steps([30.0], [1.0, 0.5])
kernel = PairKernel(ctx -> Exponential(2.0); calendar = lockdown)
```
"""
struct Steps{T <: Real}
    breaks::Vector{T}
    values::Vector{T}
    function Steps{T}(breaks, values) where {T <: Real}
        length(values) == length(breaks) + 1 || throw(
            ArgumentError("Steps needs one more value than breaks")
        )
        issorted(breaks; lt = <=) || throw(
            ArgumentError("Steps breaks must be strictly increasing")
        )
        all(isfinite, breaks) || throw(ArgumentError("Steps breaks must be finite"))
        all(v -> v >= 0, values) || throw(
            ArgumentError("Steps values must be non-negative")
        )
        return new{T}(Vector{T}(breaks), Vector{T}(values))
    end
end
function Steps(breaks, values)
    T = promote_type(eltype(breaks), eltype(values), Float64)
    return Steps{T}(breaks, values)
end

# The multiplier in force at calendar time `t`.
(s::Steps)(t::Real) = calendar_multiplier(s, t)

"""
    calendar_multiplier(schedule, t)

The multiplier a calendar `schedule` applies to transmission on day `t`
(non-negative; 1 leaves it unchanged). [`PairKernel`](@ref) multiplies its
contact-interval hazard by it.

Every schedule defines this. A new schedule is a type with this method and,
depending on its [`calendar_shape`](@ref EpiBranch.calendar_shape), either
[`next_calendar_break`](@ref EpiBranch.next_calendar_break) (piecewise
constant, the default) or a `calendar_shape` method declaring it smooth.
[`Steps`](@ref) is the piecewise-constant schedule the package provides.
"""
calendar_multiplier(s::Steps, t::Real) = s.values[searchsortedlast(s.breaks, t) + 1]

"""
    next_calendar_break(schedule, t)

The first day strictly after `t` on which a piecewise-constant `schedule`
changes value, or `Inf` if it stays constant from `t` onwards.

Simulation and the likelihood use it to find where each constant stretch of the
schedule ends. A smooth schedule has no breaks and does not define it.
"""
function next_calendar_break(s::Steps, t::Real)
    idx = searchsortedlast(s.breaks, t) + 1
    return idx <= length(s.breaks) ? s.breaks[idx] : oftype(float(t), Inf)
end

calendar_bound(s::Steps) = maximum(s.values)

"""
    PiecewiseConstantCalendar()

Marks a calendar schedule as constant between the days
[`next_calendar_break`](@ref EpiBranch.next_calendar_break) reports, like
[`Steps`](@ref). This is the default [`calendar_shape`](@ref
EpiBranch.calendar_shape), and its effect on transmission is computed exactly.
"""
struct PiecewiseConstantCalendar end

"""
    SmoothCalendar()

Marks a calendar schedule whose multiplier changes continuously, such as a
seasonal curve, as its [`calendar_shape`](@ref EpiBranch.calendar_shape). Its
effect on transmission is computed numerically, so the multiplier should be
smooth. It is assumed never to stop transmission for good.
"""
struct SmoothCalendar end

"""
    calendar_shape(schedule)

Whether a calendar `schedule` changes in steps or smoothly:
[`PiecewiseConstantCalendar`](@ref EpiBranch.PiecewiseConstantCalendar), the
default, which needs [`next_calendar_break`](@ref
EpiBranch.next_calendar_break), or [`SmoothCalendar`](@ref
EpiBranch.SmoothCalendar), which a smooth schedule declares for its own type.
A seasonal multiplier, for example:

```julia
struct DayOfWeekRamp{T <: Real}
    slope::T
end
EpiBranch.calendar_multiplier(s::DayOfWeekRamp, t) = 1 + s.slope * mod(t, 7)
EpiBranch.calendar_shape(::DayOfWeekRamp) = EpiBranch.SmoothCalendar()
```

[`Seasonal`](@ref) is the smooth schedule the package provides, and a plain
callable `t -> multiplier` is read as a smooth schedule too, with no
`calendar_shape` method of its own needed.
"""
calendar_shape(schedule) = PiecewiseConstantCalendar()

"""
    calendar_bound(schedule)

The largest value a calendar `schedule`'s multiplier takes, or `nothing` if
none is known. [`Steps`](@ref) declares its largest step and
[`Seasonal`](@ref) its `1 + amplitude`; a schedule written outside the
package may add a method on its own type to declare a bound too.

The mass-action route ([`HomogeneousProcess`](@ref), [`MixingProcess`](@ref))
uses this to decide how to sample under a `calendar`: with a bound, it thins
at it, keeping the per-contact cost a fixed rate allows; with none, a
schedule's value cannot be bounded in advance, so every pair is sampled
directly, at a cost per pair instead.
"""
calendar_bound(schedule) = nothing

"""
    Seasonal(; amplitude, peak_day, period = 365.2425)

A smoothly oscillating multiplier on calendar time: `1 + amplitude *
cos(2π * (t - peak_day) / period)`. `amplitude` must be between 0 and 1, which
keeps the multiplier non-negative; `peak_day` is the calendar time of the
highest multiplier, and `period` defaults to a year.

Used as the `calendar` a [`PairKernel`](@ref) multiplies its contact-interval
hazard by, on the calendar-time axis. It declares
[`EpiBranch.SmoothCalendar`](@ref), so its cumulative hazard is integrated by
quadrature rather than exactly as [`Steps`](@ref)'s is. `amplitude` and
`peak_day` share a type parameter so the likelihood can be differentiated with
respect to either.
"""
struct Seasonal{T <: Real}
    amplitude::T
    peak_day::T
    period::Float64
    function Seasonal{T}(amplitude, peak_day, period) where {T <: Real}
        0 <= amplitude <= 1 || throw(
            ArgumentError("Seasonal amplitude must be between 0 and 1")
        )
        return new{T}(amplitude, peak_day, period)
    end
end
function Seasonal(; amplitude, peak_day, period = 365.2425)
    T = promote_type(typeof(amplitude), typeof(peak_day), Float64)
    return Seasonal{T}(T(amplitude), T(peak_day), Float64(period))
end

# The multiplier in force at calendar time `t`.
(s::Seasonal)(t::Real) = calendar_multiplier(s, t)

calendar_multiplier(s::Seasonal, t::Real) = 1 + s.amplitude * cos(2π * (t - s.peak_day) / s.period)
calendar_shape(::Seasonal) = SmoothCalendar()
calendar_bound(s::Seasonal) = 1 + s.amplitude

# A plain function read as a calendar schedule: `calendar = t -> multiplier`
# for forcing of any shape, with no breaks to declare.
calendar_multiplier(f::Function, t::Real) = f(t)
calendar_shape(::Function) = SmoothCalendar()

"""
    PairKernel(callback; state = nothing, calendar = nothing, watches = nothing)

A contact interval that depends on who the two people are or when the contact
happens: for example, faster transmission from older cases, or a lockdown that
halves transmission from day 30. `callback` returns the contact-interval
distribution (in days, measured from the start of the infector's infectious
period), and `calendar` optionally scales the rate of transmission by calendar
day. It works with `NetworkProcess`, `HouseholdProcess` and the
[`InfectionLayer`](@ref) forms of [`pairwise_surv_loglik`](@ref), so the same
kernel can be simulated and fitted.

With `state = nothing`, `callback(context::PairContext)` receives a
[`PairContext`](@ref) (the two people's numbers and the infector's infection
time) and returns a distribution, or a `(profile = ..., calendar = ...)` named
tuple. Use this when the kernel reads only fixed characteristics and the
infector's infection time:

```julia
kernel = PairKernel(context -> Exponential(exp(0.1 * context.infector_infection_time)))
```

With `state` given, `callback(context, source, target)` also receives a record
for each of the two people. In simulation, `state` is a function of the
individual that builds this record. For the likelihood, pass a vector of
records indexed by person number as `state`, or extract them from a simulation
with [`record_kernel`](@ref); the callback is the same in both.

`watches` lists every key of `individual.state` that the `state` function
reads, as a tuple of `Symbol`s (see [`EpiBranch.watched_records`](@ref)).
When one of these values changes for a person during a simulation, for example
when they are vaccinated, the contacts not yet made that depend on them are
redrawn under the new rate. A change to a key the function reads but
`watches` leaves out stops the simulation with an error. List every key the
function reads, even if nothing in the current model sets it. `watches` is required when `state` is a function; pass `()`
when the function reads no state key (for example one that looks people up by
`ind.id`). Anything that can change during the outbreak must be read from
`individual.state`, which is the only place changes are followed, and not from
a field such as `ind.susceptibility`. With no `state`, or with a vector of
records, nothing can change, and passing a non-empty `watches` is an error.

`calendar`, a [`Steps`](@ref) or [`Seasonal`](@ref) schedule, a callable
`t -> multiplier`, or any other type with a
[`calendar_multiplier`](@ref EpiBranch.calendar_multiplier) method, multiplies
the transmission rate by the schedule's value on the calendar day (the start of
the infector's infectious period plus the time since). A pair whose schedule
depends on a person's record returns it from the callback instead, as
`(profile = ..., calendar = ...)`, which overrides the shared `calendar` for
that pair.

```julia
PairKernel((ctx, source, target) -> Gamma(2.0, 1.5);
           calendar = Steps([30.0], [1.0, 0.25]),   # shared policy: rate falls to a quarter on day 30
           state = ind -> (age = ind.state[:age],),  # optional per-person record
           watches = (:age,))

PairKernel((ctx, source, target) ->
               isfinite(target.date) ?
               (profile = Exponential(2.5), calendar = Steps([target.date], [1.0, 0.25])) :
               Exponential(2.5);
           state = ind -> (date = get(ind.state, :policy_time, Inf)::Float64,),
           watches = (:policy_time,))
```

The rate at which one person infects another is the contact-interval
distribution's hazard at the time since the infector became infectious, times
the schedule's value on that calendar day. For a step schedule such as `Steps`
both simulation and the likelihood compute this exactly.

!!! warning "Records must keep their history"
    The callback must describe a hazard that only changes forwards in time:
    recording an event at day `t` must not change the hazard before `t`. A
    final "vaccinated" flag is not enough; keep the vaccination date, so the
    rate before it is unchanged. Neither the callback nor the `state` function
    may change anything, and the kernel may read a person's information only
    through `state`, since those records are all the likelihood is given.

A simulation in which no watched record changes draws outbreaks from the same
distribution as an ordinary kernel.

A plain distribution remains the simplest kernel and needs no `PairKernel`.
"""
struct PairKernel{F, S, C, W <: Tuple}
    callback::F
    state::S
    calendar::C
    watches::W
end
function PairKernel(callback; state = nothing, calendar = nothing, watches = nothing)
    return PairKernel(callback, state, calendar, _kernel_watches(state, watches))
end

# A kernel with no `state` reads no host state, and a vector of records is
# fixed while a likelihood reads it, so neither has anything that can move and
# neither has anything to declare. Refused rather than ignored: with no `state`
# a declaration would otherwise send a route live over nothing, and on records
# it would say something the kernel cannot honour. A projection has to say what
# it reads:
# there is no safe default, since inferring "nothing moves" would draw a run's
# contacts from stale hazards and inferring "everything moves" would compare
# the whole population at every case.
_kernel_watches(::Nothing, watches) = _no_records_to_watch(watches, "no `state`")
function _kernel_watches(::AbstractVector, watches)
    return _no_records_to_watch(watches, "a vector of records as its `state`")
end
function _no_records_to_watch(watches, what)
    (watches === nothing || isempty(_watch_keys(watches))) || throw(
        ArgumentError(
            "a `PairKernel` with $what reads no `individual.state`, so it has " *
                "no records to watch; drop `watches`"
        )
    )
    return ()
end
function _kernel_watches(state, watches)
    watches === nothing && throw(
        ArgumentError(
            "a `PairKernel` whose `state` is a projection must declare every " *
                "`individual.state` key that projection reads: " *
                "`PairKernel(callback; state = project, " *
                "watches = (:vaccination_time,))`. Pass `watches = ()` for a " *
                "projection that reads no state key."
        )
    )
    return _watch_keys(watches)
end
_watch_keys(key::Symbol) = (key,)
function _watch_keys(keys)
    all(k -> k isa Symbol, keys) || throw(
        ArgumentError("`watches` must name `individual.state` keys as `Symbol`s")
    )
    return Tuple(keys)
end

"""
    LayerHost

One person of an [`InfectionLayer`](@ref), as a [`PairKernel`](@ref) `state`
function sees them when the likelihood is evaluated. It mirrors a simulated
individual: its population `id`, its `infection_time` (`NaN` if never
infected), and a `state` holding the person's recorded event times under their
keys. `state` reads like an individual's: `state[key]` and
`get(state, key, default)` give the recorded time, including a recorded `NaN`,
and a person whose entry is `missing` has none, so `get` returns the default and
`state[key]` throws.
Reading a key the infection record does not hold throws an `ArgumentError`,
so a `state` function cannot silently fall back to a default for a time the
likelihood was never given.
"""
struct LayerHost{T, S}
    id::Int
    infection_time::T
    state::S
end

# The recorded times of one host of an infection layer, read from the layer's
# columns on demand so that a column holding `missing` stays type-stable to
# read. `missing` marks a time the host does not have, as an absent key does on
# an individual.
struct _LayerHostState{S <: NamedTuple}
    columns::S
    index::Int
end
@inline function _layer_time(s::_LayerHostState, key::Symbol)
    haskey(s.columns, key) || throw(
        ArgumentError(
            "the infection layer holds no host time `$key`; add it to `host_times`"
        )
    )
    return s.columns[key][s.index]
end
@inline function Base.get(s::_LayerHostState, key::Symbol, default)
    value = _layer_time(s, key)
    return ismissing(value) ? default : value
end
@inline function Base.getindex(s::_LayerHostState, key::Symbol)
    value = _layer_time(s, key)
    ismissing(value) && throw(KeyError(key))
    return value
end
Base.haskey(s::_LayerHostState, key::Symbol) = !ismissing(_layer_time(s, key))

function _layer_host(data, i)
    return LayerHost(i, data.infection_time[i], _LayerHostState(host_times(data), i))
end

_pair_state(project, individual) = project(individual)
_pair_state(records::AbstractVector, individual) = records[individual.id]

"""
    watched_records(kernel)

The keys of `individual.state` that `kernel`'s transmission rate depends on,
as a tuple of `Symbol`s. The default, `()`, is a kernel whose rates are fixed
for the whole outbreak.

When one of these values changes for a person during a simulation (a
vaccination date being set, say), the contacts already drawn from a kernel that
lists that key are drawn again under the new rate, given the exposure that has
already happened. A kernel that lists nothing is never redrawn.

Declare every key the kernel reads, whether or not anything in a given model
writes it, and read anything that can move from `individual.state` rather than
from a field of the individual, which this cannot name. A [`PairKernel`](@ref)
reports its `watches`, and `()` once its records are extracted; any other
kernel type declares its own method.
"""
watched_records(kernel) = ()
watched_records(k::PairKernel) = k.watches
# A vector of records cannot move while a likelihood reads it, and a race given
# one has nothing to watch.
watched_records(::PairKernel{F, <:AbstractVector}) where {F} = ()
# A per-edge collection declares the union of what its entries declare. Every
# such entry a network resolves today reads no host state, so this is `()` in
# practice; it keeps the union right if one ever does.
function watched_records(ks::AbstractVector)
    keys = Symbol[]
    for k in ks, key in watched_records(k)
        key in keys || push!(keys, key)
    end
    return Tuple(keys)
end

"""
    kernel_projection(kernel)

The projection a live [`PairKernel`](@ref) reads a host's record through, or
`nothing` for a kernel with no such projection: one with no `state`, a vector
of already-recorded states, or any other kernel type. A per-edge vector
reports the projection its entries share, or `nothing` if they don't share
one.

A continuous-time race uses this alongside [`watched_records`](@ref
EpiBranch.watched_records) to check a kernel's declaration against what it
actually reads: after a case settles, it re-projects every host a pending or
future draw still reads and compares the result against what the projection
read last, throwing when a record it reads has changed without any of the
declared `watches` moving.
"""
kernel_projection(kernel) = nothing
kernel_projection(k::PairKernel) = k.state
# Neither a vector of records nor no `state` at all has anything live to
# re-project; `watched_records` refuses `watches` on both for the same reason.
kernel_projection(::PairKernel{F, Nothing}) where {F} = nothing
kernel_projection(::PairKernel{F, <:AbstractVector}) where {F} = nothing
# A per-edge collection has one projection to check only when every entry
# shares it, the common case of a route built from the same live `PairKernel`
# repeated along each edge; a mix of projections cannot be checked without
# resolving the edge, so it is `nothing` like any other kernel type.
function kernel_projection(ks::AbstractVector)
    isempty(ks) && return nothing
    proj = kernel_projection(first(ks))
    proj === nothing && return nothing
    for k in ks
        kernel_projection(k) === proj || return nothing
    end
    return proj
end

# Evaluate a PairKernel's callback for an ordered pair, given whatever host
# records `state` selects. `Nothing` state is the contextual case (the callback
# takes only the context); an `AbstractVector` state is already recorded and
# indexed directly by population id; anything else is a live projection that
# needs the running `SimulationState` to apply to each `Individual`.
_pair_result(k::PairKernel{F, Nothing}, ctx, i, j) where {F} = k.callback(ctx)
_pair_result(k::PairKernel{F, Nothing}, ctx, i, j, ::SimulationState) where {F} = k.callback(ctx)
function _pair_result(k::PairKernel{F, <:AbstractVector}, ctx, i, j) where {F}
    return k.callback(ctx, k.state[i], k.state[j])
end
function _pair_result(k::PairKernel{F, <:AbstractVector}, ctx, i, j, ::SimulationState) where {F}
    return k.callback(ctx, k.state[i], k.state[j])
end
function _pair_result(k::PairKernel, ctx, i, j, state::SimulationState)
    # A live projection reads both records whether or not either is
    # `watched_records`-declared: an undeclared one is read only this once, a
    # declared one again whenever it moves. Either way the target's record is
    # needed on the first contact considered, so this realises it on a
    # lazily populated model (see `EpiBranch.lazily_populate!`) rather than
    # waiting for `traits` or a watched key to force it.
    return k.callback(
        ctx, _pair_state(k.state, _realised!(state, i)),
        _pair_state(k.state, _realised!(state, j))
    )
end
function _pair_result(::PairKernel, ctx, i, j)
    throw(
        ArgumentError(
            "a live PairKernel needs simulation state; " *
                "supply recorded host states for likelihood evaluation with record_kernel"
        )
    )
end

# Turn a callback's result into the usable contact-interval kernel. A bare
# profile is scaled by the kernel's own calendar; a `(profile, calendar)` named
# tuple names the pair's schedule, falling back to the kernel's when it omits one.
_finish_kernel(k::PairKernel, result, opening) = _calendar_scaled(result, k.calendar, opening)
function _finish_kernel(k::PairKernel, result::NamedTuple, opening)
    return _calendar_scaled(result.profile, get(result, :calendar, k.calendar), opening)
end

# The profile unchanged with no calendar in force, or its hazard scaled by the
# calendar schedule from the infector's infectious `opening`.
_calendar_scaled(profile, ::Nothing, opening) = profile
_calendar_scaled(profile, ::Nothing, ::Nothing) = profile
function _calendar_scaled(profile, sched, ::Nothing)
    throw(
        ArgumentError("PairKernel's calendar schedule needs the infector's infectious opening time")
    )
end
_calendar_scaled(profile, sched, opening) = _CalendarScaledKernel(profile, sched, opening)

pair_kernel(k::PairKernel, i, j, infection_time) =
    _finish_kernel(k, _pair_result(k, PairContext(i, j, infection_time), i, j), nothing)
function pair_kernel(k::PairKernel, i, j, infection_time, infectious_time)
    return _finish_kernel(
        k, _pair_result(k, PairContext(i, j, infection_time), i, j), infectious_time
    )
end
function pair_kernel(k::PairKernel, i, j, infection_time, infectious_time, state::SimulationState)
    return _finish_kernel(
        k, _pair_result(k, PairContext(i, j, infection_time), i, j, state), infectious_time
    )
end

"""
    pair_kernel(kernel, infector, susceptible, infector_infection_time)
    pair_kernel(kernel, infector, susceptible, infector_infection_time, infectious_time)
    pair_kernel(kernel, infector, susceptible, infector_infection_time, infectious_time,
        state)

The contact-interval distribution for one infector and one susceptible person.
A shared distribution is returned unchanged; a function receives the two
people's numbers; a [`PairKernel`](@ref) receives a [`PairContext`](@ref) and,
when it has a `state`, each person's record. The five-argument form adds the
start of the infector's infectious period, which a `PairKernel` with a calendar
schedule needs. The six-argument form also passes the `SimulationState`, from
which a `PairKernel` with a `state` function reads both people's records;
simulation must use it, and the shorter forms, meant for the likelihood, throw
for such a kernel. Every other kernel returns what the five-argument form does.
"""
pair_kernel(k::ContinuousUnivariateDistribution, i, j, infection_time) = k
pair_kernel(k, i, j, infection_time) = k(i, j)

# The five-argument interface also supplies the origin of the contact interval.
pair_kernel(k, i, j, infection_time, infectious_time) = pair_kernel(k, i, j, infection_time)

# The extra argument is supplied only by simulation; ordinary kernels retain
# their existing extension methods and compiled likelihood fast paths.
function pair_kernel(k, i, j, infection_time, opening, state)
    return pair_kernel(k, i, j, infection_time, opening)
end

"""
    record_kernel(kernel, state::SimulationState)

Prepare a [`PairKernel`](@ref) for fitting to a simulated outbreak: returns a
kernel with the same callback and calendar whose `state` is the vector of each
person's record at the end of `state`. Each record is copied, so later changes
to the simulation do not alter it. Other kernels are returned unchanged.

Only the record the `state` function returns at the end of the run is kept;
values overwritten during the run are lost. Where rates change, the
function must return event dates or full histories. The likelihood then gives
the transmission part along these histories; the probabilities of the
characteristics and intervention events themselves need their own terms if
they belong in the joint likelihood. During inference, build
`PairKernel(callback; state = records)` from the current values of unobserved
records at each evaluation.
"""
record_kernel(k, state::SimulationState) = k
record_kernel(k::PairKernel{F, Nothing}, state::SimulationState) where {F} = k
function record_kernel(k::PairKernel, state::SimulationState)
    return PairKernel(
        k.callback,
        [deepcopy(_pair_state(k.state, ind)) for ind in state.individuals],
        k.calendar, ()
    )
end

# ── Calendar-scaled kernels ───────────────────────────────────────────
#
# The contact interval since `opening` under a `profile` hazard multiplied by a
# calendar schedule on the calendar-time axis. How the cumulative hazard is
# integrated and inverted depends on the schedule's `calendar_shape`. A
# piecewise-constant schedule splits it into segments, each a scaled difference
# of the profile's cumulative hazard, which keeps simulation and the likelihood
# exact wherever the profile's own hazard is. A smooth schedule integrates the
# scaled hazard by quadrature and inverts it by bisection.

struct _CalendarScaledKernel{P, S, T <: Real}
    profile::P
    calendar::S
    opening::T
end

function SurvivalDistributions.cumhazard(k::_CalendarScaledKernel, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be non-negative"))
    return _calendar_cumhazard(calendar_shape(k.calendar), k, τ)
end

function SurvivalDistributions.loghazard(k::_CalendarScaledKernel, τ::Real)
    lh = _loghazard(k.profile, τ)
    # Past the profile's support the hazard is zero whatever the calendar says.
    lh == -Inf && return lh
    return lh + log(calendar_multiplier(k.calendar, k.opening + τ))
end

# `loghazard` of a truncated kernel is `logpdf - logccdf`, which is `-Inf - -Inf`
# beyond the upper bound. The hazard there is zero and its log is `-Inf`.
function _loghazard(k, τ::Real)
    lh = loghazard(k, τ)
    # Other NaNs (a NaN time or NaN kernel parameters) are left to propagate.
    isnan(lh) && !isnan(τ) && logpdf(k, τ) == -Inf && return oftype(lh, -Inf)
    return lh
end

SurvivalDistributions.hazard(k::_CalendarScaledKernel, τ::Real) = _calendar_hazard(k, τ)

Distributions.logccdf(k::_CalendarScaledKernel, τ::Real) = -cumhazard(k, τ)

function Distributions.invlogccdf(k::_CalendarScaledKernel, lp::Real)
    isfinite(lp) || return oftype(float(lp), Inf)
    lp <= 0 || throw(ArgumentError("invlogccdf needs a non-positive log-survival"))
    return _calendar_invlogccdf(calendar_shape(k.calendar), k, float(-lp))
end

Base.rand(rng::AbstractRNG, k::_CalendarScaledKernel) = _time_at_log_survival(k, log(rand(rng)))
Base.minimum(k::_CalendarScaledKernel) = minimum(k.profile)
Base.maximum(k::_CalendarScaledKernel) = maximum(k.profile)
Distributions.partype(k::_CalendarScaledKernel) = Distributions.partype(k.profile)

# The segment starting at calendar time `calendar_lo`: its multiplier and the
# calendar time its schedule step ends at (`Inf` for the last step). The walk
# stays in calendar time throughout rather than converting back from
# time-since-opening: a break the segment reaches is carried into the next
# segment unrounded. Converting would let `opening + t_lo` round short of the
# break it was meant to equal, so the walk would keep reaching the same break
# without moving past it.
function _calendar_segment(k::_CalendarScaledKernel, calendar_lo)
    m = calendar_multiplier(k.calendar, calendar_lo)
    calendar_hi = next_calendar_break(k.calendar, calendar_lo)
    while isfinite(calendar_hi) && calendar_hi <= calendar_lo
        calendar_hi = next_calendar_break(k.calendar, calendar_hi)
    end
    return m, calendar_hi
end

function _calendar_cumhazard(::PiecewiseConstantCalendar, k::_CalendarScaledKernel, τ)
    # A segment with a zero multiplier adds no hazard, even past the end of a
    # bounded profile, and survival mass can remain there. A positive segment
    # that starts at or beyond that end has an infinite total, returned
    # directly because the difference of two infinite endpoint values would
    # read `Inf - Inf`.
    _, upper, certain = _profile_bounds(k.profile)
    total = zero(float(τ))
    calendar_end = k.opening + τ
    calendar_lo = k.opening
    while calendar_lo < calendar_end
        m, calendar_hi = _calendar_segment(k, calendar_lo)
        calendar_hi = min(calendar_end, calendar_hi)
        if m != 0
            certain && calendar_lo - k.opening >= upper && return oftype(total, Inf)
            total += m * (
                cumhazard(k.profile, calendar_hi - k.opening) -
                    cumhazard(k.profile, calendar_lo - k.opening)
            )
        end
        calendar_lo = calendar_hi
    end
    return total
end

# Invert a target cumulative hazard segment by segment: consume each step's
# cumulative hazard budget until the remaining budget is met inside the
# current step, then invert within the profile alone with the same machinery
# used for a constant rate multiplier.
function _calendar_invlogccdf(::PiecewiseConstantCalendar, k::_CalendarScaledKernel, target)
    # Zero-multiplier segments are skipped. A draw whose target outlasts every
    # positive segment is `Inf` (the contact never happens), even for a
    # bounded profile. A positive segment that starts at or beyond the end of
    # a bounded profile settles the draw at the start of the segment, where the
    # cumulative hazard jumps to infinity immediately afterwards.
    _, upper, certain = _profile_bounds(k.profile)
    acc = zero(target)
    calendar_lo = k.opening
    while true
        t_lo = calendar_lo - k.opening
        m, calendar_hi = _calendar_segment(k, calendar_lo)
        if m > 0
            certain && t_lo >= upper && return t_lo
            base = cumhazard(k.profile, t_lo)
            seg_cum = isfinite(calendar_hi) ?
                m * (cumhazard(k.profile, calendar_hi - k.opening) - base) :
                oftype(target, Inf)
            if acc + seg_cum >= target
                want = base + (target - acc) / m
                return _time_at_log_survival(k.profile, -want)
            end
            acc += seg_cum
        end
        isfinite(calendar_hi) || return oftype(target, Inf)
        calendar_lo = calendar_hi
    end
    return
end

# The scaled hazard at time-since-opening `s`. A zero multiplier gives zero
# even where the profile's hazard is infinite.
function _calendar_hazard(k::_CalendarScaledKernel, s)
    # The profile first: where its hazard is already zero (a `HazardKernel`
    # beyond `upper`, or a trajectory that has cleared) the calendar is not
    # evaluated, because a bracket grown far past the profile's reach would
    # otherwise hand a smooth calendar a time whose phase overflows.
    lh = loghazard(k.profile, s)
    isfinite(lh) || return zero(float(s))
    m = calendar_multiplier(k.calendar, k.opening + s)
    return iszero(m) ? zero(m) * zero(s) : m * exp(lh)
end

# Where the profile's survival has certainly ended (the top of a bounded
# support, beyond which its hazard is undefined) the contact interval ends
# too. Where its integration merely stops (a `HazardKernel`'s `upper`) the
# total up to there is the total, as `_calendar_invlogccdf` treats it.
# Quadrature cannot settle an integral over an unbounded horizon for a
# multiplier that keeps oscillating, so that integral is taken to be infinite.
function _calendar_cumhazard(::SmoothCalendar, k::_CalendarScaledKernel, τ)
    start, upper, certain = _profile_bounds(k.profile)
    certain && τ >= upper && return oftype(float(τ), Inf)
    # `upper` is reached only by a `HazardKernel`, a bounded named support
    # having returned above, and is kept as it is so a dual in it survives.
    hi = τ < upper ? float(τ) : upper
    isfinite(hi) || return oftype(hi, Inf)
    # From `start`, as the inverse brackets: a run of zero hazard before the
    # onset would otherwise hide a short positive stretch from quadrature's
    # first pass over the whole range. The range is mapped onto the unit
    # interval so a dual at the support's bottom enters as part of the
    # integrand, where its boundary term -λ(start) start' is picked up,
    # rather than as an endpoint, which quadrature does not take.
    hi <= start && return zero(hi)
    return _hazard_integral_from(s -> _calendar_hazard(k, s), start, hi)
end

# Where a profile's hazard is first positive, where its integration stops,
# and whether its survival has certainly ended there: a named distribution's
# hazard starts at the bottom of its support (or at zero, for one that
# reaches below it) and the top of a bounded support ends it, while a
# `HazardKernel` starts at its `onset` and its `upper` only bounds the
# integration, as in `_hazard_invert`.
function _profile_bounds(profile)
    m = float(minimum(profile))
    return max(zero(m), m), float(maximum(profile)), true
end
_profile_bounds(k::HazardKernel) = k.onset, k.upper, false
# A truncated kernel starts where its base kernel does and its survival ends
# at the cap, or at the base kernel's `upper` where that comes first, beyond
# which the conditional hazard has already diverged; except where the window
# has no mass at all: the start then sits at or past the end, nothing is
# integrated, and a draw is `Inf` here as it is without a calendar.
function _profile_bounds(k::_TruncatedHazardKernel)
    return k.kernel.onset, min(k.kernel.upper, k.hi), !_no_mass(k)
end

# A value stripped of every dual level, for the ends of a bracket.
_plain(x::Real) = x
_plain(x::ForwardDiff.Dual) = _plain(ForwardDiff.value(x))

function _calendar_invlogccdf(::SmoothCalendar, k::_CalendarScaledKernel, target)
    rate = s -> _calendar_hazard(k, s)
    start, upper, certain = _profile_bounds(k.profile)
    t, t_acc = _invert_hazard(rate, start, upper, certain, target)
    t_acc < target && return oftype(t, Inf)
    return _implicit_hazard_step(rate, t, t_acc, target)
end

# ── Shared quadrature hazard inversion ───────────────────────────────
#
# A `HazardKernel`'s own hazard and a calendar-scaled smooth schedule's
# combined hazard are both given only as a function to integrate, with no
# closed form for their cumulative hazard or its inverse. Both find a time
# from a target cumulative hazard the same way, so they share the machinery
# below rather than each doubling and bisecting on their own.

# Integrate `rate` over `[lo, hi]` by quadrature.
_hazard_integral(rate, lo, hi) = first(quadgk(rate, lo, hi))

# Integrate `rate` over `[lo, hi]` through the unit interval, so that a bound
# holding a dual parameter (an `onset` or `upper` being fitted) enters as part
# of the integrand, where its boundary term is kept, rather than as an
# endpoint, which quadrature does not take. The interior breakpoints are the
# doublings the inverter's bracket walks: a hazard active for a short stretch
# at the start of a long range is otherwise missed by the first pass over the
# whole range, every node of which reads zero.
# The segments are walked one by one rather than handed to quadgk as
# breakpoints, which would splat a vector of runtime length and lose the
# return type on every cumulative-hazard read.
function _hazard_integral_from(rate, lo, hi)
    w = hi - lo
    wp = _plain(w)
    g = u -> rate(lo + u * w) * w
    total = _hazard_integral(g, 0.0, min(1.0, 1.0 / wp))
    reach = 1.0
    while reach < wp
        nxt = 2reach + 1
        # What is already summed sets an absolute scale for a tail segment,
        # as one quadrature over the whole range would have it: a segment
        # holding a negligible share of a decaying hazard is not refined on
        # its own account. The relative tolerance is passed as well, since
        # quadgk drops it once an absolute one is given, and a segment that
        # dwarfs everything before it (a steep rise after a faint run-up)
        # would otherwise be refined to the evaluation cap.
        rtol = sqrt(eps(one(wp)))
        atol = rtol * abs(_plain(total))
        total += first(quadgk(g, reach / wp, min(nxt, wp) / wp; atol, rtol))
        reach = nxt
    end
    return total
end

# Bisect `[lo, hi]` down to adjacent floats; `acc` and `hi_acc`, the integrals
# of `rate` from 0 to each end, bracket `target`. Returns the bisected time
# and its own integral, the second needed only by a caller that refines it
# further.
function _hazard_bisect(rate, lo, hi, acc, hi_acc, target)
    hi_acc >= target || return hi, hi_acc
    while true
        mid = lo + (hi - lo) / 2
        (lo < mid < hi) || return hi, hi_acc
        mid_acc = acc + _hazard_integral(rate, lo, mid)
        if mid_acc >= target
            hi, hi_acc = mid, mid_acc
        else
            lo, acc = mid, mid_acc
        end
    end
    return
end

# The bisection compares values only, so the time it finds carries no
# derivative in `rate`'s own parameters. One implicit-function step
# `t + (target - Λ(t)) / λ(t)` gives it the derivative `-(∂Λ/∂θ) / λ` that the
# drawn time has. `Λ(t)` is the bisection's own running integral, which
# brackets `target` within one bisection step, so the step moves the value by
# no more than that step even where the rate is small; a separately computed
# integral would have a different quadrature error, which a small rate
# would magnify into the drawn time.
function _implicit_hazard_step(rate, t, t_acc, target)
    isfinite(t_acc) || return t
    r = rate(t)
    (iszero(r) || !isfinite(r)) && return t
    return t + (target - t_acc) / r
end

# Grow a bracket `[start, hi]` from `start` by doubling `width` each step
# until the integral of `rate` reaches `target` or `hi` meets `upper`, then
# bisect the last interval. `boundary_is_certain` says what the cumulative
# hazard is once `hi` reaches `upper` without reaching `target`: `true` for a
# profile whose survival has certainly ended there, so the remaining hazard
# is treated as infinite rather than integrated through what may be a
# singularity at the edge (a calendar-scaled profile's own bounded support);
# `false` for a bound the integration simply stops at, handing back a total
# short of `target` for the caller to treat as a draw that never happens (a
# `HazardKernel`'s `upper`).
#
# `start` must already be where `rate` is positive: a bracket that starts
# where `rate` is zero cannot tell a `rate` still flat for now, about to
# resume, from one that has converged to a finite total below `target` and
# will never reach it, and grows all the way to `upper` (or, with `upper`
# unbounded, until `hi` overflows to `Inf`) to find out either way. Callers
# that let `rate` start flat, such as a `HazardKernel`'s own `onset`, rule
# that out once, before the bracket is grown at all, rather than leaving this
# loop to discover it slowly.
function _invert_hazard(rate, start, upper, boundary_is_certain::Bool, target::Real)
    # The bracket ends are plain floats: quadrature takes no dual endpoint,
    # and the drawn time keeps its derivative through the integrand and the
    # implicit step rather than through where the bracket happens to sit.
    start, upper = _plain(start), _plain(upper)
    iszero(target) && return start, zero(target)
    lo = start
    acc = zero(target)
    width = one(lo)
    while true
        hi = min(lo + width, upper)
        isfinite(hi) || return oftype(target, Inf), oftype(target, Inf)
        hi_acc = hi == upper && boundary_is_certain ? oftype(target, Inf) :
            acc + _hazard_integral(rate, lo, hi)
        hi_acc >= target && return _hazard_bisect(rate, lo, hi, acc, hi_acc, target)
        hi == upper && return upper, hi_acc
        lo, acc = hi, hi_acc
        width *= 2
    end
    return
end
