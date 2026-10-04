"""
    PairContext(infector, susceptible, infector_infection_time)

Information available to a pair kernel in both simulation and an infection-layer
likelihood. `infector` and `susceptible` are population IDs. The infector's infection
time retains its number type, including automatic-differentiation values.

The susceptible's eventual infection time is excluded: it is unknown when
simulation selects the contact-interval distribution. Fixed host covariates can
be indexed by either ID in tables captured by the kernel callback.
"""
struct PairContext{T <: Real}
    infector::Int
    susceptible::Int
    infector_infection_time::T
end

"""
    Steps(breaks, values)

A piecewise-constant multiplier on calendar time: `values[1]` before
`breaks[1]`, `values[k+1]` from `breaks[k]` (inclusive) to `breaks[k+1]`, and
`values[end]` from `breaks[end]` onwards. `breaks` must be finite and strictly
increasing, and every value non-negative; a value of zero switches transmission
off from that point on.

Used as the `calendar` a [`PairKernel`](@ref) multiplies its contact-interval
hazard by, on the calendar-time axis rather than time since infectious opening.
It is the piecewise-constant implementation of the calendar schedule interface,
[`calendar_multiplier`](@ref EpiBranch.calendar_multiplier) and
[`next_calendar_break`](@ref EpiBranch.next_calendar_break).
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

The non-negative multiplier a calendar `schedule` applies to a
[`PairKernel`](@ref)'s contact-interval hazard at calendar time `t`.

Every schedule implements this. A new schedule is a type with this method and,
depending on its [`calendar_shape`](@ref EpiBranch.calendar_shape), either
[`next_calendar_break`](@ref EpiBranch.next_calendar_break) (piecewise
constant, the default) or a `calendar_shape` method declaring it smooth.
[`Steps`](@ref) is the piecewise-constant schedule the package provides.
"""
calendar_multiplier(s::Steps, t::Real) = s.values[searchsortedlast(s.breaks, t) + 1]

"""
    next_calendar_break(schedule, t)

The first calendar time strictly after `t` at which a piecewise-constant
`schedule` changes value, or `Inf` if it stays constant from `t` onwards.

Simulation and the likelihood integrate a piecewise-constant schedule exactly,
one constant segment at a time, so they rely on this to find where each segment
ends. A smooth schedule has no breaks and does not implement it.
"""
function next_calendar_break(s::Steps, t::Real)
    idx = searchsortedlast(s.breaks, t) + 1
    return idx <= length(s.breaks) ? s.breaks[idx] : oftype(float(t), Inf)
end

"""
    PiecewiseConstantCalendar()

The [`calendar_shape`](@ref EpiBranch.calendar_shape) of a schedule that is
constant between the breaks [`next_calendar_break`](@ref
EpiBranch.next_calendar_break) reports. Its cumulative hazard is the exact sum
of scaled differences of the profile's own, one per segment.
"""
struct PiecewiseConstantCalendar end

"""
    SmoothCalendar()

The [`calendar_shape`](@ref EpiBranch.calendar_shape) of a schedule with no
breaks, whose multiplier may change continuously. Its cumulative hazard is the
integral of the multiplier times the profile's hazard, computed by adaptive
Gauss–Kronrod quadrature, and a contact interval is drawn by bisecting that
integral for the target log-survival. The schedule's multiplier should be
smooth enough to integrate accurately and is evaluated many times per draw.
The cumulative hazard over an unbounded horizon is taken to be infinite, so a
smooth multiplier is assumed not to switch transmission off for good.
"""
struct SmoothCalendar end

"""
    calendar_shape(schedule)

How a calendar `schedule` is integrated: [`PiecewiseConstantCalendar`](@ref
EpiBranch.PiecewiseConstantCalendar), the default, which needs
[`next_calendar_break`](@ref EpiBranch.next_calendar_break), or
[`SmoothCalendar`](@ref EpiBranch.SmoothCalendar), which a smooth schedule
declares for its own type:

```julia
struct Seasonal{T <: Real}
    amplitude::T
end
EpiBranch.calendar_multiplier(s::Seasonal, t) = 1 + s.amplitude * sin(2π * t / 365)
EpiBranch.calendar_shape(::Seasonal) = EpiBranch.SmoothCalendar()
```
"""
calendar_shape(schedule) = PiecewiseConstantCalendar()

"""
    PairKernel(callback; state = nothing, calendar = nothing, watches = nothing)

A pair kernel: `callback` returns the contact-interval profile, measured from
the infector's infectious opening, and optionally a calendar schedule that
multiplies the rate on the calendar. Supported by `NetworkProcess`,
`HouseholdProcess`, and the `InfectionLayer` forms of
[`pairwise_surv_loglik`](@ref).

With `state = nothing`, `callback(context::PairContext)` returns a
contact-interval distribution, or a `(profile, calendar)` named tuple. This is
the case where a kernel reads only fixed covariates and the infector's
infection time:

```julia
kernel = PairKernel(context -> Exponential(exp(0.1 * context.infector_infection_time)))
```

With `state` given, `callback(context, source, target)` also receives each
host's record: `state(individual)` selects it in simulation, from an
`EpiBranch.Individual`. For likelihood evaluation, supply a vector of records
indexed by population ID as `state`, or use [`record_kernel`](@ref) to extract
them after simulation; the callback is identical in both paths.

`watches` names every `individual.state` key the projection reads, as a tuple
of `Symbol`s, and is what [`EpiBranch.watched_records`](@ref) reports. A race
redraws a case's pending contacts when one of these keys moves on a host it
reads, so a key left out is a hazard that changes without the contacts
following it — declare each one the projection reads, whether or not anything
in today's model writes it. It is required with a projection, since no default
is safe; `()` is for a projection that reads no state key, such as one indexing
a table by `ind.id`. Only `individual.state` is followed, so anything that can
move has to be read from there rather than from a field such as
`ind.susceptibility`. With no `state`, or with a vector of records, there is
nothing that can move and so nothing to declare: `watches` is refused there
rather than ignored.

`calendar`, a [`Steps`](@ref) schedule or any type implementing
[`calendar_multiplier`](@ref EpiBranch.calendar_multiplier), multiplies the
returned profile's hazard by the schedule's value at the calendar date (the infector's infectious
opening plus time elapsed). A pair whose schedule differs from the shared one —
because it depends on a host's record — returns it instead from the callback,
as `(profile = ..., calendar = ...)`; that overrides the kernel's own
`calendar` for that pair.

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

The pair's hazard is the profile's hazard at time since opening, multiplied by
the schedule's value on the calendar day. For a piecewise-constant schedule
such as `Steps`, simulation and the likelihood both compute this exactly,
splitting the cumulative hazard into segments at the schedule's breakpoints. A
schedule declaring [`SmoothCalendar`](@ref EpiBranch.SmoothCalendar) is
integrated by quadrature in both.

Callbacks must describe a predictable hazard: adding an event at time `t` must
not change the hazard before `t`. A final vaccinated flag alone is insufficient;
retain its date and the earlier hazard. State changes occur through the existing
case-resolution and intervention hooks. The callback and projection must not
mutate state, and a kernel may read host state only through its projection,
since that record is all the likelihood is given.

Simulation keeps contacts consistent with the hazards in force as records
change, and a run whose records never change follows the same distribution as
an ordinary kernel. A kernel that declares watched records puts every
household of a household model on one clock, so that a policy can read cases in
other households, which draws the same outbreak from a different random
stream.

A plain distribution remains the simplest kernel and needs no `PairKernel`
wrapper.
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

One host of an [`InfectionLayer`](@ref) as a live [`PairKernel`](@ref)
projection sees it in a likelihood: its population `id`, its `infection_time`
(`NaN` if never infected), and a `state` holding the layer's per-host times
under their keys. `state` reads like an individual's: `state[key]` and
`get(state, key, default)` give the recorded time, including a recorded `NaN`,
and a host whose entry is `missing` has none, so `get` returns the default and
`state[key]` throws.
Reading a key the layer did not record throws an `ArgumentError`, so a
projection cannot silently fall back to a default for a time the likelihood
was never given.
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
    return LayerHost(i, data.infection_time[i], _LayerHostState(_host_times(data), i))
end

_pair_state(project, individual) = project(individual)
_pair_state(records::AbstractVector, individual) = records[individual.id]
# The projection a kernel reads host state through, or `nothing` for one that
# needs no running state to evaluate: a kernel given records, or none at all. A
# kernel may depend on host state only through this record, which is what lets
# `record_kernel` reproduce a run's hazards from recorded records alone. Which
# of those records a race watches is the kernel's own declaration, through
# `watched_records`.
_kernel_projection(k) = nothing
_kernel_projection(k::PairKernel) = k.state
_kernel_projection(k::PairKernel{F, <:AbstractVector}) where {F} = nothing
_live_kernel(k) = _kernel_projection(k) !== nothing

"""
    watched_records(kernel)

The `individual.state` keys `kernel`'s hazards depend on, as a tuple of
`Symbol`s. `()`, the default, is a kernel whose hazards are fixed for a run.

A continuous-time race keeps drawn contacts consistent with the hazards in
force: when one of these keys moves on a host, the contacts drawn from a kernel
declaring that key are drawn again, conditioned on the exposure already
elapsed. The race watches the union of the keys its routes declare and compares
only the hosts a pending or future draw reads, so a key no kernel declares
costs nothing and a route that declares nothing is never redrawn.

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
    return k.callback(
        ctx, _pair_state(k.state, state.individuals[i]),
        _pair_state(k.state, state.individuals[j])
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

Resolve a contact-interval distribution for an ordered pair. A shared continuous
distribution is returned unchanged; an ordinary callable receives the two IDs;
a [`PairKernel`](@ref) receives a [`PairContext`](@ref) and, when it has host
state, each host's record. The five-argument form supplies the infectious
opening required by a `PairKernel` with a calendar schedule. The six-argument
form also passes the `SimulationState`, from which a live `PairKernel` reads
both hosts' records; simulation must use it, since the shorter forms are for
likelihoods and throw for a live kernel. Every other kernel returns what the
five-argument form does.
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

Return a kernel with the same callback and calendar and a vector of host
records extracted from a finished simulation. For a live [`PairKernel`](@ref)
(one whose `state` is a projection function), apply the projection to each
individual and copy the results so later simulation mutations cannot change the
record. Other kernels are returned unchanged.

The projection must retain event dates or full histories when hazards change.
This is extraction, not automatic history logging: overwritten past values cannot
be recovered. The resulting likelihood evaluates the transmission contribution along these
histories. Include separate attribute/intervention models when their probabilities
also belong in the joint likelihood. During inference, construct
`PairKernel(callback; state = records)` with the current latent records on each call.
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
    return loghazard(k.profile, τ) + log(calendar_multiplier(k.calendar, k.opening + τ))
end

Distributions.logccdf(k::_CalendarScaledKernel, τ::Real) = -cumhazard(k, τ)

function Distributions.invlogccdf(k::_CalendarScaledKernel, lp::Real)
    isfinite(lp) || return oftype(float(lp), Inf)
    lp <= 0 || throw(ArgumentError("invlogccdf needs a non-positive log-survival"))
    return _calendar_invlogccdf(calendar_shape(k.calendar), k, float(-lp))
end

Base.rand(rng::AbstractRNG, k::_CalendarScaledKernel) = _time_at_log_survival(k, log(rand(rng)))
Base.minimum(k::_CalendarScaledKernel) = minimum(k.profile)
Distributions.partype(k::_CalendarScaledKernel) = Distributions.partype(k.profile)

# The segment starting at time-since-opening `t_lo`: its multiplier and the
# time-since-opening its schedule step ends at (`Inf` for the last step).
function _calendar_segment(k::_CalendarScaledKernel, t_lo)
    calendar_t = k.opening + t_lo
    m = calendar_multiplier(k.calendar, calendar_t)
    next_break = next_calendar_break(k.calendar, calendar_t)
    seg_hi = isfinite(next_break) ? next_break - k.opening : oftype(float(t_lo), Inf)
    return m, seg_hi
end

function _calendar_cumhazard(::PiecewiseConstantCalendar, k::_CalendarScaledKernel, τ)
    total = zero(float(τ))
    t_lo = zero(float(τ))
    while t_lo < τ
        m, seg_hi = _calendar_segment(k, t_lo)
        t_hi = min(τ, seg_hi)
        m == 0 ||
            (total += m * (cumhazard(k.profile, t_hi) - cumhazard(k.profile, t_lo)))
        t_lo = t_hi
    end
    return total
end

# Invert a target cumulative hazard segment by segment: consume each step's
# cumulative hazard budget until the remaining budget is met inside the
# current step, then invert within the profile alone with the same machinery
# used for a constant rate multiplier.
function _calendar_invlogccdf(::PiecewiseConstantCalendar, k::_CalendarScaledKernel, target)
    acc = zero(target)
    t_lo = zero(target)
    while true
        m, seg_hi = _calendar_segment(k, t_lo)
        if m > 0
            base = cumhazard(k.profile, t_lo)
            seg_cum = isfinite(seg_hi) ?
                m * (cumhazard(k.profile, seg_hi) - base) : oftype(target, Inf)
            if acc + seg_cum >= target
                want = base + (target - acc) / m
                return _time_at_log_survival(k.profile, -want)
            end
            acc += seg_cum
        end
        isfinite(seg_hi) || return oftype(target, Inf)
        t_lo = seg_hi
    end
    return
end

# The scaled hazard at time-since-opening `s`. A zero multiplier gives zero
# even where the profile's hazard is infinite.
function _calendar_hazard(k::_CalendarScaledKernel, s)
    m = calendar_multiplier(k.calendar, k.opening + s)
    return iszero(m) ? zero(m) * zero(s) : m * exp(loghazard(k.profile, s))
end

function _calendar_integral(k::_CalendarScaledKernel, lo, hi)
    return first(quadgk(s -> _calendar_hazard(k, s), lo, hi))
end

# The profile's survival reaches zero at the top of a bounded support, beyond
# which its hazard is undefined, so the contact interval ends there. Quadrature
# cannot settle an integral over an unbounded horizon for a multiplier that
# keeps oscillating, so that integral is taken to be infinite.
function _calendar_cumhazard(::SmoothCalendar, k::_CalendarScaledKernel, τ)
    (isfinite(τ) && τ < maximum(k.profile)) || return oftype(float(τ), Inf)
    iszero(τ) && return zero(float(τ))
    return _calendar_integral(k, zero(float(τ)), float(τ))
end

# Find the time at which the integrated hazard reaches `target`: grow a bracket
# by doubling until it does, then bisect the last interval to adjacent floats.
# Each step integrates only the new interval and adds it to the running total,
# which is monotone because the scaled hazard is non-negative.
function _calendar_invlogccdf(::SmoothCalendar, k::_CalendarScaledKernel, target)
    iszero(target) && return target
    upper = float(maximum(k.profile))
    lo = zero(target)
    acc = zero(target)
    width = one(target)
    while true
        hi = lo + width
        hi >= upper && (hi = oftype(lo, upper))
        isfinite(hi) || return oftype(target, Inf)
        hi_acc = hi == upper ? oftype(target, Inf) : acc + _calendar_integral(k, lo, hi)
        if hi_acc >= target
            t, t_acc = _bisect_calendar(k, lo, hi, acc, hi_acc, target)
            return _implicit_step(k, t, t_acc, target)
        end
        lo, acc = hi, hi_acc
        width *= 2
    end
    return
end

# The bisection compares values only, so the time it finds carries no
# derivative in the schedule's parameters. One implicit-function step
# `t + (target - Λ(t)) / λ(t)` gives it the derivative `-(∂Λ/∂θ) / λ` that the
# drawn time has. `Λ(t)` is the bisection's own running integral, which
# brackets `target` within one bisection step, so the step moves the value by
# no more than that step even where the hazard is small; a separately computed
# integral would carry a different quadrature error, which a small hazard
# would magnify into the drawn time.
function _implicit_step(k::_CalendarScaledKernel, t, t_acc, target)
    isfinite(t_acc) || return t
    rate = _calendar_hazard(k, t)
    (iszero(rate) || !isfinite(rate)) && return t
    return t + (target - t_acc) / rate
end

# Bisect `[lo, hi]`, whose running integrals `acc` and `hi_acc` bracket
# `target`, down to adjacent floats; return the upper end and its integral.
function _bisect_calendar(k::_CalendarScaledKernel, lo, hi, acc, hi_acc, target)
    while true
        mid = lo + (hi - lo) / 2
        (lo < mid < hi) || return hi, hi_acc
        mid_acc = acc + _calendar_integral(k, lo, mid)
        if mid_acc >= target
            hi, hi_acc = mid, mid_acc
        else
            lo, acc = mid, mid_acc
        end
    end
    return
end
