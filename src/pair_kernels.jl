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
function (s::Steps)(t::Real)
    return s.values[searchsortedlast(s.breaks, t) + 1]
end

"""
    PairKernel(callback; state = nothing, calendar = nothing)

A pair kernel: `callback` returns the contact-interval profile, measured from
the infector's infectious opening, and optionally a step schedule that
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

`calendar`, a [`Steps`](@ref) schedule, multiplies the returned profile's
hazard by the schedule's value at the calendar date (the infector's infectious
opening plus time elapsed). A pair whose schedule differs from the shared one —
because it depends on a host's record — returns it instead from the callback,
as `(profile = ..., calendar = ...)`; that overrides the kernel's own
`calendar` for that pair.

```julia
PairKernel((ctx, source, target) -> Gamma(2.0, 1.5);
           calendar = Steps([30.0], [1.0, 0.25]),   # shared policy: rate falls to a quarter on day 30
           state = ind -> (age = ind.state[:age],))  # optional per-person record

PairKernel((ctx, source, target) ->
               isfinite(target.date) ?
               (profile = Exponential(2.5), calendar = Steps([target.date], [1.0, 0.25])) :
               Exponential(2.5);
           state = ind -> (date = get(ind.state, :policy_time, Inf)::Float64,))
```

The pair's hazard is the profile's hazard at time since opening, multiplied by
the schedule's value on the calendar day. Simulation and the likelihood both
score this exactly, splitting the cumulative hazard into segments at the
schedule's breakpoints.

Callbacks must describe a predictable hazard: adding an event at time `t` must
not change the hazard before `t`. A final vaccinated flag alone is insufficient;
retain its date and the earlier hazard. State changes occur through the existing
case-resolution and intervention hooks. The callback and projection must not
mutate state, and a kernel may read host state only through its projection,
since that record is all the likelihood is given.

Simulation keeps contacts consistent with the hazards in force as records
change, and a run whose records never change follows the same distribution as
an ordinary kernel. With interventions, a household model races every household
together so that a policy can read cases in other households, which draws the
same outbreak from a different random stream.

A plain distribution remains the simplest kernel and needs no `PairKernel`
wrapper.
"""
struct PairKernel{F, S, C}
    callback::F
    state::S
    calendar::C
end
PairKernel(callback; state = nothing, calendar = nothing) = PairKernel(callback, state, calendar)

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
# The projection a live kernel reads host state through, or `nothing` for a
# kernel whose hazards cannot change during a run. A race compares successive
# projections to decide whether pending contacts need redrawing, so a kernel may
# depend on host state only through this record — the restriction `record_kernel`
# already relies on to reproduce a run's hazards from recorded records alone.
_kernel_projection(k) = nothing
_kernel_projection(k::PairKernel) = k.state
_kernel_projection(k::PairKernel{F, <:AbstractVector}) where {F} = nothing
_live_kernel(k) = _kernel_projection(k) !== nothing

# The projection a race has to watch for changes. Resolving a case writes only to
# that case's own record, and no contact already drawn depends on it: contacts to
# the case are settled, and its own contacts are drawn afterwards. So records that
# pending contacts depend on can move only through an intervention, and without
# one a live kernel races exactly as an ordinary kernel does.
function _watched_projection(kernel, interventions)
    return isempty(interventions) ? nothing : _kernel_projection(kernel)
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
        k.calendar
    )
end

# ── Calendar-scaled kernels ───────────────────────────────────────────
#
# The contact interval since `opening` under a `profile` hazard multiplied by a
# `Steps` schedule on the calendar-time axis. The multiplier is constant on
# each step, so the cumulative hazard over any interval splits into segments,
# each a scaled difference of the profile's cumulative hazard, and inverting a
# target log-survival walks the same segments. This keeps simulation and the
# likelihood exact wherever the profile's own hazard is.

struct _CalendarScaledKernel{P, S <: Steps, T <: Real}
    profile::P
    calendar::S
    opening::T
end

# The segment starting at time-since-opening `t_lo`: its multiplier and the
# time-since-opening its schedule step ends at (`Inf` for the last step).
function _calendar_segment(k::_CalendarScaledKernel, t_lo)
    calendar_t = k.opening + t_lo
    idx = searchsortedlast(k.calendar.breaks, calendar_t) + 1
    m = k.calendar.values[idx]
    seg_hi = idx <= length(k.calendar.breaks) ?
        k.calendar.breaks[idx] - k.opening : oftype(float(t_lo), Inf)
    return m, seg_hi
end

function SurvivalDistributions.cumhazard(k::_CalendarScaledKernel, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be non-negative"))
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

function SurvivalDistributions.loghazard(k::_CalendarScaledKernel, τ::Real)
    m, _ = _calendar_segment(k, τ)
    return loghazard(k.profile, τ) + log(m)
end

Distributions.logccdf(k::_CalendarScaledKernel, τ::Real) = -cumhazard(k, τ)

# Invert a target log-survival `lp` segment by segment: consume each step's
# cumulative hazard budget until the remaining budget is met inside the
# current step, then invert within the profile alone with the same machinery
# used for a constant rate multiplier.
function Distributions.invlogccdf(k::_CalendarScaledKernel, lp::Real)
    isfinite(lp) || return oftype(float(lp), Inf)
    lp <= 0 || throw(ArgumentError("invlogccdf needs a non-positive log-survival"))
    target = float(-lp)
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

Base.rand(rng::AbstractRNG, k::_CalendarScaledKernel) = _time_at_log_survival(k, log(rand(rng)))
Base.minimum(k::_CalendarScaledKernel) = minimum(k.profile)
Distributions.partype(k::_CalendarScaledKernel) = Distributions.partype(k.profile)
