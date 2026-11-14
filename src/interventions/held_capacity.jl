"""
    HeldCapacity(intervention; capacity, hold = Inf,
                 priority = default_capacity_priority, on_refusal = DropRefused())

Limit how many people can hold a resource at once: a place taken when an
action is admitted and given back later, such as a bed occupied for the length
of a quarantine. [`CapacityConstrained`](@ref) instead spends a budget per
period and does not return any of it.

# Examples
```julia
# Twenty quarantine places, each held for as long as the quarantine lasts
HeldCapacity(
    ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 14.0));
    capacity = 20, hold = 14.0
)

# Five places until day 30, twenty after
HeldCapacity(
    ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 14.0));
    capacity = Steps([30.0], [5.0, 20.0]), hold = 14.0
)

# Refused contacts wait for the first place to free up
HeldCapacity(
    ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 14.0));
    capacity = 20, hold = 14.0, on_refusal = DeferRefused()
)
```

# Arguments
- `capacity`: the number of places, at most this many held at any one time. A
  number, or a piecewise-constant calendar schedule such as [`Steps`](@ref) for
  places that open or close over the outbreak, read at each day as for the
  `budget_per_period` of [`CapacityConstrained`](@ref). A smooth schedule is
  refused because the days the capacity changes must be known.
- `hold`: how long a place is kept once taken, in days: a number, a
  `Distribution` or an `(rng, individual)` callable, as a delay is anywhere
  else. The default, `Inf`, keeps it until [`release_capacity!`](@ref) gives it
  back, which the `effect!` of an action can do by returning a further action
  dated when the stay ends. For an intervention that sets its own effect, such as
  [`Isolation`](@ref), a place is held for the isolation itself, and `hold`,
  when finite, ends it sooner.
- `priority`: as for [`CapacityConstrained`](@ref).
- `on_refusal`: what happens to an action no place is free for, a
  [`RefusalPolicy`](@ref). [`DeferRefused`](@ref) moves it to the first moment
  a place frees up; the default [`DropRefused`](@ref) discards it.

A place is held from the date of the admitted action. Each action is checked
against every stay that overlaps its own, whatever order the engine admits
actions in, and a varying capacity must hold throughout the new stay, from its
start to its end. When the capacity falls below the places already held, no stay
is ended: new admissions are refused until the places held drop below the new
capacity. [`DeferRefused`](@ref) moves an action to the first moment a place is
free for its whole stay, which a release or a rise in capacity can bring about. A person holds at most one place.
[`held_usage`](@ref) reports the places held. Like `CapacityConstrained`, the
wrapper splits its places by [`resource_key`](@ref EpiBranch.resource_key) and
composes with [`Scheduled`](@ref); the intervention must define
[`capacity_key`](@ref EpiBranch.capacity_key).
"""
struct HeldCapacity{I <: AbstractIntervention, C, H, F, P <: RefusalPolicy} <: _CapacityWrapper
    intervention::I
    capacity::C
    hold::H
    priority::F
    on_refusal::P
end

function HeldCapacity(
        intervention::AbstractIntervention;
        capacity, hold = Inf, priority = default_capacity_priority,
        on_refusal::RefusalPolicy = DropRefused()
    )
    _check_refusal_policy(on_refusal, intervention)
    return HeldCapacity(intervention, _prepared_capacity(capacity), hold, priority, on_refusal)
end

function _prepared_capacity(capacity::Real)
    capacity >= 0 || throw(ArgumentError("capacity must be non-negative, got $capacity"))
    return Float64(capacity)
end
# A schedule validates its own values when it is built. Its breaks must be
# known to check a stay across them, so a smooth one is refused.
function _prepared_capacity(schedule)
    hasmethod(calendar_multiplier, Tuple{typeof(schedule), Float64}) || throw(
        ArgumentError(
            "capacity must be a number or a calendar schedule such as `Steps`, " *
                "got a $(typeof(schedule))"
        )
    )
    calendar_shape(schedule) isa PiecewiseConstantCalendar || throw(
        ArgumentError(
            "capacity as a schedule must be piecewise constant, like `Steps`, " *
                "got a $(typeof(schedule))"
        )
    )
    return schedule
end

_capacity_at(capacity::Real, t) = capacity
_capacity_at(schedule, t) = calendar_multiplier(schedule, t)

# The days in `(s, e)` on which the capacity changes.
_capacity_breaks(::Real, s, e) = Float64[]
function _capacity_breaks(schedule, s, e)
    out = Float64[]
    t = next_calendar_break(schedule, s)
    while t < e
        push!(out, t)
        t = next_calendar_break(schedule, t)
    end
    return out
end

# Everyone's stay, kept in `state.scratch` under the capacity key. Wrappers
# that share a key share places, and a stay outlives the call that admitted it.
# Maps a person's id to `(resource, start, end)`.
const _HeldStays = Dict{Int, Tuple{Any, Float64, Float64}}
function _held_stays(state, key)
    return get!(_HeldStays, state.scratch, (:held_capacity, key))::_HeldStays
end

# The stays of everyone but `exclude` on one resource, as `(start, end)`.
function _stays_on(stays::_HeldStays, resource, exclude = nothing)
    out = Tuple{Float64, Float64}[]
    for (id, (res, a, b)) in stays
        id == exclude && continue
        res == resource && push!(out, (a, b))
    end
    return out
end

# Whether a stay over `[s, e)` keeps the resource within `capacity` throughout.
# Occupancy only rises where another stay starts and the capacity only falls at
# one of its breaks, so the tightest moments of the stay are its own start, the
# start of another stay inside it and a capacity break inside it.
function _fits_stay(capacity, stays, s, e)
    moments = Float64[s]
    for (a, _) in stays
        s < a < e && push!(moments, a)
    end
    append!(moments, _capacity_breaks(capacity, s, e))
    for m in moments
        held = count(((a, b),) -> a <= m < b, stays)
        held + 1 <= _capacity_at(capacity, m) || return false
    end
    return true
end

_next_break(::Real, t) = Inf
_next_break(schedule, t) = next_calendar_break(schedule, t)

# The earliest moment after `action.time` that fits a stay of `duration`. A
# place becomes free when a stay is released or the capacity rises, so only
# release times and capacity breaks are tried.
function refused_action(::DeferRefused, hc::HeldCapacity, usage, action)
    stays, duration = usage.stays, usage.duration
    times = Float64[b for (_, b) in stays if action.time < b && isfinite(b)]
    last_release = isempty(times) ? action.time : maximum(times)
    append!(times, _capacity_breaks(hc.capacity, action.time, nextfloat(last_release)))
    # After the last release only the capacity matters. Breaks are candidates
    # until one fits the whole stay, as a later fall can rule an earlier rise out.
    t = _next_break(hc.capacity, max(last_release, action.time))
    while isfinite(t)
        push!(times, t)
        _fits_stay(hc.capacity, stays, t, t + duration) && break
        t = _next_break(hc.capacity, t)
    end
    for t in sort!(unique(times))
        _fits_stay(hc.capacity, stays, t, t + duration) || continue
        return InterventionAction(action.individual, t, action.effect!, action.claim!)
    end
    return nothing
end

function _hold_duration(hc::HeldCapacity, ind, state)
    rng = stream(state.rng, ind.id, (:held_capacity, :hold))
    return _removal_duration(hc.hold, rng, ind, "`HeldCapacity`'s `hold`")
end

function _admit_actions!(hc::HeldCapacity, state, actions, admitter = hc)
    isempty(actions) && return nothing
    key = capacity_key(hc.intervention)
    stays = _held_stays(state, key)
    already_used = filter(a -> _capacity_used(get(a.individual.state, key, nothing)), actions)
    candidates = filter(
        a -> !_capacity_used(get(a.individual.state, key, nothing)) &&
            !haskey(stays, a.individual.id), actions
    )
    _admit_actions!(hc.intervention, state, already_used, admitter)
    isempty(candidates) && return nothing
    group_idxs, group_keys = _resource_groups(hc, state, [a.individual for a in candidates])
    for (idxs, resource) in zip(group_idxs, group_keys)
        group = candidates[idxs]
        order = sortperm([hc.priority(a.individual, state) for a in group]; alg = MergeSort)
        for i in order
            action = group[i]
            haskey(stays, action.individual.id) && continue
            duration = _hold_duration(hc, action.individual, state)
            others = _stays_on(stays, resource)
            if !_fits_stay(hc.capacity, others, action.time, action.time + duration)
                usage = (; stays = others, duration)
                action = refused_action(hc.on_refusal, hc, usage, action)
                action === nothing && continue
            end
            stays[action.individual.id] = (resource, action.time, action.time + duration)
            _admit_actions!(hc.intervention, state, [action], admitter)
        end
    end
    return nothing
end

"""
    release_capacity!(state, intervention, person, at)

Give back the place `person` holds under a [`HeldCapacity`](@ref) wrapping
`intervention` (or any intervention with the same
[`capacity_key`](@ref EpiBranch.capacity_key)) at day `at`, ending their stay
there. A person holding no place, or whose stay ends sooner, is left alone. A
stay released at or before its own start is withdrawn altogether.

Call it from the `effect!` of an action, which can return a further
[`InterventionAction`](@ref) dated when the stay ends to do so.
"""
function release_capacity!(state, intervention, person, at::Real)
    stays = _held_stays(state, capacity_key(intervention))
    stay = get(stays, person.id, nothing)
    stay === nothing && return nothing
    resource, a, b = stay
    at < b || return nothing
    if at <= a
        delete!(stays, person.id)
    else
        stays[person.id] = (resource, a, Float64(at))
    end
    return nothing
end

"""
    held_usage(hc::HeldCapacity, state::SimulationState, at, key = nothing)
        -> (held, capacity)

The number of places held at day `at` and the number there are. `key` selects
one budget when [`resource_key`](@ref EpiBranch.resource_key) splits
the intervention of `hc` into more than one; the default, `nothing`, is the single
one everyone shares.
"""
function held_usage(hc::HeldCapacity, state::SimulationState, at::Real, key = nothing)
    stays = _stays_on(_held_stays(state, capacity_key(hc.intervention)), key)
    return (held = count(((a, b),) -> a <= at < b, stays), capacity = _capacity_at(hc.capacity, at))
end

# ── Delegation of every other hook ────────────────────────────────────

# An intervention that sets its own effect, as `Isolation` does, holds a place
# for as long as that effect lasts; one that does not fit its place is undone,
# as `CapacityConstrained` does.
_ration_own_effect!(hc::HeldCapacity, ind, state) = _hold_or_reset!(hc, ind, state)

function _hold_or_reset!(hc::HeldCapacity, ind::Individual, state)
    key = capacity_key(hc.intervention)
    _capacity_used(get(ind.state, key, nothing)) || return nothing
    s = intervention_time(hc.intervention, ind)
    isfinite(s) || return nothing
    e = min(held_until(hc.intervention, ind), s + _hold_duration(hc, ind, state))
    stays = _held_stays(state, key)
    resource = resource_key(hc.intervention, ind, state)
    if _fits_stay(hc.capacity, _stays_on(stays, resource, ind.id), s, e)
        stays[ind.id] = (resource, s, e)
    else
        reset!(hc.intervention, ind)
    end
    return nothing
end

"""
    held_until(intervention, individual) -> Float64

For extension authors: the day the effect `intervention` set directly on
`individual` in `resolve_individual!` ends, which is when a
[`HeldCapacity`](@ref) gets the place back. The default, `Inf`, holds it until
[`release_capacity!`](@ref) or the `hold` given to the wrapper. Defined for
[`Isolation`](@ref), where the stay ends at the release time of the isolation.
"""
held_until(::AbstractIntervention, ind) = Inf
held_until(::Isolation, ind) = isolation_release_time(ind)
