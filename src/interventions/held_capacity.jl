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

# Refused contacts wait for the first place to free up
HeldCapacity(
    ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 14.0));
    capacity = 20, hold = 14.0, on_refusal = DeferRefused()
)
```

# Arguments
- `capacity`: the number of places, at most this many held at any one time.
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
actions in. A person already holding a place is not given a second.
[`held_usage`](@ref) reports the places held. Like `CapacityConstrained`, the
wrapper splits its places by [`resource_key`](@ref EpiBranch.resource_key) and
composes with [`Scheduled`](@ref); the intervention must define
[`capacity_key`](@ref EpiBranch.capacity_key).
"""
struct HeldCapacity{I <: AbstractIntervention, H, F, P <: RefusalPolicy} <: InterventionWrapper
    intervention::I
    capacity::Float64
    hold::H
    priority::F
    on_refusal::P
end

function HeldCapacity(
        intervention::AbstractIntervention;
        capacity::Real, hold = Inf, priority = default_capacity_priority,
        on_refusal::RefusalPolicy = DropRefused()
    )
    capacity >= 0 || throw(ArgumentError("capacity must be non-negative, got $capacity"))
    _check_refusal_policy(on_refusal, intervention)
    return HeldCapacity(intervention, Float64(capacity), hold, priority, on_refusal)
end

# Everyone's stay, kept in `state.scratch` under the capacity key so two
# wrappers sharing a key share places, and so a stay outlives the call that
# admitted it. Maps a person's id to `(resource, start, end)`.
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
# The busiest moment of the stay is its own start or the start of another stay
# inside it, so only those are counted.
function _fits_stay(capacity, stays, s, e)
    moments = Float64[s]
    for (a, _) in stays
        s < a < e && push!(moments, a)
    end
    for m in moments
        held = count(((a, b),) -> a <= m < b, stays)
        held + 1 <= capacity || return false
    end
    return true
end

# The earliest moment after `action.time` that fits a stay of `duration`. Only
# the release of a stay frees a place, so only release times are tried.
function refused_action(::DeferRefused, hc::HeldCapacity, usage, action)
    stays, duration = usage.stays, usage.duration
    for t in sort!(unique([b for (_, b) in stays if action.time < b && isfinite(b)]))
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
`hc`'s intervention into more than one; the default, `nothing`, is the single
one everyone shares.
"""
function held_usage(hc::HeldCapacity, state::SimulationState, at::Real, key = nothing)
    stays = _stays_on(_held_stays(state, capacity_key(hc.intervention)), key)
    return (held = count(((a, b),) -> a <= at < b, stays), capacity = hc.capacity)
end

# ── Delegation of every other hook ────────────────────────────────────

function intervention_actions(hc::HeldCapacity, state, candidates)
    capacity_key(hc.intervention)
    return intervention_actions(hc.intervention, state, candidates)
end
_discovers_actions(hc::HeldCapacity) =
    _discovers_actions(_hook_owner(intervention_actions, hc.intervention))
function _proposes_actions(hc::HeldCapacity)
    acts_on_population(hc) && return false
    return !_resolves_under_capacity(hc.intervention)
end

# An intervention that sets its own effect, as `Isolation` does, holds a place
# for as long as that effect lasts; one that does not fit its place is undone,
# as `CapacityConstrained` does.
function resolve_individual!(hc::HeldCapacity, ind, state)
    resolve_individual!(hc.intervention, ind, state)
    _resolves_under_capacity(hc.intervention) && _hold_or_reset!(hc, ind, state)
    return nothing
end

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
[`Isolation`](@ref), whose isolation ends at its release time.
"""
held_until(::AbstractIntervention, ind) = Inf
held_until(w::InterventionWrapper, ind) = held_until(w.intervention, ind)
held_until(::Isolation, ind) = isolation_release_time(ind)

traces_contacts(hc::HeldCapacity) = traces_contacts(hc.intervention)
function trace_contacts!(hc::HeldCapacity, state, infector, contacts, not_before = nothing)
    key = (:trace_admitter, _innermost(hc))
    previous = get(state.scratch, key, nothing)
    state.scratch[key] = hc
    try
        if not_before === nothing
            trace_contacts!(hc.intervention, state, infector, contacts)
        else
            trace_contacts!(hc.intervention, state, infector, contacts, not_before)
        end
    finally
        previous === nothing ? delete!(state.scratch, key) : (state.scratch[key] = previous)
    end
    return nothing
end

removal_gap_host_times(w::HeldCapacity) = removal_gap_host_times(w.intervention)
removal_block_probability(w::HeldCapacity) = removal_block_probability(w.intervention)
