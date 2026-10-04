"""
    InterventionAction(individual, time, effect!)

A candidate intervention action on `individual` at simulation `time`.
`effect!(individual, time, state)` records the admitted action. Discovery must
not record delivery. Effects may record future dates; admission occurs at the
current simulation clock. A resource-constrained effect sets its intervention's
`capacity_key` flag only when it consumes that resource.
"""
struct InterventionAction{I, T, F}
    individual::I
    time::T
    effect!::F
end

"""
    intervention_actions(intervention, state, candidates)

Discover candidate [`InterventionAction`](@ref)s before scheduling or admission.
An intervention may expand the incoming candidates, for example to their whole
groups. Return `nothing` to use the legacy batch hook. An external producer can
implement this method and call `apply_actions!` from its batch hook.
"""
intervention_actions(::AbstractIntervention, state, candidates) = nothing
function intervention_actions(w::InterventionWrapper, state, candidates)
    return intervention_actions(w.intervention, state, candidates)
end

"""
    _action_cache(individual)

The per-individual cache action discovery draws into (see [`action_draw!`](@ref)).
Exposed so an intervention can also record bookkeeping of its own, such as which
discovery a decision was made under. The cache is omitted from line-list output.
"""
function _action_cache(individual)
    return get!(individual.state, :_intervention_actions) do
        Dict{Any, Any}()
    end
end

"""
    action_draw!(sample, individual, key)

Cache `sample()` for one individual's action identity `key`. Repeated discovery
returns the same value, including a rejected acceptance draw. Use distinct keys
for distinct visits or dose labels. The cache belongs to the simulation's
individual and is omitted from line-list output.
"""
function action_draw!(sample, individual, key)
    return get!(sample, _action_cache(individual), key)
end

"""
    apply_actions!(intervention, state, candidates)

Discover and admit actions through the intervention's wrappers. Schedules use
candidate action times; capacity uses the simulation clock at admission. A
candidate rejected by a wrapper may be considered on later discovery, using
its cached draws. Completed actions are excluded by their producer.
"""
function apply_actions!(iv, state, candidates)
    actions = intervention_actions(iv, state, candidates)
    actions === nothing && throw(ArgumentError("$(typeof(iv)) does not declare actions"))
    _admit_actions!(iv, state, actions)
    return nothing
end

function _admit_actions!(iv::AbstractIntervention, state, actions)
    for action in actions
        isfinite(action.time) || continue
        action.effect!(action.individual, action.time, state)
    end
    return nothing
end

function _action_in_schedule(s::Scheduled, action, state)
    # Predicates see the proposed action clock; restore the admission clock even
    # if a user predicate throws. Each simulation owns its state.
    previous_time = state.max_infection_time
    try
        state.max_infection_time = action.time
        return action.time >= s.start_time && is_active(s, state)
    finally
        state.max_infection_time = previous_time
    end
end
function _admit_actions!(s::Scheduled, state, actions)
    selected = filter(a -> isfinite(a.time) && _action_in_schedule(s, a, state), actions)
    return _admit_actions!(s.intervention, state, selected)
end

function _admit_actions!(cc::CapacityConstrained, state, actions)
    key = capacity_key(cc.intervention)
    already_used = filter(a -> get(a.individual.state, key, false), actions)
    candidates = filter(a -> !get(a.individual.state, key, false), actions)
    _admit_actions!(cc.intervention, state, already_used)
    order = sortperm([cc.priority(a.individual, state) for a in candidates]; alg = MergeSort)
    usage = _capacity_usage(cc, state)
    used = usage.used
    admission_key = _capacity_admission_key(key)
    for i in order
        used + 1 <= usage.available || break
        action = candidates[i]
        get(action.individual.state, key, false) && continue
        _admit_actions!(cc.intervention, state, [action])
        if get(action.individual.state, key, false)
            action.individual.state[admission_key] = state.max_infection_time
            used += 1
        end
    end
    return nothing
end

function _dose_action(v, ind, time)
    effect! = function (person, at, state)
        get(person.state, _vaccinated_key(dose_label(v)), false) && return nothing
        _record_vaccination!(v, person, at, state.rng)
        _after_action_dose!(v, person, at, state.rng)
        return nothing
    end
    return InterventionAction(ind, time, effect!)
end

_after_action_dose!(::AbstractVaccination, ind, time, rng) = nothing
function _after_action_dose!(rv::RingVaccination, ind, time, rng)
    _maybe_positive(rv.post_exposure_efficacy) && _abort_infection!(rv, ind, time, rng)
    return nothing
end

function intervention_actions(rv::RingVaccination, state, candidates)
    actions = InterventionAction[]
    label = dose_label(rv)
    for ind in candidates
        is_traced(ind) || continue
        if get(ind.state, _vaccinated_key(label), false)
            if _maybe_positive(rv.post_exposure_efficacy)
                # Reconsider protection at this exposure without recording a new dose.
                effect! = (person, at, st) -> _abort_infection!(
                    rv, person, person.state[_vaccination_time_key(label)], st.rng
                )
                push!(actions, InterventionAction(ind, state.max_infection_time, effect!))
            end
            continue
        end
        trace_t = haskey(ind.state, :trace_time) ? ind.state[:trace_time] :
            min(_recorded_isolation_time(ind), get(ind.state, :_traced_isolation_time, Inf))
        isfinite(trace_t) || continue
        delay = action_draw!(ind, (rv, :delay)) do
            _sample_value(rv.dose_delay, state.rng, ind)
        end
        time = trace_t + delay
        isfinite(time) || continue
        _has_required_dose(rv, ind, time) || continue
        window = action_draw!(ind, (rv, :eligibility_window)) do
            _sample_value(rv.eligibility_window, state.rng, ind)
        end
        _within_window(window, ind, time) || continue
        accepted = action_draw!(ind, (rv, :acceptance)) do
            _covers(rv.coverage, ind, state.rng)
        end
        accepted && push!(actions, _dose_action(rv, ind, time))
    end
    return actions
end

_group_trigger_cache_key(gv::GroupVaccination) = (gv, :trigger)

# Records the trigger a dose this action gave was timed from, so a later
# discovery can tell whether its own trigger is a genuine revision of *this*
# action's dose rather than a dose another vaccination (sharing the same
# `dose_label`) already gave.
function _group_dose_action(gv::GroupVaccination, ind, trigger, time)
    base = _dose_action(gv, ind, time)
    effect! = function (person, at, state)
        base.effect!(person, at, state)
        _action_cache(person)[_group_trigger_cache_key(gv)] = trigger
        return nothing
    end
    return InterventionAction(ind, time, effect!)
end

# Moves an already-given dose from this action to an earlier trigger, on a
# member the race has not yet settled: `:vaccination_time` moves to the new
# trigger plus the member's own (already-drawn) delay, and `:immunity_time`
# shifts by the same amount, keeping the delay to immunity that was drawn for
# it. Efficacy and severity efficacy are untouched, since neither depends on
# when the dose was given.
function _revise_group_dose_action(gv::GroupVaccination, ind, trigger, time)
    label = dose_label(gv)
    effect! = function (person, at, state)
        old_time = person.state[_vaccination_time_key(label)]
        person.state[_vaccination_time_key(label)] = at
        person.state[_immunity_time_key(label)] += at - old_time
        _action_cache(person)[_group_trigger_cache_key(gv)] = trigger
        return nothing
    end
    return InterventionAction(ind, time, effect!)
end

function intervention_actions(gv::GroupVaccination, state, candidates)
    actions = InterventionAction[]
    allowed = Set(ind.id for ind in candidates)
    groups_here = Set(
        get(ind.state, gv.group_key, nothing)
            for ind in candidates
            if haskey(ind.state, gv.group_key)
    )
    for group in groups_here
        trigger = _group_trigger_time(gv, state, group)
        isfinite(trigger) || continue
        for id in _group_members(state, gv.group_key, group)
            ind = state.individuals[id]
            if get(ind.state, _vaccinated_key(dose_label(gv)), false)
                # A member not yet settled in a continuous-time race can still
                # be reached by an earlier trigger this same action gave it;
                # a settled member's dose, and one another vaccination gave,
                # keep their date. `intervention_actions` is called before the
                # candidate restriction the continuous-time race applies, so
                # `allowed` mirrors it here.
                ind.id in allowed || continue
                prior = get(_action_cache(ind), _group_trigger_cache_key(gv), nothing)
                (prior === nothing || trigger >= prior) && continue
                delay = action_draw!(ind, (gv, :delay)) do
                    _sample_value(gv.dose_delay, state.rng, ind)
                end
                push!(actions, _revise_group_dose_action(gv, ind, trigger, trigger + delay))
                continue
            end
            get(ind.state, _coverage_declined_key(dose_label(gv)), false) && continue
            accepted = action_draw!(ind, (gv, :acceptance)) do
                _covers(gv.coverage, ind, state.rng)
            end
            if !accepted
                ind.state[_coverage_declined_key(dose_label(gv))] = true
                continue
            end
            delay = action_draw!(ind, (gv, :delay)) do
                _sample_value(gv.dose_delay, state.rng, ind)
            end
            push!(actions, _group_dose_action(gv, ind, trigger, trigger + delay))
        end
    end
    return actions
end

function intervention_actions(mv::MassVaccination, state, candidates)
    actions = InterventionAction[]
    for ind in candidates
        get(ind.state, _vaccinated_key(dose_label(mv)), false) && continue
        time = action_draw!(ind, (mv, :time)) do
            _sample_value(mv.eligibility_time, state.rng, ind)
        end
        isfinite(time) && push!(actions, _dose_action(mv, ind, time))
    end
    return actions
end

"""
    continuous_actions(intervention) -> Bool

Whether candidate actions can be discovered after a case is finalised on a
network or household race. External producers opt in only when discovery does
not require a pending contact's unknown infection time or revise an already
finalised case. The default is `false`.
"""
continuous_actions(::AbstractIntervention) = false
function continuous_actions(w::InterventionWrapper)
    return continuous_actions(w.intervention)
end
continuous_actions(::GroupVaccination) = true
# A finite `eligibility_window` needs the candidate's own exposure to check
# against, which a pending, still-uninfected member of a continuous-time race
# does not have yet (see `_within_window`). `post_exposure_efficacy` has no
# such requirement: `on_infection_settled!` reconsiders it once that
# member's own infection settles, against the exposure the race then knows.
function continuous_actions(rv::RingVaccination)
    return rv.eligibility_window isa Real && rv.eligibility_window == Inf
end

# Whether `id` belongs to this race and the race has not finalised it yet, so
# that an action may still reach it. `pos` is the race's id-to-index map into
# `members`; a caller with no map walks the members instead, which is what the
# map exists to avoid on the race's own path.
function _pending(id, members, processed, pos)
    if pos === nothing
        k = findfirst(==(id), members)
        return k !== nothing && !processed[k]
    end
    k = get(pos, id, 0)
    return k != 0 && !processed[k]
end

"""
    _continuous_candidates(intervention, state, current, members, processed, contacts, pos, traced)

The individuals to offer `intervention`'s [`intervention_actions`](@ref) once
`current` has just settled. The default is every other still-pending member
together with `current` itself — safe for any intervention, but a full scan
of the population on every settled case. [`RingVaccination`](@ref) and
[`GroupVaccination`](@ref) need far less: the members the tracing walk
reached from `current`, or the members of `current`'s own group found through
the group-to-members index (see `EpiBranch._group_members`), so they override
this with a candidate list bounded by ring or group size rather than
population size.

`traced` is the set the race's walk reached, or `nothing` where no walk ran. A
ring wider than one hop reaches people `current` does not neighbour, so a
candidate list built from `contacts(current.id, state)` alone would leave them
out.
"""
function _continuous_candidates(
        ::AbstractIntervention, state, current, members, processed, contacts, pos,
        traced = nothing
    )
    return [
        state.individuals[id]
            for (k, id) in enumerate(members)
            if !processed[k] || id == current.id
    ]
end
function _continuous_candidates(
        w::InterventionWrapper, state, current,
        members, processed, contacts, pos, traced = nothing
    )
    return _continuous_candidates(
        w.intervention, state, current, members, processed, contacts, pos, traced
    )
end

function _continuous_candidates(
        ::RingVaccination, state, current, members, processed, contacts, pos,
        traced = nothing
    )
    candidates = [current]
    # The walk's own reach where there was one: a ring of depth above 1 grows
    # past `current`'s neighbours, and those members are the ones it traced.
    if traced === nothing
        contacts === nothing && return candidates
        traced = (c isa Tuple ? c[1] : c for c in contacts(current.id, state))
    end
    for cid in traced
        _pending(cid, members, processed, pos) || continue
        push!(candidates, state.individuals[cid])
    end
    return candidates
end

function _continuous_candidates(
        gv::GroupVaccination, state, current, members, processed, contacts, pos,
        traced = nothing
    )
    candidates = [current]
    group = get(current.state, gv.group_key, nothing)
    group === nothing && return candidates
    for id in _group_members(state, gv.group_key, group)
        id == current.id && continue
        _pending(id, members, processed, pos) || continue
        push!(candidates, state.individuals[id])
    end
    return candidates
end

function _apply_continuous_actions!(
        state, current, interventions, members, processed,
        contacts = nothing, pos = nothing, traced = nothing
    )
    any(continuous_actions, interventions) || return nothing
    # Finalised cases have already generated proposals. A new action may affect
    # the current case and pending people, but must not revise earlier cases —
    # `_continuous_candidates` is what keeps each intervention within that
    # boundary while choosing its own, much smaller, set of people to visit.
    for iv in interventions
        continuous_actions(iv) || continue
        candidates = _continuous_candidates(
            iv, state, current, members, processed, contacts, pos, traced
        )
        allowed = Set(ind.id for ind in candidates)
        actions = intervention_actions(iv, state, candidates)
        actions === nothing && continue
        selected = filter(
            a -> a.individual.id in allowed &&
                a.time >= state.max_infection_time, actions
        )
        _admit_actions!(iv, state, selected)
    end
    return nothing
end
