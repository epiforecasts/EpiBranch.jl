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
function intervention_actions(w::Union{Scheduled, CapacityConstrained}, state, candidates)
    intervention_actions(w.intervention, state, candidates)
end

"""
    action_draw!(sample, individual, key)

Cache `sample()` for one individual's action identity `key`. Repeated discovery
returns the same value, including a rejected acceptance draw. Use distinct keys
for distinct visits or dose labels. The cache belongs to the simulation's
individual and is omitted from line-list output.
"""
function action_draw!(sample, individual, key)
    cache = get!(individual.state, :_intervention_actions) do
        Dict{Any, Any}()
    end
    return get!(sample, cache, key)
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
    _admit_actions!(s.intervention, state, selected)
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
                    rv, person, person.state[_vaccination_time_key(label)], st.rng)
                push!(actions, InterventionAction(ind, state.max_infection_time, effect!))
            end
            continue
        end
        trace_t = haskey(ind.state, :trace_time) ? ind.state[:trace_time] :
                  min(isolation_time(ind), get(ind.state, :traced_isolation_time, Inf))
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

function intervention_actions(gv::GroupVaccination, state, candidates)
    actions = InterventionAction[]
    groups_here = Set(get(ind.state, gv.group_key, nothing)
    for ind in candidates
    if haskey(ind.state, gv.group_key))
    for group in groups_here
        trigger = _group_trigger_time(gv, state, group)
        isfinite(trigger) || continue
        for ind in state.individuals
            get(ind.state, gv.group_key, nothing) == group || continue
            get(ind.state, _vaccinated_key(dose_label(gv)), false) && continue
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
            push!(actions, _dose_action(gv, ind, trigger + delay))
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
function continuous_actions(w::Union{Scheduled, CapacityConstrained})
    continuous_actions(w.intervention)
end
continuous_actions(::GroupVaccination) = true
function continuous_actions(rv::RingVaccination)
    rv.eligibility_window isa Real &&
        rv.eligibility_window == Inf && rv.post_exposure_efficacy isa Real &&
        rv.post_exposure_efficacy == 0
end

function _apply_continuous_actions!(state, current, interventions, members, processed)
    any(continuous_actions, interventions) || return nothing
    # Finalised cases have already generated proposals. A new action may affect
    # the current case and pending people, but must not revise earlier cases.
    candidates = [state.individuals[id]
                  for (k, id) in enumerate(members)
                  if !processed[k] || id == current.id]
    allowed = Set(ind.id for ind in candidates)
    for iv in interventions
        continuous_actions(iv) || continue
        actions = intervention_actions(iv, state, candidates)
        actions === nothing && continue
        selected = filter(
            a -> a.individual.id in allowed &&
                 a.time >= state.max_infection_time, actions)
        _admit_actions!(iv, state, selected)
    end
    return nothing
end
