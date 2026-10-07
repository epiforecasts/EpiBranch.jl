"""
    InterventionAction(individual, time, effect!)

A proposed action of an intervention (such as giving one dose) on
`individual` at day `time`, which [`Scheduled`](@ref) and
[`CapacityConstrained`](@ref) can accept or turn down before it happens.

For extension authors: `effect!(individual, time, state)` performs the
action once accepted; proposing an action must not record it. The effect may
record a future date, but acceptance happens on the current simulation clock.
An effect limited by capacity sets its intervention's
[`capacity_key`](@ref EpiBranch.capacity_key) only when it uses up capacity.
"""
struct InterventionAction{I, T, F}
    individual::I
    time::T
    effect!::F
end

"""
    intervention_actions(intervention, state, candidates)

The actions (such as doses) an intervention proposes for `candidates`, as
[`InterventionAction`](@ref EpiBranch.InterventionAction)s, so that
[`Scheduled`](@ref) and [`CapacityConstrained`](@ref) can check each one
before it happens. An intervention may widen the candidates, for example to
everyone in their groups.

For extension authors: return `nothing` (the default) to have the
intervention's own [`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!)
called instead. An intervention defining this method calls
[`apply_actions!`](@ref EpiBranch.apply_actions!) from its
`apply_post_transmission!`.
"""
intervention_actions(::AbstractIntervention, state, candidates) = nothing
function intervention_actions(w::InterventionWrapper, state, candidates)
    return intervention_actions(w.intervention, state, candidates)
end

"""
    _action_cache(individual)

For extension authors: the per-person store of random draws made when
proposing actions (see [`action_draw!`](@ref EpiBranch.action_draw!)). An
intervention can also keep its own records there, such as which proposal a
decision was made under. It does not appear in the line list.
"""
function _action_cache(individual)
    return get!(individual.state, :_intervention_actions) do
        Dict{Any, Any}()
    end
end

"""
    action_draw!(sample, individual, key)

Draw `sample()` once for this person and action `key`, and return the same
value whenever the action is proposed again, so a person who was turned down
(for example by a coverage draw) stays turned down. Use different keys for
different visits or doses. The stored draws do not appear in the line list.
"""
function action_draw!(sample, individual, key)
    return get!(sample, _action_cache(individual), key)
end

"""
    apply_actions!(intervention, state, candidates)

Propose the intervention's actions for `candidates` and perform those that
its [`Scheduled`](@ref) and [`CapacityConstrained`](@ref) settings accept.
Schedules check each action's own day; capacity uses the simulation clock. A
person turned down may be offered the action again later, with the same
random draws. The intervention itself leaves out actions already performed.
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

function _give_dose!(v::AbstractVaccination, person, at, rng)
    get(person.state, _vaccinated_key(dose_label(v)), false) && return nothing
    _record_vaccination!(v, person, at, rng)
    _after_action_dose!(v, person, at, rng)
    return nothing
end

function _dose_action(v, ind, time)
    effect! = (person, at, state) -> _give_dose!(v, person, at, state.rng)
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

"""
    may_revise(intervention, prior_trigger, new_trigger) -> Bool

Whether a dose already accepted under `prior_trigger` may be moved to
`new_trigger`, found later. The default is `false`: an accepted dose keeps its
date. [`GroupVaccination`](@ref) allows the move when `new_trigger` is
earlier, since in continuous-time models a group's earliest eligible case may
only be found after a later one."""
may_revise(::AbstractIntervention, prior_trigger, new_trigger) = false
function may_revise(w::InterventionWrapper, prior_trigger, new_trigger)
    return may_revise(w.intervention, prior_trigger, new_trigger)
end
may_revise(::GroupVaccination, prior_trigger, new_trigger) = new_trigger < prior_trigger

_group_trigger_cache_key(gv::GroupVaccination) = (gv, :trigger)

# Records the trigger a dose this action gave was timed from, so a later
# discovery can tell whether its own trigger is a genuine revision of *this*
# action's dose rather than a dose another vaccination (sharing the same
# `dose_label`) already gave. Used for both a member's first dose and a
# revision of one already given: the effect reads the member's own current
# state at admission time to tell which it is, rather than two separate
# action builders each hardcoding one.
#
# A revision moves `:vaccination_time` to the new trigger plus the member's
# own (already-drawn) delay, and shifts `:immunity_time` by the same amount,
# keeping the delay to immunity that was drawn for it. Efficacy and severity
# efficacy are untouched, since neither depends on when the dose was given,
# so this does not go through `_record_vaccination!`, which would resample
# them.
function _group_dose_action(gv::GroupVaccination, ind, trigger, time)
    label = dose_label(gv)
    effect! = function (person, at, state)
        if get(person.state, _vaccinated_key(label), false)
            old_time = person.state[_vaccination_time_key(label)]
            person.state[_vaccination_time_key(label)] = at
            person.state[_immunity_time_key(label)] += at - old_time
        else
            _give_dose!(gv, person, at, state.rng)
        end
        _action_cache(person)[_group_trigger_cache_key(gv)] = trigger
        return nothing
    end
    return InterventionAction(ind, time, effect!)
end

function intervention_actions(gv::GroupVaccination, state, candidates)
    actions = InterventionAction[]
    # `is_settled` only ever becomes `true` on the continuous-time race this
    # call may be part of; the ordinary generation engine never sets it, so
    # it is always `false` there. `allowed` is the candidate-based guard
    # `is_settled` replaced for the revise branch below, kept here as well to
    # stop a member the generation engine already finalised in an earlier
    # generation (not among this call's own `candidates`) from having its
    # dose moved.
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
                # A settled member's dose keeps its date; a member still
                # pending in a continuous-time race (or the member currently
                # being settled, mid-round) can still be reached by a
                # genuinely earlier trigger. `ind.id in allowed` keeps this
                # within the members this call actually received.
                (ind.id in allowed && !is_settled(state, ind)) || continue
                prior = get(_action_cache(ind), _group_trigger_cache_key(gv), nothing)
                (prior !== nothing && may_revise(gv, prior, trigger)) || continue
                delay = action_draw!(ind, (gv, :delay)) do
                    _sample_value(gv.dose_delay, state.rng, ind)
                end
                push!(actions, _group_dose_action(gv, ind, trigger, trigger + delay))
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

Whether this intervention can act in network and household models, by
proposing its actions after each case's infection is settled. Return `true`
only if proposing actions needs no infection time of a contact not yet
infected and never changes a case already settled. The default is `false`.
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

The people to offer `intervention`'s actions to once `current`'s infection
is settled in a network or household model. The default is `current` and
everyone not yet settled, which suits any intervention. [`RingVaccination`](@ref)
and [`GroupVaccination`](@ref) narrow it to the people tracing reached from
`current` or the members of `current`'s group.

`traced` is the set tracing reached, or `nothing` if there was no tracing. A
ring wider than one step reaches people who are not `current`'s direct
contacts, so a list built from `current`'s contacts alone would miss them.
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
    if any(continuous_actions, interventions)
        # Finalised cases have already generated proposals. A new action may
        # affect the current case and pending people, but must not revise
        # earlier cases — `_continuous_candidates` is what keeps each
        # intervention within that boundary while choosing its own, much
        # smaller, set of people to visit.
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
    end
    # `current`'s own round of discovery is done: mark it settled only now,
    # after the loop above, so a policy consulting `is_settled` during this
    # same round (an admitted dose of `current`'s own being reconsidered by
    # a trigger `current` itself just supplied) still sees it as pending.
    current.state[:_settled] = true
    return nothing
end
