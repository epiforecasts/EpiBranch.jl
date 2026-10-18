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
[`InterventionAction`](@ref EpiBranch.InterventionAction)s, so that each one
can be checked before it happens. An intervention may widen the candidates,
for example to everyone in their groups. Return `nothing`, the default, when
the intervention has no actions.

Both engines call it once per case, after every intervention's
[`trace_contacts!`](@ref) for that case, so a producer can read what tracing
wrote. The race, which runs every built-in model, passes the case and the
contacts it can still reach, or on a fixed-population model everyone not yet
settled for an intervention declaring [`continuous_actions`](@ref), and admits
only actions dated no earlier than the case's infection time. The generation engine, which runs a model written
against `generate_offspring` or `contacts_of`, passes the contacts the case
made and admits every action with a finite time; once a generation's cases
are settled, it also calls it again for each settled case in turn, with that
case alone as the candidate, for an intervention that declares
[`settles_generation_batch`](@ref), so a trigger that only becomes true on
settlement (such as a case's own lab confirmation) still reaches its group.
[`Scheduled`](@ref) and [`CapacityConstrained`](@ref) admit the actions
through their own gates.
"""
intervention_actions(::AbstractIntervention, state, candidates) = nothing
function intervention_actions(w::InterventionWrapper, state, candidates)
    return intervention_actions(w.intervention, state, candidates)
end
# Asking for the resource key first makes a wrapped intervention that has none,
# and so cannot be rationed, fail when its actions are first sought.
function intervention_actions(cc::CapacityConstrained, state, candidates)
    capacity_key(cc.intervention)
    return intervention_actions(cc.intervention, state, candidates)
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
Both check each action's own day: a schedule against its start and end, and
capacity against the budget of the period that day falls in. A person turned
down may be offered the action again later, with the same random draws. The
intervention itself leaves out actions already performed.
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
    return action.time >= s.start_time && _active_at(s, action.time, state)
end
function _admit_actions!(s::Scheduled, state, actions)
    selected = filter(a -> isfinite(a.time) && _action_in_schedule(s, a, state), actions)
    return _admit_actions!(s.intervention, state, selected)
end

# Each candidate action already names the day its dose would fall on, so it
# is checked against the budget of the period that day is in, not the clock
# of whichever case's call proposes it: a dose for day 12 can be committed
# before a case settled later asks for day 11, and either leaves the other's
# own day's budget untouched. A period exhausted by an earlier candidate
# therefore skips to the next-ranked one (`continue`, not `break`), which may
# fall in a different, still-open period.
function _admit_actions!(cc::CapacityConstrained, state, actions)
    key = capacity_key(cc.intervention)
    already_used = filter(a -> get(a.individual.state, key, false), actions)
    candidates = filter(a -> !get(a.individual.state, key, false), actions)
    _admit_actions!(cc.intervention, state, already_used)
    # The usage scan covers the whole population, which only a call with
    # someone to admit needs.
    isempty(candidates) && return nothing
    order = sortperm([cc.priority(a.individual, state) for a in candidates]; alg = MergeSort)
    times = _dose_times(cc, state)
    extra = Float64[]
    for i in order
        action = candidates[i]
        get(action.individual.state, key, false) && continue
        usage = _usage_at(cc, times, extra, action.time)
        usage.used + 1 <= usage.available || continue
        _admit_actions!(cc.intervention, state, [action])
        get(action.individual.state, key, false) && push!(extra, action.time)
    end
    return nothing
end

"""
    record_dose!(v::AbstractVaccination, person, at, rng)

Record one dose of `v` on `person` at time `at`, skipping a `person` already
recorded as vaccinated under `v`'s own `dose_label`, and reconsidering a
post-exposure abort ([`RingVaccination`](@ref)'s `post_exposure_efficacy`)
against the dose just given. The action protocol calls this through
[`dose_action`](@ref); call it directly to record a dose outside that
protocol, for example from a custom `trace_contacts!` or when
seeding a dose before a run starts.
"""
function record_dose!(v::AbstractVaccination, person, at, rng)
    get(person.state, _vaccinated_key(dose_label(v)), false) && return nothing
    _record_vaccination!(v, person, at, rng)
    _after_action_dose!(v, person, at, rng)
    return nothing
end

"""
    dose_action(v::AbstractVaccination, ind, time) -> InterventionAction

The [`InterventionAction`](@ref) that gives `ind` one dose of `v` at `time`
via [`record_dose!`](@ref), exactly as the built-in vaccinations do. A custom
vaccination's own [`intervention_actions`](@ref) method returns one of these
per candidate it will dose, so [`Scheduled`](@ref) and
[`CapacityConstrained`](@ref) admit it the same way they admit a
[`MassVaccination`](@ref) dose.
"""
function dose_action(v::AbstractVaccination, ind, time)
    effect! = (person, at, state) -> record_dose!(v, person, at, state.rng)
    return InterventionAction(ind, time, effect!)
end

_after_action_dose!(::AbstractVaccination, ind, time, rng) = nothing
function _after_action_dose!(rv::RingVaccination, ind, time, rng)
    _maybe_positive(rv.post_exposure_efficacy) && _abort_infection!(rv, ind, time, rng)
    return nothing
end

"""
    may_revise(intervention, prior_trigger, new_trigger) -> Bool

Whether a dose already accepted under `prior_trigger` may be moved to
`new_trigger`, found later. The default is `false`: an accepted dose keeps its
date. [`GroupVaccination`](@ref) and [`RingVaccination`](@ref) allow the move
when `new_trigger` is earlier, since in continuous-time models a group's
earliest eligible case, or a contact's earliest trace, may only be found
after a later one."""
may_revise(::AbstractIntervention, prior_trigger, new_trigger) = false
function may_revise(w::InterventionWrapper, prior_trigger, new_trigger)
    return may_revise(w.intervention, prior_trigger, new_trigger)
end
may_revise(::GroupVaccination, prior_trigger, new_trigger) = new_trigger < prior_trigger
may_revise(::RingVaccination, prior_trigger, new_trigger) = new_trigger < prior_trigger

_trigger_cache_key(v::AbstractVaccination) = (v, :trigger)

# Records the trigger a dose this action gave was timed from, so a later
# discovery can tell whether its own trigger is a genuine revision of *this*
# action's dose rather than a dose another vaccination (sharing the same
# `dose_label`) already gave. Used for both a person's first dose and a
# revision of one already given: the effect reads the person's own current
# state at admission time to tell which it is, rather than two separate
# action builders each hardcoding one.
#
# A revision moves `:vaccination_time` to the new trigger plus the person's
# own (already-drawn) delay, and shifts `:immunity_time` by the same amount,
# keeping the delay to immunity that was drawn for it. Efficacy and severity
# efficacy are untouched, since neither depends on when the dose was given,
# so this does not go through `_record_vaccination!`, which would resample
# them.
function _revisable_dose_action(v::AbstractVaccination, ind, trigger, time)
    label = dose_label(v)
    effect! = function (person, at, state)
        if get(person.state, _vaccinated_key(label), false)
            old_time = person.state[_vaccination_time_key(label)]
            person.state[_vaccination_time_key(label)] = at
            person.state[_immunity_time_key(label)] += at - old_time
        else
            record_dose!(v, person, at, state.rng)
        end
        _action_cache(person)[_trigger_cache_key(v)] = trigger
        return nothing
    end
    return InterventionAction(ind, time, effect!)
end

function intervention_actions(rv::RingVaccination, state, candidates)
    actions = InterventionAction[]
    label = dose_label(rv)
    for ind in candidates
        is_traced(ind) || continue
        trace_t = haskey(ind.state, :trace_time) ? ind.state[:trace_time] :
            min(_recorded_isolation_time(ind), _state_real(ind.state, :_traced_isolation_time, Inf))
        if get(ind.state, _vaccinated_key(label), false)
            if _maybe_positive(rv.post_exposure_efficacy)
                # Reconsider protection at this exposure without recording a new dose.
                effect! = (person, at, st) -> _abort_infection!(
                    rv, person, person.state[_vaccination_time_key(label)], st.rng
                )
                push!(actions, InterventionAction(ind, state.max_infection_time, effect!))
            end
            prior = get(_action_cache(ind), _trigger_cache_key(rv), nothing)
            if isfinite(trace_t) && prior !== nothing && may_revise(rv, prior, trace_t)
                delay = action_draw!(ind, (rv, :delay)) do
                    _sample_value(rv.dose_delay, state.rng, ind)
                end
                time = trace_t + delay
                # An earlier date must still follow the dose this one requires.
                _has_required_dose(rv, ind, time) &&
                    push!(actions, _revisable_dose_action(rv, ind, trace_t, time))
            end
            continue
        end
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
        accepted && push!(actions, _revisable_dose_action(rv, ind, trace_t, time))
    end
    return actions
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
                prior = get(_action_cache(ind), _trigger_cache_key(gv), nothing)
                (prior !== nothing && may_revise(gv, prior, trigger)) || continue
                delay = action_draw!(ind, (gv, :delay)) do
                    _sample_value(gv.dose_delay, state.rng, ind)
                end
                push!(actions, _revisable_dose_action(gv, ind, trigger, trigger + delay))
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
            push!(actions, _revisable_dose_action(gv, ind, trigger, trigger + delay))
        end
    end
    return actions
end

function intervention_actions(mv::MassVaccination, state, candidates)
    actions = InterventionAction[]
    label = dose_label(mv)
    for ind in candidates
        get(ind.state, _vaccinated_key(label), false) && continue
        get(ind.state, _coverage_declined_key(label), false) && continue
        time = action_draw!(ind, (mv, :time)) do
            _sample_value(mv.eligibility_time, state.rng, ind)
        end
        isfinite(time) || continue
        accepted = action_draw!(ind, (mv, :acceptance)) do
            _covers(mv.coverage, ind, state.rng)
        end
        if accepted
            push!(actions, dose_action(mv, ind, time))
        else
            ind.state[_coverage_declined_key(label)] = true
        end
    end
    return actions
end

"""
    continuous_actions(intervention) -> Bool

Whether this intervention can act once a case's infection is settled, rather
than only when it makes a contact. The race offers every intervention its
actions per settled case, and offers one declaring this, on a fixed-population
model, everyone not yet settled rather than only the contacts the case can
reach. The generation engine, by default (see [`settles_generation_batch`](@ref)),
also offers it the whole batch once a generation is settled. Return `true` only if proposing actions
needs no infection time of a contact not yet infected and never changes a
case already settled. The default is `false`.
"""
continuous_actions(::AbstractIntervention) = false
function continuous_actions(w::InterventionWrapper)
    return continuous_actions(w.intervention)
end
continuous_actions(::GroupVaccination) = true
# `_within_window` treats a pending, still-uninfected member as within any
# window, so a finite `eligibility_window` admits a dose for one here just as
# an infinite one would. `on_infection_settled!` reconsiders that dose once
# the member's own infection settles and its exposure is known, withdrawing
# it if the window turns out not to cover it, the same way it already
# reconsiders `post_exposure_efficacy`.
continuous_actions(::RingVaccination) = true

"""
    acts_on_population(intervention) -> Bool

Whether this intervention's actions belong to the whole population rather
than to a case's contacts or group. On the race,
[`intervention_actions`](@ref) is then called once, before any case settles,
with every member as a candidate, and for each contact a branching process
creates, instead of per settled case. Return `true`
only if proposing an action needs nothing a case's settlement would supply:
no infection time, no exposure, no settled state. The default is `false`.

[`MassVaccination`](@ref) declares it: each person's dosing time comes from
their own `eligibility_time`, not from who infected whom, so the whole
rollout can be decided upfront. A dose dated later in the simulation still
protects correctly, because the per-contact competing risk is checked against
the recorded dose time when a candidate infection is popped, not when it was
proposed.
"""
acts_on_population(::AbstractIntervention) = false
function acts_on_population(w::InterventionWrapper)
    return acts_on_population(w.intervention)
end
acts_on_population(::MassVaccination) = true

"""
    needs_contacts(intervention) -> Bool

Whether this intervention's actions need the model to name each case's
contacts. The race narrows `intervention_actions`'s candidates to a case and
the people it can still reach, through
[`supplies_contacts`](@ref EpiBranch.supplies_contacts); the
homogeneous and mixing models name none, so an intervention that needs them
is never offered them there. The default is `true`, matching
[`RingVaccination`](@ref), whose candidates come from walking the contact
graph. [`GroupVaccination`](@ref) declares `false`: its candidates are a
case's group, found through its `group_key`, not through any contact.
"""
needs_contacts(::AbstractIntervention) = true
function needs_contacts(w::InterventionWrapper)
    return needs_contacts(w.intervention)
end
needs_contacts(::GroupVaccination) = false

"""
    settles_generation_batch(intervention) -> Bool

Whether `_settle_generation!` should also offer `intervention`'s actions for
the whole settled batch, on top of the contacts each source is already
offered mid-generation ([`_case_hooks!`](@ref)). Defaults to
[`continuous_actions`](@ref). `RingVaccination` is the one exception: a
contact reaches it, pre-infection, through `_case_hooks!` already, which is
where `post_exposure_efficacy` needs to reconsider a dose, against an
exposure not yet complicated by that contact's own clinical resolution.
Offering it again from the settled batch would reconsider a dose after this
same round has already resolved the case's onset and isolation from it.
"""
settles_generation_batch(iv) = continuous_actions(iv)
function settles_generation_batch(w::InterventionWrapper)
    return settles_generation_batch(w.intervention)
end
settles_generation_batch(::RingVaccination) = false

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
is settled. An intervention declaring [`continuous_actions`](@ref) is offered
`current` and everyone not yet settled, and narrows that itself; any other is
offered `current` and the contacts it can still reach, those tracing reached
from it where there was tracing, as [`RingVaccination`](@ref) is.
[`GroupVaccination`](@ref) offers the members of `current`'s group instead.

`traced` is the set tracing reached, or `nothing` if there was no tracing. A
ring wider than one step reaches people who are not `current`'s direct
contacts, so a list built from `current`'s contacts alone would miss them.
"""
function _continuous_candidates(
        iv::AbstractIntervention, state, current, members, processed, contacts, pos,
        traced = nothing
    )
    continuous_actions(iv) && return _unsettled_candidates(
        members, state, current, processed, contacts, pos, traced
    )
    return _reached_candidates(state, current, members, processed, contacts, pos, traced)
end
function _continuous_candidates(
        ::RingVaccination, state, current, members, processed, contacts, pos,
        traced = nothing
    )
    return _reached_candidates(state, current, members, processed, contacts, pos, traced)
end

# `current` and everyone the race has not yet settled.
function _unsettled_candidates(members, state, current, processed, contacts, pos, traced)
    return [
        state.individuals[id]
            for (k, id) in enumerate(members)
            if !processed[k] || id == current.id
    ]
end

# `current` and the contacts it can still reach.
function _reached_candidates(state, current, members, processed, contacts, pos, traced)
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
        w::InterventionWrapper, state, current,
        members, processed, contacts, pos, traced = nothing
    )
    return _continuous_candidates(
        w.intervention, state, current, members, processed, contacts, pos, traced
    )
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
        contacts::C = nothing, pos = nothing, traced = nothing
    ) where {C}
    # Finalised cases have already generated proposals. A new action may
    # affect the current case and pending people, but must not revise earlier
    # cases — `_continuous_candidates` is what keeps each intervention within
    # that boundary while choosing its own, much smaller, set of people to
    # visit. The caller passes only the interventions that propose actions
    # (`_proposes_actions`).
    for iv in interventions
        # `needs_contacts` rather than `contacts !== nothing` alone, so an
        # intervention that needs no contacts (`GroupVaccination`) still
        # runs on a model that names none, even alongside one that does
        # need them.
        needs_contacts(iv) && contacts === nothing && continue
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
    # `current`'s own round of discovery is done: mark it settled only now,
    # after the loop above, so a policy consulting `is_settled` during this
    # same round (an admitted dose of `current`'s own being reconsidered by
    # a trigger `current` itself just supplied) still sees it as pending.
    current.state[:_settled] = true
    return nothing
end

# Whether an intervention proposes actions per settled case: it has an
# `intervention_actions` method of its own, found past any wrapper that only
# forwards it, and is not one offered the whole population at once instead.
function _proposes_actions(iv::AbstractIntervention)
    acts_on_population(iv) && return false
    return _discovers_actions(_hook_owner(intervention_actions, iv))
end

# Offer every intervention declaring `acts_on_population` its actions once,
# with every one of `members` as a candidate, before a network or household
# race processes any of them. An intervention with no `intervention_actions`
# method of its own contributes nothing, the same as `_apply_continuous_actions!`
# does for the per-case path.
function _apply_population_actions!(state, interventions, members)
    any(acts_on_population, interventions) || return nothing
    candidates = [state.individuals[id] for id in members]
    for iv in interventions
        acts_on_population(iv) || continue
        actions = intervention_actions(iv, state, candidates)
        actions === nothing && continue
        _admit_actions!(iv, state, actions)
    end
    return nothing
end
