"""
    CapacityConstrained(intervention; budget_per_period, period=Inf,
                        carry_over=true, priority=default_capacity_priority)

Limit admissions to an intervention using a shared resource budget. Ring,
group and mass vaccination expose their candidate actions before admission;
group candidates include every eligible member of a triggered group. External
producers implement `intervention_actions` and the `capacity_key` protocol.
Legacy batch interventions still receive admitted candidates one at a time.

Candidates are ordered by `priority(individual, state)`, lower first, with
stable ties. The default uses trace time; candidates without one retain their
incoming order. Priority is evaluated once per candidate. A rejected action
uses no budget, and a completed action is not charged again.

`budget_per_period` becomes available every `period` units on the admission
clock, `state.max_infection_time`. With `carry_over=true`, unused allowances
accumulate. Otherwise each period has a separate allowance. `period=Inf` is a
single lifetime budget. Fractional remaining capacity cannot admit a whole
action.

This budget counts admissions. An admitted action can have a future delivery
date, so it may be recorded in a different calendar period. Usage from another
intervention with the same `capacity_key` counts against the shared budget;
when it has no admission stamp, its delivery time determines the period.
Use distinct dose labels for independent budgets. [`capacity_usage`](@ref)
reports the used and available allowances.

[`Scheduled`](@ref) and this wrapper compose in either order. For action
producers, the schedule tests each candidate's action time while the budget
uses the admission clock. Candidates denied admission are not queued, but
later discovery may offer them again with their cached draws.

Network and household races execute supported ring and group actions through
this protocol. Mass vaccination and legacy batch-only interventions remain
unsupported there. Ring delivery on these races requires an infinite eligibility
window and zero post-exposure efficacy; pending infection times are unknown.
The homogeneous pool has no contact-tracing action path.

```julia
CapacityConstrained(RingVaccination(efficacy = 0.8);
    budget_per_period = 5.0, period = 1.0)
CapacityConstrained(GroupVaccination(efficacy = 0.8);
    budget_per_period = 200.0)
```
"""
struct CapacityConstrained{I <: AbstractIntervention, F} <: InterventionWrapper
    intervention::I
    budget_per_period::Float64
    period::Float64
    carry_over::Bool
    priority::F
end

function CapacityConstrained(intervention::AbstractIntervention;
        budget_per_period::Real, period::Real = Inf, carry_over::Bool = true,
        priority = default_capacity_priority)
    budget_per_period >= 0 || throw(ArgumentError(
        "budget_per_period must be non-negative, got $budget_per_period"))
    period > 0 || throw(ArgumentError("period must be positive, got $period"))
    return CapacityConstrained(
        intervention, Float64(budget_per_period), Float64(period), carry_over, priority)
end

"""
    default_capacity_priority(individual, state) -> Float64

Default [`CapacityConstrained`](@ref) ordering: first-come-first-served by
`:trace_time`. A candidate with no recorded trace time (not traced, for a
tracing-driven intervention) sorts last, so it is admitted only once every
timed candidate in the same call has been.
"""
function default_capacity_priority(individual, state)
    t = get(individual.state, :trace_time, NaN)
    isnan(t) ? Inf : t
end

"""
    capacity_key(intervention) -> Symbol

The `Individual.state` flag [`CapacityConstrained`](@ref) counts to measure
how much of `intervention`'s shared resource has been used so far. Defined
for [`RingVaccination`](@ref) and [`MassVaccination`](@ref) (the dose flag
for their `dose_label`). Define this — together with
[`capacity_time_key`](@ref) unless `carry_over = true` is always used — for
a custom intervention to make it capacity-constrained the same way.

[`GroupVaccination`](@ref) uses the same dose keys. Its action discovery expands
a triggered group before the wrapper admits individual members.
"""
function capacity_key(iv::AbstractIntervention)
    throw(ArgumentError(
        "CapacityConstrained needs a `capacity_key` method for $(typeof(iv)) " *
        "to know which state key measures how much of its resource has been " *
        "used. See the Extending guide."))
end

"""
    capacity_time_key(intervention) -> Symbol

The `Individual.state` key holding *when* [`capacity_key`](@ref) was set,
read by [`CapacityConstrained`](@ref) when `carry_over = false` to place
usage it did not itself admit — a dose given by some other intervention
sharing the same [`capacity_key`](@ref) — in a period.
"""
function capacity_time_key(iv::AbstractIntervention)
    throw(ArgumentError(
        "CapacityConstrained needs a `capacity_time_key` method for " *
        "$(typeof(iv)) to measure usage within a period (carry_over = false). " *
        "See the Extending guide."))
end

# Read through any wrapper, so `CapacityConstrained` composes with
# `Scheduled` in either order.
capacity_key(w::InterventionWrapper) = capacity_key(w.intervention)
capacity_time_key(w::InterventionWrapper) = capacity_time_key(w.intervention)
capacity_key(v::RingVaccination) = _vaccinated_key(dose_label(v))
capacity_time_key(v::RingVaccination) = _vaccination_time_key(dose_label(v))
capacity_key(v::MassVaccination) = _vaccinated_key(dose_label(v))
capacity_time_key(v::MassVaccination) = _vaccination_time_key(dose_label(v))

capacity_key(v::GroupVaccination) = _vaccinated_key(dose_label(v))
capacity_time_key(v::GroupVaccination) = _vaccination_time_key(dose_label(v))

# The `Individual.state` key on which `CapacityConstrained` stamps the time
# of the call that admitted an individual, so a later call in the same period
# sees the budget that call spent. Derived from `capacity_key`, so two
# wrappers rationing different resources keep separate stamps.
_capacity_admission_key(key::Symbol) = Symbol("capacity_admission_time_", key)

# Doses (or whatever `cc` rations) used and available, in the scope
# `carry_over` implies, at `state`'s current point in time. Reused by the
# admission decision and by `capacity_usage`.
#
# With `carry_over = false`, usage is placed in a period by the time of the
# call that admitted it, since a dose is dated later than its admission and
# often falls outside the period whose budget paid for it. Usage this wrapper
# never admitted, which any other intervention writing the same
# `capacity_key` also draws on, has no such stamp and is placed by
# `capacity_time_key` instead.
function _capacity_usage(cc::CapacityConstrained, state)
    key = capacity_key(cc.intervention)
    t = state.max_infection_time
    if cc.carry_over
        n_periods = isinf(cc.period) ? 1.0 : floor(t / cc.period) + 1.0
        available = n_periods * cc.budget_per_period
        used = count(ind -> get(ind.state, key, false), state.individuals)
    else
        time_key = capacity_time_key(cc.intervention)
        admission_key = _capacity_admission_key(key)
        period_start = isinf(cc.period) ? 0.0 : floor(t / cc.period) * cc.period
        available = cc.budget_per_period
        used = count(state.individuals) do ind
            get(ind.state, key, false) || return false
            used_at = get(ind.state, admission_key, nothing)
            isnothing(used_at) && (used_at = get(ind.state, time_key, -Inf))
            return used_at >= period_start
        end
    end
    return (used = used, available = available)
end

# A candidate already carrying `capacity_key` (dosed in an earlier call, now
# re-exposed) does not draw on the budget when it is handed to the wrapped
# intervention again — `RingVaccination`/`MassVaccination` only redraw a
# post-exposure-efficacy abort for it, so it is always passed through. Fresh
# candidates are ranked by `priority` (computed once each, not on every
# comparison a sort makes, so a random `priority` is not resampled mid-sort)
# and admitted one at a time in that order for as long as the budget has
# anything left. Usage is counted once per call and then grows by each
# candidate whose `capacity_key` the wrapped intervention sets, which spares
# rescanning the whole population per candidate. Each of those candidates is
# stamped with the time of this call, so a dose uses the budget of the period
# that admitted it whatever date the dose itself bears, and later calls in
# that period see it.
# Some candidates fail the wrapped intervention's own checks (coverage,
# eligibility window, a missing required dose, an unresolved trace time)
# without using a dose; this keeps offering the budget to the next-ranked
# candidate instead of stopping after a fixed count, which would otherwise
# leave it under-used.
function apply_post_transmission!(cc::CapacityConstrained, state, new_contacts)
    actions = intervention_actions(cc, state, new_contacts)
    if actions !== nothing
        _admit_actions!(cc, state, actions)
        return nothing
    end
    key = capacity_key(cc.intervention)
    already_used = filter(ind -> get(ind.state, key, false), new_contacts)
    candidates = filter(ind -> !get(ind.state, key, false), new_contacts)
    isempty(already_used) || apply_post_transmission!(cc.intervention, state, already_used)
    isempty(candidates) && return nothing
    order = sortperm([cc.priority(ind, state) for ind in candidates]; alg = MergeSort)
    usage = _capacity_usage(cc, state)
    used = usage.used
    admission_key = _capacity_admission_key(key)
    for i in order
        used + 1 <= usage.available || break
        candidate = candidates[i]
        apply_post_transmission!(cc.intervention, state, [candidate])
        if get(candidate.state, key, false)
            candidate.state[admission_key] = state.max_infection_time
            used += 1
        end
    end
    return nothing
end

"""
    capacity_usage(cc::CapacityConstrained, state::SimulationState) -> (used, available)

Doses (or whatever `cc` rations) used and available at the point `state` has
reached: `used` is drawn from `state.individuals` via
[`capacity_key`](@ref EpiBranch.capacity_key). With `carry_over = true`
(the default), `available` is the lifetime allowance implied by
`state.max_infection_time`, `budget_per_period` and `period`, and `used` is
counted over the whole run. With `carry_over = false`, both are scoped to
the period `state.max_infection_time` falls in: `available` is a single
`budget_per_period`, and `used` counts the doses `cc` admitted during that
period together with any dose another intervention sharing the same
[`capacity_key`](@ref EpiBranch.capacity_key) timestamped
([`capacity_time_key`](@ref EpiBranch.capacity_time_key)) within it.
"""
function capacity_usage(cc::CapacityConstrained, state::SimulationState)
    return _capacity_usage(cc, state)
end

# ── Delegation of every other hook ────────────────────────────────────

# The remaining hooks are inherited from `InterventionWrapper`.
function keep_active(cc::CapacityConstrained, state, targets, is_new)
    keep_active(cc.intervention, state, targets, is_new)
end

# The continuous-time race has no batch of competing contacts to ration, so
# passing its tracing through would let the wrapped intervention act with no
# budget at all. Opting out keeps the documented "no effect" there.
traces_contacts(::CapacityConstrained) = false
trace_contacts!(::CapacityConstrained, state, infector, contacts) = nothing
