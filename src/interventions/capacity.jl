"""
    CapacityConstrained(intervention; budget_per_period, period=Inf,
                        carry_over=true, priority=default_capacity_priority)

Limit how many people an intervention can reach, such as a fixed number of
vaccine doses per day: `budget_per_period` people every `period` days.

# Examples
```julia
# Ring vaccination with 5 doses a day
CapacityConstrained(RingVaccination(efficacy = 0.8); budget_per_period = 5.0, period = 1.0)

# Group vaccination with 200 doses in total
CapacityConstrained(GroupVaccination(efficacy = 0.8); budget_per_period = 200.0)
```

# Arguments
- `budget_per_period`: number of people that can be served per period.
- `period`: length of a period in days, measured on the simulation clock (the
  latest infection time so far). `Inf` (default) gives a single budget for
  the whole outbreak.
- `carry_over`: whether unused capacity is added to later periods
  (default `true`); with `false` each period has its own allowance.
- `priority`: function `(individual, state) -> number` ordering who is served
  first, lowest first, with ties kept in order. The default,
  [`default_capacity_priority`](@ref), serves traced contacts in the order
  they were traced.

People turned away use no capacity and are not queued, though they may be
offered the intervention again if reached again later. Only whole people are
served, so a fraction of remaining capacity serves nobody. Capacity is
counted when a person is accepted, which can be before the dose is given (for
example after a delay), so it may fall in an earlier period than the dose.
Interventions with the same dose label share one budget, and doses given by
another intervention under that label count against it; use different
labels for separate budgets. [`capacity_usage`](@ref) reports the capacity
used and available.

[`Scheduled`](@ref) and `CapacityConstrained` can be combined in either
order: the schedule checks each person's action time, while capacity uses the
simulation clock.

!!! note "Supported models"
    In network and household models, ring and group vaccination can be
    capacity-limited, but ring vaccination then needs an infinite
    `eligibility_window`. Mass vaccination and interventions written without
    [`intervention_actions`](@ref EpiBranch.intervention_actions) cannot be
    capacity-limited there. A [`HomogeneousProcess`](@ref) does not trace
    contacts, so ring vaccination cannot act in it. Contact tracing itself
    cannot be capacity-limited in any model.

For extension authors: a new intervention can be capacity-limited by defining
[`intervention_actions`](@ref EpiBranch.intervention_actions) and
[`capacity_key`](@ref EpiBranch.capacity_key). Group vaccination offers every
eligible member of a triggered group as a separate person to serve.
Interventions without `intervention_actions` are offered the accepted people
one at a time.
"""
struct CapacityConstrained{I <: AbstractIntervention, F} <: InterventionWrapper
    intervention::I
    budget_per_period::Float64
    period::Float64
    carry_over::Bool
    priority::F
end

function CapacityConstrained(
        intervention::AbstractIntervention;
        budget_per_period::Real, period::Real = Inf, carry_over::Bool = true,
        priority = default_capacity_priority
    )
    budget_per_period >= 0 || throw(
        ArgumentError(
            "budget_per_period must be non-negative, got $budget_per_period"
        )
    )
    period > 0 || throw(ArgumentError("period must be positive, got $period"))
    return CapacityConstrained(
        intervention, Float64(budget_per_period), Float64(period), carry_over, priority
    )
end

"""
    default_capacity_priority(individual, state) -> Float64

Default order in which [`CapacityConstrained`](@ref) serves people: first
come, first served by the day they were traced. Someone who was not traced
comes last, after everyone traced at the same point of the simulation.
"""
function default_capacity_priority(individual, state)
    t = get(individual.state, :trace_time, NaN)
    return isnan(t) ? Inf : t
end

"""
    capacity_key(intervention) -> Symbol

For extension authors: the key in `ind.state` that marks a person as having
used one unit of `intervention`'s capacity (for vaccination, received the dose
with its `dose_label`). [`CapacityConstrained`](@ref) counts it to measure how
much capacity has been used. Defined for [`RingVaccination`](@ref),
[`MassVaccination`](@ref) and [`GroupVaccination`](@ref). Define it, and
[`capacity_time_key`](@ref EpiBranch.capacity_time_key) unless only
`carry_over = true` is used, to make a new intervention capacity-limited.
"""
function capacity_key(iv::AbstractIntervention)
    throw(
        ArgumentError(
            "CapacityConstrained needs a `capacity_key` method for $(typeof(iv)) " *
                "to know which state key measures how much of its resource has been " *
                "used. See the Extending guide."
        )
    )
end

"""
    capacity_time_key(intervention) -> Symbol

For extension authors: the key in `ind.state` holding the day the
[`capacity_key`](@ref EpiBranch.capacity_key) was set (for vaccination, the
vaccination day). With `carry_over = false`, [`CapacityConstrained`](@ref)
uses it to place a dose given by another intervention sharing the same
capacity in the right period.
"""
function capacity_time_key(iv::AbstractIntervention)
    throw(
        ArgumentError(
            "CapacityConstrained needs a `capacity_time_key` method for " *
                "$(typeof(iv)) to measure usage within a period (carry_over = false). " *
                "See the Extending guide."
        )
    )
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

The number of doses (or other units of capacity) given so far, and the number
that could have been given by now, at the end of a simulation or at the point
`state` has reached. With `carry_over = true` (the default) both count the
whole outbreak: `available` is `budget_per_period` for every period begun so
far. With `carry_over = false` both refer to the current period only:
`available` is one `budget_per_period`, and `used` counts the doses given in
that period, including doses given by another intervention sharing the same
capacity.
"""
function capacity_usage(cc::CapacityConstrained, state::SimulationState)
    return _capacity_usage(cc, state)
end

# ── Delegation of every other hook ────────────────────────────────────

# The remaining hooks are inherited from `InterventionWrapper`.
function keep_active(cc::CapacityConstrained, state, targets, is_new)
    return keep_active(cc.intervention, state, targets, is_new)
end

# The continuous-time race has no batch of competing contacts to ration, so
# passing its tracing through would let the wrapped intervention act with no
# budget at all. Opting out keeps the documented "no effect" there.
traces_contacts(::CapacityConstrained) = false
trace_contacts!(::CapacityConstrained, state, infector, contacts) = nothing

# Capacity gates who a removal reaches, never whether one already delivered
# still stands, so the stretches it recorded are read back unchanged.
function removal_gap_host_times(c::CapacityConstrained)
    return removal_gap_host_times(c.intervention)
end

# Admission reads `_capacity_usage`, how much of the shared budget every other
# individual has already used — population-wide state regardless of what the
# wrapped intervention declares.
reads_population_state(::CapacityConstrained) = true
