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
- `period`: length of a period in days, measured on the date each dose is
  actually given. `Inf` (default) gives a single budget for the whole
  outbreak.
- `carry_over`: whether unused capacity is added to later periods
  (default `true`); with `false` each period has its own allowance.
- `priority`: function `(individual, state) -> number` ordering who is served
  first, lowest first, with ties kept in order. The default,
  [`default_capacity_priority`](@ref), serves traced contacts in the order
  they were traced.

Each case's actions are offered together, in the order the engine settles
cases, and each is checked against the budget of the period its own proposed
date falls in, whatever day the case itself was settled on. People turned away
use no capacity and are not queued, though they may be offered the
intervention again if reached again later. Only whole people are served, so a
fraction of remaining capacity serves nobody. Interventions with the same dose
label share one budget, and doses given by another intervention under that
label count against it; use different labels for separate budgets.
[`capacity_usage`](@ref) reports the capacity used and available.

[`Scheduled`](@ref) and `CapacityConstrained` can be combined in either
order: both check each action's own proposed date.

!!! note "Supported models"
    In network and household models, ring and group vaccination can both be
    capacity-limited, with any `eligibility_window`: a dose a network or
    household race withdraws for falling outside the window (see
    [`RingVaccination`](@ref)) counts against no capacity from then on, so
    the budget it used is available to the next candidate. Mass vaccination
    does not reach anyone in those models at all, with or without a limit,
    and `simulate` warns that it will have no effect. It gives the same
    warning for any other intervention that relies on a hook these models do
    not call, such as one that reaches its targets only through the deprecated
    [`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!)
    batch hook, so for a custom intervention the warning is the thing to check.
    The mass-action models,
    [`HomogeneousProcess`](@ref) and [`MixingProcess`](@ref), do not trace
    contacts, so ring vaccination cannot act in either. Contact tracing itself
    cannot be capacity-limited in any model.

For extension authors: a new intervention can be capacity-limited by defining
[`intervention_actions`](@ref EpiBranch.intervention_actions) and
[`capacity_key`](@ref EpiBranch.capacity_key). Group vaccination offers every
eligible member of a triggered group as a separate person to serve.
Interventions without `intervention_actions`, offered through the deprecated
[`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!) batch
hook, propose no dated action, so each is checked against the budget of the
period the admitting call itself falls in rather than the day it is given.
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
    t = _state_real(individual.state, :trace_time, NaN)
    return isnan(t) ? Inf : t
end

"""
    capacity_key(intervention) -> Symbol

For extension authors: the key in `ind.state` that marks a person as having
used one unit of `intervention`'s capacity (for vaccination, received the dose
with its `dose_label`). [`CapacityConstrained`](@ref) counts it to measure how
much capacity has been used. Defined for [`RingVaccination`](@ref),
[`MassVaccination`](@ref) and [`GroupVaccination`](@ref). Define it, and
[`capacity_time_key`](@ref EpiBranch.capacity_time_key), to make a new
intervention capacity-limited.
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
vaccination day). [`CapacityConstrained`](@ref) places usage, including a
dose given by another intervention sharing the same capacity, in the period
this date falls in, so a dose a delay dates later than its admission is
charged to the day it is actually given.
"""
function capacity_time_key(iv::AbstractIntervention)
    throw(
        ArgumentError(
            "CapacityConstrained needs a `capacity_time_key` method for " *
                "$(typeof(iv)) to measure usage within a period. " *
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

# The `Individual.state` key `CapacityConstrained`'s deprecated
# `apply_post_transmission!` branch stamps with the time of the call that
# admitted an individual: that legacy protocol proposes no dated action, so
# it has nothing but its own clock to place the dose by. `_admit_actions!`,
# which every action-based intervention goes through, never sets it, since
# its candidates already name the date the dose is to be given.
_capacity_admission_key(key::Symbol) = Symbol("capacity_admission_time_", key)

# The start and end of the period containing `at`, in the scope `carry_over`
# implies: `(0.0, Inf)`, a single period for the whole outbreak, when `period`
# is infinite.
function _period_bounds(cc::CapacityConstrained, at)
    isinf(cc.period) && return (0.0, Inf)
    period_start = floor(at / cc.period) * cc.period
    return (period_start, period_start + cc.period)
end

# The time each already-given dose counts against, sorted: the admission
# stamp above where the deprecated branch left one, or otherwise the dose's
# own recorded date, so a dose a delay dates later than its admission is
# charged to the day it is actually given rather than to whichever call
# admitted it. Built once per call, so the admission loop below can test
# each candidate's own date against cumulative usage without rescanning the
# population once per candidate.
function _dose_times(cc::CapacityConstrained, state)
    key = capacity_key(cc.intervention)
    time_key = capacity_time_key(cc.intervention)
    admission_key = _capacity_admission_key(key)
    times = Float64[]
    for ind in state.individuals
        get(ind.state, key, false) || continue
        t = get(ind.state, admission_key, nothing)
        isnothing(t) && (t = get(ind.state, time_key, -Inf))
        push!(times, t)
    end
    sort!(times)
    return times
end

# How many of the sorted `times`, together with `extra` (doses admitted so far
# within the current call, not yet visible to a fresh scan of `state`), fall
# strictly before `before`.
function _count_before(times::Vector{Float64}, extra::Vector{Float64}, before::Float64)
    return searchsortedfirst(times, before) - 1 + count(<(before), extra)
end

# Doses used and available for the period containing `at`, in the scope
# `carry_over` implies. Reused by the admission decision, once per candidate
# with its own proposed date, and by `capacity_usage`, with the simulation's
# current clock.
function _usage_at(cc::CapacityConstrained, times, extra, at)
    period_start, period_end = _period_bounds(cc, at)
    if cc.carry_over
        available = isinf(cc.period) ? cc.budget_per_period :
            (period_end / cc.period) * cc.budget_per_period
        used = _count_before(times, extra, period_end)
    else
        available = cc.budget_per_period
        used = _count_before(times, extra, period_end) -
            _count_before(times, extra, period_start)
    end
    return (used = used, available = available)
end

# Whether one more dose at `at` keeps every period within its allowance. With
# `carry_over` a dose also counts against every later period, and the race
# admits doses in the order cases settle rather than by date, so a period
# that later doses already fill must still have room for this one.
function _fits_budget(cc::CapacityConstrained, times, extra, at)
    usage = _usage_at(cc, times, extra, at)
    usage.used + 1 <= usage.available || return false
    (cc.carry_over && isfinite(cc.period)) || return true
    _, own_end = _period_bounds(cc, at)
    for later in (@view(times[searchsortedfirst(times, own_end):end]), extra)
        for t in later
            t >= own_end || continue
            u = _usage_at(cc, times, extra, t)
            u.used + 1 <= u.available || return false
        end
    end
    return true
end

function _capacity_usage(cc::CapacityConstrained, state, at = state.max_infection_time)
    return _usage_at(cc, _dose_times(cc, state), Float64[], at)
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
_uses_batch_hook(cc::CapacityConstrained) = _uses_batch_hook(cc.intervention)
# `CapacityConstrained` always has its own `intervention_actions` method, to
# add the `capacity_key` check above, so asking `iv` directly would credit
# every instance with discovering actions whether or not the one it wraps
# does. Unwrapping through `_hook_owner`, as `_generation_hooks` already does
# for `keep_active`/`continuous_actions`/`on_infection_settled!`, reaches past
# any further `InterventionWrapper` nested inside (`Scheduled`, say) to the
# intervention that would answer `intervention_actions`.
function _discovers_actions(cc::CapacityConstrained)
    return _has_own_method(
        intervention_actions,
        typeof(_hook_owner(intervention_actions, cc.intervention)),
        AbstractIntervention
    )
end
# Asked for its actions on every settled case even when the intervention it
# wraps proposes none, so that wrapping one with nothing to ration fails when
# the run starts rather than going unrationed (see `intervention_actions`).
_proposes_actions(cc::CapacityConstrained) = !acts_on_population(cc)
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

The number of doses (or other units of capacity) dated by now, and the number
that could have been given by now, at the end of a simulation or at the point
`state` has reached. With `carry_over = true` (the default) both count the
whole outbreak: `available` is `budget_per_period` for every period begun so
far, and `used` counts every dose dated no later than it. With
`carry_over = false` both refer to the current period only: `available` is one
`budget_per_period`, and `used` counts the doses dated within it, including
one given in that period by another intervention sharing the same capacity.
"""
function capacity_usage(cc::CapacityConstrained, state::SimulationState)
    return _capacity_usage(cc, state)
end

# ── Delegation of every other hook ────────────────────────────────────

# `keep_active` is inherited from `InterventionWrapper`'s plain forwarding:
# a ring's growth spends no budget, so capacity has nothing to gate there.

# Tracing records no resource use for the budget to count, so passing it
# through would let the wrapped intervention act with no budget at all. Opting
# out leaves the tracing undone; on the generation engine a wrapped intervention
# without a `capacity_key` fails instead, when its actions are first sought.
traces_contacts(::CapacityConstrained) = false
trace_contacts!(::CapacityConstrained, state, infector, contacts) = nothing

# Capacity gates who a removal reaches, never whether one already delivered
# still stands, so the stretches it recorded are read back unchanged.
function removal_gap_host_times(c::CapacityConstrained)
    return removal_gap_host_times(c.intervention)
end
removal_block_probability(c::CapacityConstrained) = removal_block_probability(c.intervention)

# Admission reads `_capacity_usage`, how much of the shared budget every other
# individual has already used — population-wide state regardless of what the
# wrapped intervention declares.
reads_population_state(::CapacityConstrained) = true
