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
    In network and household models, ring, group and mass vaccination can all
    be capacity-limited, with any `eligibility_window`: a dose a network or
    household race withdraws for falling outside the window (see
    [`RingVaccination`](@ref)) counts against no capacity from then on, and
    the budget it used becomes available to the next candidate. The
    mass-action models,
    [`HomogeneousProcess`](@ref) and [`MixingProcess`](@ref), do not trace
    contacts, so ring vaccination cannot act in either.

    [`ContactTracing`](@ref) can be capacity-limited too, when its `action` is
    a [`Quarantine`](@ref): a contact is still found and still grows a ring
    once budget runs out, but is not itself quarantined until a place frees
    up. `FlagOnly` has no removal to ration. [`Isolation`](@ref) can also be
    capacity-limited, as a count of isolations admitted per period rather than
    beds held at once: a case isolated once the period's budget is spent is
    released again, the same way `Scheduled` releases one isolated outside
    its window.

For extension authors: a new intervention can be capacity-limited by defining
[`intervention_actions`](@ref EpiBranch.intervention_actions) and
[`capacity_key`](@ref EpiBranch.capacity_key). Group vaccination offers every
eligible member of a triggered group as a separate person to serve. One that
instead sets its effect directly in
[`resolve_individual!`](@ref EpiBranch.resolve_individual!), as `Isolation`
does, defines [`intervention_time`](@ref EpiBranch.intervention_time),
[`reset!`](@ref EpiBranch.reset!) and `capacity_key` instead:
`CapacityConstrained` then undoes the effect once the budget is spent, the
way `Scheduled` already undoes one outside its own window.
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

# `Isolation` sets its own effect directly in `resolve_individual!` rather
# than proposing an action; `:_isolated_by_isolation` is the flag it already
# keeps to tell its own isolation from a quarantine layered under it (see its
# `reset!`), which is exactly "this budget was spent on this person".
capacity_key(::Isolation) = :_isolated_by_isolation
capacity_time_key(::Isolation) = :isolation_time

# `ContactTracing` proposes its follow-up as an `InterventionAction` (see
# `_trace_action`) whose own resource key depends on what the action does:
# a `Quarantine` has a removal to ration, `FlagOnly` does not.
capacity_key(ct::ContactTracing) = capacity_key(ct.action)
capacity_time_key(ct::ContactTracing) = capacity_time_key(ct.action)

function capacity_key(action::TraceAction)
    throw(
        ArgumentError(
            "CapacityConstrained needs a `capacity_key` method for $(typeof(action)) " *
                "to know which state key measures how much of `ContactTracing`'s " *
                "follow-up capacity a person has used. See the Extending guide."
        )
    )
end
function capacity_time_key(action::TraceAction)
    throw(
        ArgumentError(
            "CapacityConstrained needs a `capacity_time_key` method for " *
                "$(typeof(action)) to measure usage within a period. See the " *
                "Extending guide."
        )
    )
end
capacity_key(::Quarantine) = :_quarantined_by_tracing
capacity_time_key(::Quarantine) = :trace_time

# The start and end of the period containing `at`, in the scope `carry_over`
# implies: `(0.0, Inf)`, a single period for the whole outbreak, when `period`
# is infinite.
function _period_bounds(cc::CapacityConstrained, at)
    isinf(cc.period) && return (0.0, Inf)
    period_start = floor(at / cc.period) * cc.period
    return (period_start, period_start + cc.period)
end

# The time each already-given dose counts against, sorted: the dose's own
# recorded date, so a dose a delay dates later than its admission is charged
# to the day it is actually given rather than to whichever call admitted it.
# Built once per call, so the admission loop below can test
# each candidate's own date against cumulative usage without rescanning the
# population once per candidate.
function _dose_times(cc::CapacityConstrained, state)
    key = capacity_key(cc.intervention)
    time_key = capacity_time_key(cc.intervention)
    times = Float64[]
    for ind in state.individuals
        get(ind.state, key, false) || continue
        push!(times, get(ind.state, time_key, -Inf))
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

# `CapacityConstrained` always has its own `intervention_actions` method, to
# add the `capacity_key` check above, so asking `iv` directly would credit
# every instance with discovering actions whether or not the one it wraps
# does. Unwrapping through `_hook_owner` reaches past any further
# `InterventionWrapper` nested inside (`Scheduled`, say) to the intervention
# that would answer `intervention_actions`.
function _discovers_actions(cc::CapacityConstrained)
    return _has_own_method(
        intervention_actions,
        typeof(_hook_owner(intervention_actions, cc.intervention)),
        AbstractIntervention
    )
end
# Asked for its actions on every settled case even when the intervention it
# wraps proposes none, so that wrapping one with nothing to ration fails when
# the run starts rather than going unrationed (see `intervention_actions`) —
# unless it instead rations through `resolve_individual!` and `reset!`, as
# `Isolation` does (`_resolves_under_capacity`). That seam needs no candidates
# or contacts, and routing it through the action funnel too would only cost a
# contactless model (`HomogeneousProcess`, `MixingProcess`) a spurious warning
# that it has nothing for `Isolation` to act on.
function _proposes_actions(cc::CapacityConstrained)
    acts_on_population(cc) && return false
    return !_resolves_under_capacity(cc.intervention)
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

# Tracing's own bookkeeping (who was reached, and from when) runs
# unconditionally, so that ring growth and later eligibility checks, decided
# before any action is admitted, see it; only the follow-up action it
# proposes (see `_trace_action`) is rationed. `state.scratch` carries this
# wrapper down to the span of the wrapped intervention's own
# `trace_contacts!` call, which admits through it instead of itself (see
# `_admit_trace_actions!`) — composing with `Scheduled` in either order, as a
# dose already does.
traces_contacts(cc::CapacityConstrained) = traces_contacts(cc.intervention)
function trace_contacts!(cc::CapacityConstrained, state, infector, contacts, not_before = nothing)
    key = (:trace_admitter, _innermost(cc))
    previous = get(state.scratch, key, nothing)
    state.scratch[key] = cc
    try
        if not_before === nothing
            trace_contacts!(cc.intervention, state, infector, contacts)
        else
            trace_contacts!(cc.intervention, state, infector, contacts, not_before)
        end
    finally
        previous === nothing ? delete!(state.scratch, key) : (state.scratch[key] = previous)
    end
    return nothing
end

# Whether the intervention that owns the `capacity_key` of `iv` (unwrapped
# past anything that only forwards it) writes its own effect directly in
# `resolve_individual!`, as `Isolation` does. `CapacityConstrained` then gates
# that effect through `intervention_time` and `reset!`, the way `Scheduled`
# already does for its own window, rather than through the action funnel an
# intervention proposing no effect of its own in `resolve_individual!` (a
# dose) uses instead. Found through the owner of `capacity_key` rather than
# the owner of `resolve_individual!`: `Scheduled` always defines the latter
# itself, to gate whatever it wraps through its own window, which would
# otherwise stop the search there and credit it to a dose a
# `Scheduled(RingVaccination(...))` still proposes through the funnel.
function _resolves_under_capacity(iv::AbstractIntervention)
    owner = _hook_owner(capacity_key, iv)
    _has_own_method(capacity_key, typeof(owner), AbstractIntervention) || return false
    return _has_own_method(resolve_individual!, typeof(owner), AbstractIntervention)
end

function resolve_individual!(cc::CapacityConstrained, ind, state)
    resolve_individual!(cc.intervention, ind, state)
    _resolves_under_capacity(cc.intervention) && _reset_unless_within_budget!(cc, ind, state)
    return nothing
end

# As `_dose_times`, but leaving out `excluded`'s own record: used to judge
# whether admitting the effect `resolve_individual!` just wrote on `excluded`
# keeps every period within budget, which must not count that same use
# twice.
function _dose_times(cc::CapacityConstrained, state, excluded::Individual)
    key = capacity_key(cc.intervention)
    time_key = capacity_time_key(cc.intervention)
    times = Float64[]
    for ind in state.individuals
        ind.id == excluded.id && continue
        get(ind.state, key, false) || continue
        push!(times, get(ind.state, time_key, -Inf))
    end
    sort!(times)
    return times
end

# Undo the effect `resolve_individual!` just wrote on `ind` unless it fits
# the budget at its own date, the same check a proposed action's own date
# goes through. An effect that never ran (the capacity key not actually set
# on `ind` — for instance a quarantine `Isolation` is layered over, which is
# not its own effect) has nothing to check.
function _reset_unless_within_budget!(cc::CapacityConstrained, ind::Individual, state)
    key = capacity_key(cc.intervention)
    get(ind.state, key, false)::Bool || return nothing
    t = intervention_time(cc.intervention, ind)
    isfinite(t) || return nothing
    times = _dose_times(cc, state, ind)
    _fits_budget(cc, times, Float64[], t) || reset!(cc.intervention, ind)
    return nothing
end

# Capacity gates who a removal reaches, never whether one already delivered
# still stands, so the stretches it recorded are read back unchanged.
function removal_gap_host_times(c::CapacityConstrained)
    return removal_gap_host_times(c.intervention)
end
removal_block_probability(c::CapacityConstrained) = removal_block_probability(c.intervention)
