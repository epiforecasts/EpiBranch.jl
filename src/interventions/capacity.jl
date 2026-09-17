"""
Ration a scarce, population-level resource across the individuals competing
for it in the same period. Wraps an intervention in the way [`Scheduled`](@ref)
wraps one for time: `Scheduled` gates *when* an intervention's action may
occur, `CapacityConstrained` gates *how many* of a period's competing
candidates it may reach.

Out of the box this rations vaccine doses ([`RingVaccination`](@ref),
[`MassVaccination`](@ref)); a resource other than doses (contact-tracing
teams, geographic reach) needs a [`capacity_key`](@ref EpiBranch.capacity_key)
method for the intervention that stands for it, which is not yet defined for
anything in this package.

# How it works

`apply_post_transmission!` is the one hook the engine calls with a whole
generation's contacts at once — the only point in the protocol where several
individuals compete for the same resource in the same call.
`CapacityConstrained` intercepts that call: it ranks `new_contacts` by
`priority` (lower goes first) and hands them to the wrapped intervention one
at a time in that order, stopping once the budget is exhausted. The rest are
simply never handed to the wrapped intervention this call, so it never acts
on them; a call whose demand the budget covers in full is unaffected,
whatever the ranking. A candidate the wrapped intervention turns away for its
own reasons (failed `coverage`, outside `eligibility_window`, and so on) uses
no budget, so it does not stop a later-ranked candidate from being tried.

Every other hook (`competing_risk`, `resolve_individual!`, `keep_active`, …)
is delegated unchanged, so `CapacityConstrained` only ever rations the
population-level batch, never a per-pair decision. It rations
[`AbstractVaccination`](@ref)'s dose-recording hook out of the box (for
[`RingVaccination`](@ref) and [`MassVaccination`](@ref); see
[`capacity_key`](@ref EpiBranch.capacity_key) for why [`GroupVaccination`](@ref)
is not supported), and any other intervention that defines
[`capacity_key`](@ref EpiBranch.capacity_key) for itself.

# Budget

`budget_per_period` doses become available every `period` time units,
measured on the simulation's own continuous clock (`state.max_infection_time`),
not the generation index. `period = Inf` (the default) is a single lifetime
budget that never replenishes.

!!! warning "Generation-based models only"
    `CapacityConstrained` acts through `apply_post_transmission!`, which only
    the generation-based engine calls. A continuous-time model does not call
    it, so wrapping an intervention with `CapacityConstrained` there has no
    effect and triggers the usual "no effect" warning for an unhonoured
    intervention.

`carry_over = true` (the default) lets an unused allowance from an earlier
period add to a later one: the running total available by time `t` is
`budget_per_period * (floor(t / period) + 1)`, compared against the lifetime
count used. With `carry_over = false`, a period's unused budget is lost —
only `budget_per_period` is available within the current period, measured
against usage recorded since that period began.

Demand denied this call is not queued: a candidate who is not reached while
the budget is exhausted is not revisited in a later call. Reporting what is
left of a call's own demand is the caller's job (e.g. counting
eligible-but-unreached candidates on the finished [`linelist`](@ref)), not
this wrapper's.

# Priority

`priority(individual, state) -> Real`, lower served first, decides who is
served when a call's demand exceeds what remains of the budget. The default,
[`default_capacity_priority`](@ref), is first-come-first-served by
`:trace_time` (a candidate with no recorded trace time sorts last). Pass any
function for a different rule: nearest-first from a custom distance, a
random draw for lottery allocation, or a composite of several factors. Ties
are broken by the order `new_contacts` arrives in.

!!! warning "The default is not first-come-first-served for every intervention"
    [`MassVaccination`](@ref) candidates carry no `:trace_time`, so every one
    ties at `Inf` under the default and the tie-break — arrival order in
    `new_contacts` — decides instead. That order follows contact creation,
    not any notion of when a candidate became eligible, and favours whichever
    chain happens to be processed first (including the seeds). Pass a
    `priority` that reads a field `MassVaccination` actually sets (there is
    none before its own call records `:vaccination_time`) for a rule that
    means something for it.

# Reporting

[`capacity_usage`](@ref) reads back doses used against doses available at a
given point in the simulation.

# Composing with `Scheduled`

`CapacityConstrained` and [`Scheduled`](@ref) wrap in either order —
`CapacityConstrained(Scheduled(rv); ...)` and `Scheduled(CapacityConstrained(rv;
...))` both work — since each delegates the other's hooks through.

# Examples

```julia
# Five doses a day, replenishing, ring vaccination
CapacityConstrained(RingVaccination(efficacy = 0.8); budget_per_period = 5.0, period = 1.0)

# A one-off stockpile of 200 doses for the whole outbreak
CapacityConstrained(MassVaccination(efficacy = 0.85, eligibility_time = 30.0);
    budget_per_period = 200.0)

# Random allocation among a day's competing demand instead of FCFS
CapacityConstrained(RingVaccination(efficacy = 0.8);
    budget_per_period = 5.0, period = 1.0,
    priority = (ind, state) -> rand(state.rng))
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

[`GroupVaccination`](@ref) does not define it: it vaccinates a triggered
group by scanning the whole population, not the batch `new_contacts` that
`CapacityConstrained` can ration, so limiting that batch would not limit the
doses actually given.
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
read by [`CapacityConstrained`](@ref) when `carry_over = false` to count only
the usage recorded within the current period.
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

function capacity_key(::GroupVaccination)
    throw(ArgumentError(
        "CapacityConstrained does not support GroupVaccination: it " *
        "vaccinates a triggered group by scanning the whole population, not " *
        "the batch CapacityConstrained can ration, so limiting the batch " *
        "would not limit the doses given."))
end

# Doses (or whatever `cc` rations) used and available, in the scope
# `carry_over` implies, at `state`'s current point in time. Reused by the
# admission decision and by `capacity_usage`.
function _capacity_usage(cc::CapacityConstrained, state)
    key = capacity_key(cc.intervention)
    t = state.max_infection_time
    if cc.carry_over
        n_periods = isinf(cc.period) ? 1.0 : floor(t / cc.period) + 1.0
        available = n_periods * cc.budget_per_period
        used = count(ind -> get(ind.state, key, false), state.individuals)
    else
        time_key = capacity_time_key(cc.intervention)
        period_start = isinf(cc.period) ? 0.0 : floor(t / cc.period) * cc.period
        available = cc.budget_per_period
        used = count(state.individuals) do ind
            get(ind.state, key, false) && get(ind.state, time_key, -Inf) >= period_start
        end
    end
    return (used = used, available = available)
end

function _remaining_budget(cc::CapacityConstrained, state)
    usage = _capacity_usage(cc, state)
    return usage.available - usage.used
end

# A candidate already carrying `capacity_key` (dosed in an earlier call, now
# re-exposed) does not draw on the budget when it is handed to the wrapped
# intervention again — `RingVaccination`/`MassVaccination` only redraw a
# post-exposure-efficacy abort for it, so it is always passed through. Fresh
# candidates are ranked by `priority` (computed once each, not on every
# comparison a sort makes, so a random `priority` is not resampled mid-sort)
# and admitted one at a time in that order for as long as the budget —
# rechecked against actual usage after each admission — has anything left.
# Some candidates fail the wrapped intervention's own checks (coverage,
# eligibility window, a missing required dose, an unresolved trace time)
# without using a dose; this keeps offering the budget to the next-ranked
# candidate instead of stopping after a fixed count, which would otherwise
# leave it under-used.
function apply_post_transmission!(cc::CapacityConstrained, state, new_contacts)
    key = capacity_key(cc.intervention)
    already_used = filter(ind -> get(ind.state, key, false), new_contacts)
    candidates = filter(ind -> !get(ind.state, key, false), new_contacts)
    isempty(already_used) || apply_post_transmission!(cc.intervention, state, already_used)
    isempty(candidates) && return nothing
    order = sortperm([cc.priority(ind, state) for ind in candidates]; alg = MergeSort)
    for i in order
        _remaining_budget(cc, state) > 0 || break
        apply_post_transmission!(cc.intervention, state, [candidates[i]])
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
`budget_per_period` and `used` only counts doses timestamped
([`capacity_time_key`](@ref)) within that period.
"""
function capacity_usage(cc::CapacityConstrained, state::SimulationState)
    return _capacity_usage(cc, state)
end

# ── Delegation of every other hook ────────────────────────────────────

# The remaining hooks are inherited from `InterventionWrapper`.
function keep_active(cc::CapacityConstrained, state, targets, is_new)
    keep_active(cc.intervention, state, targets, is_new)
end
