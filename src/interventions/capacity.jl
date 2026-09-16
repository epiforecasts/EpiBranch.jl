"""
Ration a scarce, population-level resource — doses, teams, reach — across
the individuals competing for it in the same period. Wraps an intervention
in the way [`Scheduled`](@ref) wraps one for time: `Scheduled` gates *when*
an intervention's action may occur, `CapacityConstrained` gates *how many*
of a period's competing candidates it may reach.

# How it works

`apply_post_transmission!` is the one hook the engine calls with a whole
generation's contacts at once — the only point in the protocol where several
individuals compete for the same resource in the same call.
`CapacityConstrained` intercepts that call: it ranks `new_contacts` by
`priority` (lower goes first) and passes through to the wrapped intervention
only as many, from the front of that order, as the remaining budget allows.
The rest are simply never handed to the wrapped intervention this call, so
it never acts on them; a call whose demand the budget covers in full is
unaffected, whatever the ranking.

Every other hook (`competing_risk`, `resolve_individual!`, `keep_active`, …)
is delegated unchanged, so `CapacityConstrained` only ever rations the
population-level batch, never a per-pair decision. It rations
[`AbstractVaccination`](@ref)'s dose-recording hook out of the box (for
[`RingVaccination`](@ref) and [`MassVaccination`](@ref); see
[`capacity_key`](@ref EpiBranch.capacity_key) for why [`GroupVaccination`](@ref)
is not supported), and any other intervention that defines
[`capacity_key`](@ref EpiBranch.capacity_key) for itself.

# Budget

`budget_per_period` doses (or team-slots, or reach) become available every
`period` time units, measured on the simulation's own continuous clock
(`state.max_infection_time`), not the generation index. A generation-based
model still has this clock — it is built from the same generation-time draws
that place every case in time — so a budget of "5 doses a day" means the same
thing whichever engine simulates it. `period = Inf` (the default) is a single
lifetime budget that never replenishes.

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

# Reporting

[`capacity_usage`](@ref) reads back doses used against doses available at a
given point in the simulation.

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
struct CapacityConstrained{I <: AbstractIntervention, F} <: AbstractIntervention
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

# How much of the budget remains at `state`'s current point in time. Reused
# by the admission decision and by `capacity_usage`.
function _remaining_budget(cc::CapacityConstrained, state)
    key = capacity_key(cc.intervention)
    t = state.max_infection_time
    if cc.carry_over
        n_periods = isinf(cc.period) ? 1.0 : floor(t / cc.period) + 1.0
        allowance = n_periods * cc.budget_per_period
        used = count(ind -> get(ind.state, key, false), state.individuals)
    else
        time_key = capacity_time_key(cc.intervention)
        period_start = isinf(cc.period) ? 0.0 : floor(t / cc.period) * cc.period
        allowance = cc.budget_per_period
        used = count(state.individuals) do ind
            get(ind.state, key, false) && get(ind.state, time_key, -Inf) >= period_start
        end
    end
    return allowance - used
end

function apply_post_transmission!(cc::CapacityConstrained, state, new_contacts)
    remaining = _remaining_budget(cc, state)
    remaining > 0 || return nothing
    n_admit = min(length(new_contacts), floor(Int, remaining))
    n_admit > 0 || return nothing
    admitted = if n_admit == length(new_contacts)
        new_contacts
    else
        partialsort(new_contacts, 1:n_admit; by = ind -> cc.priority(ind, state))
    end
    apply_post_transmission!(cc.intervention, state, admitted)
    return nothing
end

"""
    capacity_usage(cc::CapacityConstrained, state::SimulationState) -> (used, available)

Doses (or whatever `cc` rations) used and available at the point `state` has
reached: `used` is drawn from `state.individuals` via
[`capacity_key`](@ref EpiBranch.capacity_key), and `available` is the
lifetime allowance implied by `state.max_infection_time`, `budget_per_period`
and `period`.
"""
function capacity_usage(cc::CapacityConstrained, state::SimulationState)
    key = capacity_key(cc.intervention)
    used = count(ind -> get(ind.state, key, false), state.individuals)
    t = state.max_infection_time
    n_periods = isinf(cc.period) ? 1.0 : floor(t / cc.period) + 1.0
    available = n_periods * cc.budget_per_period
    return (used = used, available = available)
end

# ── Delegation of every other hook ────────────────────────────────────

function initialise_individual!(cc::CapacityConstrained, ind, state)
    initialise_individual!(cc.intervention, ind, state)
end
function resolve_individual!(cc::CapacityConstrained, ind, state)
    resolve_individual!(cc.intervention, ind, state)
end
function competing_risk(cc::CapacityConstrained, parent, contact, state)
    competing_risk(cc.intervention, parent, contact, state)
end
function keep_active(cc::CapacityConstrained, state, targets, is_new)
    keep_active(cc.intervention, state, targets, is_new)
end
required_fields(cc::CapacityConstrained) = required_fields(cc.intervention)
function intervention_time(cc::CapacityConstrained, ind::Individual)
    intervention_time(cc.intervention, ind)
end
reset!(cc::CapacityConstrained, ind::Individual) = reset!(cc.intervention, ind)
function infectious_removal_time(cc::CapacityConstrained, ind::Individual)
    infectious_removal_time(cc.intervention, ind)
end
function is_active(cc::CapacityConstrained, state::SimulationState)
    is_active(cc.intervention, state)
end
_unwrap_scheduled(cc::CapacityConstrained) = _unwrap_scheduled(cc.intervention)
traces_contacts(cc::CapacityConstrained) = traces_contacts(cc.intervention)
function trace_contacts!(cc::CapacityConstrained, state, infector, contacts)
    trace_contacts!(cc.intervention, state, infector, contacts)
end
function trace_contacts!(cc::CapacityConstrained, state, infector, contacts, not_before)
    trace_contacts!(cc.intervention, state, infector, contacts, not_before)
end
