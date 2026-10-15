"""
    Scheduled(intervention; start_time = nothing, end_time = nothing,
              start_after_cases = nothing, start_after = nothing)
    Scheduled(intervention, condition)

Start (and optionally stop) an intervention at a given day or once the
outbreak has reached a given number of cases, for example a policy introduced
two weeks into an outbreak.

# Examples
```julia
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)

# Isolation from day 14
Scheduled(iso; start_time = 14.0)

# Tracing once there have been 50 cases
Scheduled(
    ContactTracing(
        probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
        action = Quarantine(duration = 7.0),
    );
    start_after_cases = 50,
)

# Isolation between days 10 and 30
Scheduled(iso; start_time = 10.0, end_time = 30.0)

# Isolation once 20 cases have been reported
Scheduled(iso; start_after = ReportedCases(20))

# Any condition on the simulation, here from the third generation on
Scheduled(iso, state -> state.current_generation >= 3)
```

# Arguments
- `start_time`, `end_time`: days (as decimals, such as `14.0`) between which
  the intervention acts, read against the day the action itself falls on
  (a case's isolation, a contact's trace) rather than the case's infection
  time, exactly as a scheduled vaccination's dose is checked against its own
  date.
- `start_after_cases`: number of infections after which it acts, counting
  every infection as it happens, whether or not it is ever reported.
- `start_after`: a trigger such as [`ReportedCases`](@ref), which counts only
  cases reported by that day, or [`Infections`](@ref). `start_after_cases = n`
  is the same as `start_after = Infections(n)`.
- `condition`: a function of the simulation state returning `true` while the
  intervention should act.

Give at least one; several keywords must all hold. An isolation or trace
dated before `start_time`, or after `end_time`, is undone with
[`reset!`](@ref EpiBranch.reset!): with `start_time = 10.0`, a case infected
on day 8 whose symptoms begin on day 9, with a 2-day delay putting their
isolation on day 11, is isolated after all, even though they were infected
before the policy began.

Isolation and quarantine already in place end when the schedule ends (after
`end_time`, or once `condition` returns `false`). The exception is the
mass-action, network and household models, where a perfect isolation
(`post_isolation_transmission = 0`) with no end (`duration = Inf`), or a
quarantine with no end, keeps the case out of transmission for good.
Only protection the
intervention keeps, such as vaccination, continues (see
[`persistent_competing_risks`](@ref EpiBranch.persistent_competing_risks)).
Everyone still records the intervention's starting information, such as "not
isolated", before it starts.

A finite isolation or quarantine `duration` can end it before the schedule
does, never later.

For extension authors: an intervention that proposes its actions through
[`intervention_actions`](@ref EpiBranch.intervention_actions) has each action
checked against the schedule at the action's own time (seen by the condition
as `state.max_infection_time`). Among the rest, one that defines its own
[`intervention_time`](@ref EpiBranch.intervention_time) is run whether or not
the schedule is active at the case's or contact's own clock, and its effect
undone with [`reset!`](@ref EpiBranch.reset!) once `intervention_time` is
checked against the condition the same way, with `state.max_infection_time`
moved to that time for the check; a trigger that reads other population-wide
state, such as [`ReportedCases`](@ref), sees it as it stands now, not as it
will once that time comes, the same approximation a scheduled vaccination's
dose already accepts. An intervention relying on the default `intervention_time`
(`-Inf`) is instead skipped outright while the schedule is inactive at that
clock, as every hook always was: `intervention_time` comes paired with a
`reset!` that undoes whatever it names, and without one there is nothing to
correct the effect with afterwards.
"""
struct Scheduled{I <: AbstractIntervention, F, T <: Tuple} <: InterventionWrapper
    intervention::I
    condition::F
    start_time::Float64
    # Whether `condition` can read population-wide state (a case count) rather
    # than only the time of the case being resolved. See `reads_population_state`.
    population_dependent::Bool
    # Whether the active window can close again once open. Of the conditions
    # the keyword constructor builds, `start_time` and `start_after_cases`
    # only ever open; `end_time`, and a `start_after` trigger whose
    # `trigger_can_lapse` is `true`, can withdraw an effect already
    # delivered. An opaque predicate is assumed to.
    can_lapse::Bool
    # The `start_after` triggers, kept so that the composition checks can ask
    # each what it needs of the progression; see `required_transitions`. The
    # condition itself is an opaque closure. Empty for every other form.
    triggers::T
    # Whether `intervention` has its own `intervention_time`, computed once
    # here rather than by introspection on every call. See `_time_aware`.
    time_aware::Bool
end

# Computes `time_aware` from `intervention` once, rather than asking on every
# call a hook makes.
function Scheduled(
        intervention::AbstractIntervention, condition, start_time::Float64,
        population_dependent::Bool, can_lapse::Bool, triggers::Tuple
    )
    return Scheduled(
        intervention, condition, start_time, population_dependent, can_lapse,
        triggers, _time_aware(intervention)
    )
end

# Pre-existing callers build a `Scheduled` without the trigger record.
function Scheduled(
        intervention::AbstractIntervention, condition, start_time::Float64,
        population_dependent::Bool, can_lapse::Bool
    )
    return Scheduled(
        intervention, condition, start_time, population_dependent, can_lapse, ()
    )
end

# Whether `iv` has declared its own `intervention_time`, which by that hook's
# contract comes paired with a `reset!` undoing whatever it names (see
# `intervention_time`'s docstring). `Scheduled` only runs such an
# intervention unconditionally and corrects the result afterwards, in
# `_reset_unless_in_schedule!`; one relying on the defaults (`-Inf`, a no-op
# `reset!`) has given it nothing to undo, and is instead gated up front, as
# every hook always was.
function _time_aware(iv::AbstractIntervention)
    return _has_own_method(
        intervention_time, typeof(_hook_owner(intervention_time, iv)), AbstractIntervention
    )
end

"""
    AbstractTrigger

Base type for a `start_after` trigger passed to [`Scheduled`](@ref), as in
`Scheduled(iv; start_after = ReportedCases(20))`. A subtype implements
[`is_triggered!`](@ref) to say whether its condition holds now, and may
override the trait [`trigger_can_lapse`](@ref) to say whether that can turn
false again. `Scheduled` reads both to set its own `population_dependent`
and `can_lapse` flags, so a trigger written outside the package works
without editing `Scheduled`.
"""
abstract type AbstractTrigger end

"""
    is_triggered!(trigger, state::SimulationState) -> Bool

Whether `trigger`'s condition holds now. A trigger must keep no state of its
own between calls, because the same trigger object is reused across the
repeated simulations of `simulate(model, n)` (serially, or across threads
under `parallel = true`): derive what it needs from `state`, or keep working
values in `state.scratch`, which belongs to the run.

The engine asks the gate about 1.7 times per individual: a trigger that
walks the population on every call makes a run quadratic in the number of
individuals, as [`ReportedCases`](@ref) does; one that can read a running
total the engine already keeps, as [`Infections`](@ref) reads
`cumulative_cases`, costs nothing by comparison.
"""
function is_triggered! end

"""
    trigger_can_lapse(trigger) -> Bool

Whether `trigger`'s condition, once true, can turn false again. The
default is the conservative `true`: a trigger written outside the package
is assumed able to close again until it says otherwise, as an opaque
[`Scheduled`](@ref) predicate already is. [`Infections`](@ref) and
[`ReportedCases`](@ref) both return `false`, since neither `cumulative_cases`
nor the number of reports already due can fall.

Answer about the accepted timeline. Admitting a scheduled action asks the gate
with that action's own time in place of the simulation clock, which settles
whether that action is admitted rather than whether the window closes. The
distinction is worth the care: a schedule that can close speaks for none of
the isolation releases its intervention recorded; `true` where `false`
belongs costs a likelihood term rather than buying safety.
"""
trigger_can_lapse(::AbstractTrigger) = true

"""
    required_transitions(trigger) -> Tuple of transition types

The transition types a trigger needs the model's `progression` to hold.
Default: `()`, nothing required.

A trigger reads `Individual.state`, and the keys it reads are written by the
progression: a model composed without the transition that writes them
leaves the gate shut for the whole run. [`ModelSpec`](@ref) asks this of every
`start_after` trigger and warns when the model does not hold what the trigger
names. [`ReportedCases`](@ref) names [`Reporting`](@ref).

A key an attributes function supplies, such as `:onset_time` from
[`clinical_presentation`](@ref), cannot be named this way.
"""
required_transitions(::AbstractTrigger) = ()

"""
    Infections(n)

A `start_after` trigger for [`Scheduled`](@ref) that opens once there have
been `n` infections. Each infection counts as it happens, including one that is
never reported or is reported later. `Scheduled(iv; start_after = Infections(n))`
is the same as `Scheduled(iv; start_after_cases = n)`.
"""
struct Infections <: AbstractTrigger
    n::Int
end
is_triggered!(t::Infections, state::SimulationState) = state.cumulative_cases >= t.n
trigger_can_lapse(::Infections) = false

# Whether a report in `st` is dated by the clock reading `now`. Read with
# `get` rather than indexed: a `:reported` flag set by something other than
# the `Reporting` transition need not come with a `:reporting_time`.
_report_due(st, now) =
    get(st, :reported, false) && get(st, :reporting_time, Inf) <= now

_report_time(st) = get(st, :reporting_time, Inf)

# The reports one host holds that are dated by `now`, over its archived
# episodes and its live state.
#
# A reinfection archives the episode as a copy of the live state and only
# clears the live `:reported` later, when the new infection's transitions are
# initialised (`close_episode!` in `_resolve!`, then `resolve_transitions!`).
# In between, the newest episode and the live state are one report: the
# live one is counted only when its date differs from that episode's.
function _host_reports_due(ind, now)
    due = 0
    for episode in ind.episodes
        _report_due(episode.state, now) && (due += 1)
    end
    _report_due(ind.state, now) || return due
    isempty(ind.episodes) && return due + 1
    archived = _report_time(last(ind.episodes).state)
    return archived == _report_time(ind.state) ? due : due + 1
end

# The reports the population holds that are dated by
# `state.max_infection_time`, counting every episode of a host infected more
# than once: a reinfection resets the live `:reported` flag and archives the
# episode carrying it, and a report the health system has made stays made.
#
# Derived from `state.individuals` rather than kept as a running total:
# `make_contact!` appends an individual before its clinical transitions run
# and `Reporting` is one of those, so a host counted on an earlier call
# cannot be taken as finished with.
#
# Counted on every call; the engine asks the gate about 1.7 times per
# individual, so a run with this trigger is quadratic in the number of
# individuals: about 0.02 s at 1,000 of them and 190 s at 57,000. Caching the
# count against the clock is not available, because `resolve_transitions!`
# sets `:reported` on each newly infected case as the generation resolves,
# while the clock the count is read against moves only between generations.
# Making it cheap needs the count maintained where the reports are written,
# or the gate asked once a generation rather than once a pair, neither of
# which a trigger can reach from outside the engine.
function _reported_cases_due(state::SimulationState)
    now = state.max_infection_time
    due = 0
    for ind in state.individuals
        due += _host_reports_due(ind, now)
    end
    return due
end

"""
    ReportedCases(n)

A `start_after` trigger for [`Scheduled`](@ref) that opens once `n` cases
have been reported by the current day of the simulation. A case reported after
that day does not count yet, unlike in [`Infections`](@ref). The model's
`progression` needs a [`Reporting`](@ref) step to record the reports, except
when `n = 0`, which opens from the start.
"""
struct ReportedCases <: AbstractTrigger
    n::Int
end
is_triggered!(t::ReportedCases, state::SimulationState) = _reported_cases_due(state) >= t.n
# `n = 0` opens before any report exists, so nothing in the progression is
# needed for the gate to open.
required_transitions(t::ReportedCases) = t.n > 0 ? (Reporting,) : ()
# Reports already due stay due: a report written for an earlier episode is
# counted from the archive once the host is reinfected, and the accepted
# clock only rises.
trigger_can_lapse(::ReportedCases) = false

# ── Keyword convenience constructor ──────────────────────────────────

function Scheduled(
        intervention::AbstractIntervention;
        start_time::Union{Float64, Nothing} = nothing,
        end_time::Union{Float64, Nothing} = nothing,
        start_after_cases::Union{Int, Nothing} = nothing,
        start_after::Union{AbstractTrigger, Nothing} = nothing
    )
    conditions = Function[]
    start_time !== nothing && push!(
        conditions,
        s -> s.max_infection_time >= start_time
    )
    end_time !== nothing && push!(
        conditions,
        s -> s.max_infection_time <= end_time
    )
    start_after_cases !== nothing && push!(
        conditions,
        s -> s.cumulative_cases >= start_after_cases
    )
    start_after !== nothing && push!(
        conditions,
        s -> is_triggered!(start_after, s)
    )

    isempty(conditions) && error("Scheduled requires at least one condition")

    condition = if length(conditions) == 1
        conditions[1]
    else
        s -> all(c -> c(s), conditions)
    end
    t = start_time === nothing ? 0.0 : start_time
    population_dependent = start_after_cases !== nothing || start_after !== nothing
    can_lapse = end_time !== nothing ||
        (start_after !== nothing && trigger_can_lapse(start_after))
    triggers = start_after === nothing ? () : (start_after,)
    return Scheduled(
        intervention, condition, t, population_dependent, can_lapse, triggers
    )
end

# Predicate constructor: no individual-level reset. The predicate is opaque, so
# `population_dependent` stays conservatively `true` — see `reads_population_state`.
function Scheduled(intervention::AbstractIntervention, condition)
    return Scheduled(intervention, condition, 0.0, true, true)
end

# ── Protocol delegation ──────────────────────────────────────────────

is_active(s::Scheduled, state::SimulationState) = s.condition(state)

# Whether `s`'s condition holds evaluated as of `t` rather than the clock the
# caller is otherwise running under. `state.max_infection_time` is moved
# there for the check and restored before returning, even if the condition
# throws, since each simulation owns its own state. A population-reading
# condition (a trigger, an opaque predicate) still sees the population as it
# stands right now, only the clock itself moves: the same approximation
# `apply_actions!` already makes for a proposed dose's own time.
function _active_at(s::Scheduled, t, state)
    previous_time = state.max_infection_time
    try
        state.max_infection_time = t
        return is_active(s, state)
    finally
        state.max_infection_time = previous_time
    end
end

# `initialise_individual!` is inherited ungated from `InterventionWrapper`:
# fields must exist before the policy activates.

function resolve_individual!(s::Scheduled, ind, state)
    s.time_aware || is_active(s, state) || return nothing
    resolve_individual!(s.intervention, ind, state)
    s.time_aware && _reset_unless_in_schedule!(s, ind, state)
    return nothing
end

# Kept for wrapping an intervention that still implements the deprecated batch
# hook; the engine hands the batch here only when the wrapped one does.
_uses_batch_hook(s::Scheduled) = _uses_batch_hook(s.intervention)
function apply_post_transmission!(s::Scheduled, state, new_contacts)
    actions = intervention_actions(s, state, new_contacts)
    if actions !== nothing
        _admit_actions!(s, state, actions)
        return nothing
    end
    s.time_aware || is_active(s, state) || return nothing
    apply_post_transmission!(s.intervention, state, new_contacts)
    if s.time_aware
        for c in new_contacts
            _reset_unless_in_schedule!(s, c, state)
        end
    end
    return nothing
end

# Undo the wrapped intervention's effect on `ind` unless it falls inside the
# schedule's window at its own time, exactly as `_action_in_schedule` checks a
# proposed action. An effect that never fired (`intervention_time` still at
# its default) has nothing to undo.
function _reset_unless_in_schedule!(s::Scheduled, ind::Individual, state)
    t = intervention_time(s.intervention, ind)
    isfinite(t) || return nothing
    _active_at(s, t, state) || reset!(s.intervention, ind)
    return nothing
end

# `infectious_removal_time` comes from `InterventionWrapper`, which narrows it
# for a schedule that can close: see `removal_gap_host_times` below. A case
# whose isolation or quarantine was undone by `_reset_unless_in_schedule!`
# above reads back as never isolated, so it is not removed either.

"""
    persistent_competing_risks(intervention) -> Bool

Whether protection already given (for example to a vaccinated person)
continues after a [`Scheduled`](@ref) campaign has stopped. The default is
`false`; vaccination returns `true`, and `Scheduled` and
[`CapacityConstrained`](@ref) answer for the intervention they contain.
`Scheduled` still controls who newly receives the intervention.

For extension authors: an intervention returning `true` must base its
[`competing_risk`](@ref EpiBranch.competing_risk) on the effects it has
recorded and their dates, returning `nothing` before any effect is recorded.
"""
persistent_competing_risks(::AbstractIntervention) = false
persistent_competing_risks(::AbstractVaccination) = true
function persistent_competing_risks(w::InterventionWrapper)
    return persistent_competing_risks(w.intervention)
end

# Whether `iv`'s risk, once it has fired, could later be withdrawn by a
# schedule's active window closing rather than standing for good. Only
# `Scheduled` can withdraw a risk that has already fired, and only when the
# intervention it wraps has not declared `persistent_competing_risks` — a
# vaccination's protection has, since it derives from a recorded dose rather
# than the schedule's own clock, so a `Scheduled` vaccination cannot lapse
# either. Used by the continuous-time race to tell a standing block from one
# that merely happens to be in force right now (see `_standing_risk`).
_may_lapse(::AbstractIntervention) = false
function _may_lapse(s::Scheduled)
    return persistent_competing_risks(s.intervention) ? _may_lapse(s.intervention) : true
end
_may_lapse(w::InterventionWrapper) = _may_lapse(w.intervention)

# A schedule that cannot close again honours the inner removal's releases for
# good, so its stretches can be read back; one with an end withdraws the block
# part-way through a stretch it recorded, which the record cannot express.
# Closing the window withdraws a block before the release it reported, so a
# schedule that can close speaks for none of the inner releases.
binding_release(s::Scheduled) = !s.can_lapse && binding_release(s.intervention)

function removal_gap_host_times(s::Scheduled)
    s.can_lapse && return ()
    return removal_gap_host_times(s.intervention)
end

# A `start_after_cases` count, or a `start_after` trigger, reads state the
# whole population contributes to; a `start_time`/`end_time`-only schedule
# compares against the time of the case being resolved and does not.
function reads_population_state(s::Scheduled)
    return s.population_dependent || reads_population_state(s.intervention)
end

function competing_risk(s::Scheduled, parent, contact, state)
    (persistent_competing_risks(s.intervention) || is_active(s, state)) || return nothing
    return competing_risk(s.intervention, parent, contact, state)
end

# Which contacts the ring grows from, gated by the schedule as the tracing
# hooks are: outside its window the wrapped intervention keeps nobody active.
function keep_active(s::Scheduled, state, targets, is_new)
    is_active(s, state) || return ()
    return keep_active(s.intervention, state, targets, is_new)
end

# Tracing: a time-aware intervention has each contact's own trace checked
# against the schedule at its own time, below, rather than at the
# infector's; one that is not stays gated on the infector's clock, as every
# hook always was.
function trace_contacts!(s::Scheduled, state, infector, contacts, not_before = nothing)
    s.time_aware || is_active(s, state) || return nothing
    if not_before === nothing
        trace_contacts!(s.intervention, state, infector, contacts)
    else
        trace_contacts!(s.intervention, state, infector, contacts, not_before)
    end
    if s.time_aware
        for c in contacts
            _reset_unless_in_schedule!(s, c, state)
        end
    end
    return nothing
end
