"""
Wrap an intervention in a condition on the simulation state. Individuals are
always initialised, so fields exist before the policy activates. Scheduling
gates delivery; effects declaring `persistent_competing_risks` retain their
recorded protection while the delivery policy is inactive.

# Time-based scheduling

For interventions implementing `intervention_actions`, scheduling tests each
proposed action time before delivery. Predicates see that time as
`state.max_infection_time`; case counts and generation remain at discovery.
Capacity admission uses the original simulation clock in either wrapper order.
The batch-hook behaviour below applies to interventions without this protocol.

A schedule built with an `end_time`, or from a predicate, can withdraw a block
it has already delivered, which one per-host record of a removal's stretches
cannot express. On the continuous-time models the per-contact risk re-checks
the schedule at every proposal regardless; a wrapped `Isolation`'s `duration`
or a `Quarantine`'s still hands the case back on its own terms. Only a removal
with no release of its own (`duration = Inf`) closes the infectious window at
its start, since there is nothing left for the wrapper's own withdrawal to
hand back either way.

`Scheduled` is the single entry point for time-based intervention
scheduling. It enforces start times at two levels:

- **Population-level gate** — `is_active(::Scheduled, state)` skips
  `resolve_individual!` and `apply_post_transmission!` until the
  condition returns `true`.
- **Individual-level reset** — after each per-individual hook runs,
  `Scheduled` checks whether the individual's `intervention_time` falls
  before `start_time` and, if so, calls `reset!` to undo the effect.
  This handles the case where the population gate has opened but a
  specific individual's sampled action time would still fall before the
  policy began (e.g. an isolation date computed from `onset + delay`
  that lands pre-policy).

Individual interventions therefore no longer carry a `start_time` field
of their own — wrap them with `Scheduled` to schedule them in time.

# Keyword constructor

Any combination of `start_time`, `end_time`, `start_after_cases`, and
`start_after` is accepted. They are combined with `&&`:

```julia
Scheduled(Isolation(onset_to_isolation_delay=Exponential(2.0), duration=7.0); start_time=14.0)
Scheduled(ContactTracing(probability=0.5, isolation_to_trace_delay=Exponential(1.0), action=Quarantine(duration=7.0)); start_after_cases=50)
Scheduled(iso; start_time=10.0, end_time=30.0)
Scheduled(iso; start_after=ReportedCases(20))
```

`start_after_cases = n` is shorthand for `start_after = Infections(n)`: it
counts every infection as soon as it occurs, including one never reported
or reported only after the date it opens on. [`ReportedCases`](@ref) counts
only reports dated by the simulation clock instead, which is what a
reactive policy triggered by surveillance needs. `start_after` takes any
[`AbstractTrigger`](@ref EpiBranch.AbstractTrigger), so a user-defined count
works the same way without editing `Scheduled`.

# Predicate constructor

Pass any `f(::SimulationState) -> Bool`:

```julia
Scheduled(iso, state -> state.current_generation >= 3)
```

The predicate form does not perform per-individual reset (there is no
`start_time` to compare against). Use the keyword form when you need
that behaviour.
"""
struct Scheduled{I <: AbstractIntervention, F} <: InterventionWrapper
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

Whether `trigger`'s condition holds now. The same trigger object is reused
across the repeated simulations of `simulate(model, n)` (serially, or
across threads under `parallel = true`), so a trigger must keep no state
of its own between calls: derive what it needs from `state`, or keep working
values in `state.scratch`, which belongs to the run.

The engine asks the gate about 1.7 times per individual, so a trigger that
walks the population on every call makes a run quadratic in the number of
individuals, as [`ReportedCases`](@ref) does. A trigger that can read a
running total the engine already keeps, as [`Infections`](@ref) reads
`cumulative_cases`, costs nothing by comparison.
"""
function is_triggered! end

"""
    trigger_can_lapse(trigger) -> Bool

Whether `trigger`'s condition, once true, can turn false again. The
default is the conservative `true`: a trigger written outside the package
is assumed able to close again until it says otherwise, as an opaque
[`Scheduled`](@ref) predicate already is. [`Infections`](@ref) returns
`false`, since `cumulative_cases` only ever rises. [`ReportedCases`](@ref)
keeps the default: the report count only rises, but the clock it is counted
against does not, since admitting a scheduled action asks the gate with the
action's own time in place of the simulation clock.
"""
trigger_can_lapse(::AbstractTrigger) = true

"""
    Infections(n)

A [`Scheduled`](@ref) `start_after` trigger that opens once `n` infections
have occurred: every case, as soon as it is created, including one never
reported or reported only after the date it opens on. Equivalent to the
`start_after_cases` keyword.
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
# In between, the newest episode and the live state are one report, so the
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
# episode carrying it, and a report the health system has made does not stop
# having been made.
#
# Derived from `state.individuals` rather than kept as a running total, since
# `make_contact!` appends an individual before its clinical transitions run
# and `Reporting` is one of those, so a host counted on an earlier call
# cannot be taken as finished with.
#
# Counted on every call, and the engine asks the gate about 1.7 times per
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

A [`Scheduled`](@ref) `start_after` trigger that opens once `n` cases have
been reported *as of the simulation clock* (`state.max_infection_time`):
unlike [`Infections`](@ref), a case whose `:reporting_time` lies after the
current time does not count yet. Needs a `Reporting` transition in the
model's `progression`, which sets `:reported` and `:reporting_time`.
"""
struct ReportedCases <: AbstractTrigger
    n::Int
end
is_triggered!(t::ReportedCases, state::SimulationState) = _reported_cases_due(state) >= t.n

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
    return Scheduled(intervention, condition, t, population_dependent, can_lapse)
end

# Predicate constructor: no individual-level reset. The predicate is opaque, so
# `population_dependent` stays conservatively `true` — see `reads_population_state`.
function Scheduled(intervention::AbstractIntervention, condition)
    return Scheduled(intervention, condition, 0.0, true, true)
end

# ── Protocol delegation ──────────────────────────────────────────────

is_active(s::Scheduled, state::SimulationState) = s.condition(state)

# `initialise_individual!` is inherited ungated from `InterventionWrapper`:
# fields must exist before the policy activates.

function resolve_individual!(s::Scheduled, ind, state)
    is_active(s, state) || return nothing
    resolve_individual!(s.intervention, ind, state)
    _maybe_reset!(s, ind)
    return nothing
end

function apply_post_transmission!(s::Scheduled, state, new_contacts)
    actions = intervention_actions(s, state, new_contacts)
    if actions !== nothing
        _admit_actions!(s, state, actions)
        return nothing
    end
    is_active(s, state) || return nothing
    apply_post_transmission!(s.intervention, state, new_contacts)
    for c in new_contacts
        _maybe_reset!(s, c)
    end
    return nothing
end

# Individual-level reset if the action time falls before policy start.
@inline function _maybe_reset!(s::Scheduled, ind::Individual)
    s.start_time <= 0.0 && return nothing
    intervention_time(s.intervention, ind) < s.start_time &&
        reset!(s.intervention, ind)
    return nothing
end

# `infectious_removal_time` comes from `InterventionWrapper`, which narrows it
# for a schedule that can close: see `removal_gap_host_times` below. The loop
# resolves it against the running clock, so a case whose infection time is
# before `start_time` never has its wrapped intervention run (its gate is
# closed) and so is not removed.

"""
    persistent_competing_risks(intervention) -> Bool

Whether recorded intervention effects continue when a delivery schedule is
inactive. The default is `false`; vaccination returns `true`, and wrappers
delegate to their inner intervention. An extension opting in must derive its
risks from recorded effects and their dates, returning `nothing` before any
effect has been recorded. `Scheduled` still controls delivery and other hooks.
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

# Tracing on the continuous-time path, gated by the schedule exactly as
# `apply_post_transmission!` is on the generation-based one.
function trace_contacts!(s::Scheduled, state, infector, contacts, not_before = nothing)
    is_active(s, state) || return nothing
    if not_before === nothing
        trace_contacts!(s.intervention, state, infector, contacts)
    else
        trace_contacts!(s.intervention, state, infector, contacts, not_before)
    end
    for c in contacts
        _maybe_reset!(s, c)
    end
    return nothing
end
