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

Any combination of `start_time`, `end_time`, and `start_after_cases` is
accepted.  They are combined with `&&`:

```julia
Scheduled(Isolation(onset_to_isolation_delay=Exponential(2.0), duration=7.0); start_time=14.0)
Scheduled(ContactTracing(probability=0.5, isolation_to_trace_delay=Exponential(1.0), action=Quarantine(duration=7.0)); start_after_cases=50)
Scheduled(iso; start_time=10.0, end_time=30.0)
```

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
    # Whether the active window can close again once open. Every condition the
    # keyword constructor builds is monotone in the state except `end_time`, so
    # only that one can withdraw an effect already delivered. An opaque
    # predicate is assumed to.
    can_lapse::Bool
end

# ── Keyword convenience constructor ──────────────────────────────────

function Scheduled(
        intervention::AbstractIntervention;
        start_time::Union{Float64, Nothing} = nothing,
        end_time::Union{Float64, Nothing} = nothing,
        start_after_cases::Union{Int, Nothing} = nothing
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

    isempty(conditions) && error("Scheduled requires at least one condition")

    condition = if length(conditions) == 1
        conditions[1]
    else
        s -> all(c -> c(s), conditions)
    end
    t = start_time === nothing ? 0.0 : start_time
    return Scheduled(
        intervention, condition, t, start_after_cases !== nothing,
        end_time !== nothing
    )
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

# A count-gated `start_after_cases` reads the running case count, a
# population-wide read; a `start_time`/`end_time`-only schedule compares
# against the time of the case being resolved and is not.
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
