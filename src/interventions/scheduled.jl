"""
    Scheduled(intervention; start_time = nothing, end_time = nothing,
              start_after_cases = nothing)
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

# Any condition on the simulation, here from the third generation on
Scheduled(iso, state -> state.current_generation >= 3)
```

# Arguments
- `start_time`, `end_time`: days (as decimals, such as `14.0`) between which
  the intervention acts, measured on the simulation clock (the latest
  infection time so far).
- `start_after_cases`: number of cases after which it acts.
- `condition`: a function of the simulation state returning `true` while the
  intervention should act.

Give at least one; several keywords must all hold. A case whose isolation (or
other action) would fall before `start_time` is not acted on, even if the
case itself appears after the start. The `condition` form has no start time
to compare against and does not do this check.

Isolation and quarantine already in place end when the schedule ends (after
`end_time`, or once `condition` returns `false`). The exception is the
network, household and homogeneous models, where a perfect isolation
(`post_isolation_transmission = 0`) or a quarantine with no end
(`duration = Inf`) keeps the case out of transmission for good.
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
as `state.max_infection_time`). Other interventions are skipped while the
schedule is inactive, and an effect whose
[`intervention_time`](@ref EpiBranch.intervention_time) falls before
`start_time` is undone with [`reset!`](@ref EpiBranch.reset!).
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
