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

`Scheduled` is the single entry point for time-based intervention
scheduling. It enforces start times at two levels:

- **Population-level gate** — `is_active(::Scheduled, state)` skips
  `resolve_individual!` and `apply_post_transmission!` until the
  condition returns `true`.
- **Individual-level reset** — after each per-individual hook fires,
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
Scheduled(Isolation(onset_to_isolation_delay=Exponential(2.0)); start_time=14.0)
Scheduled(ContactTracing(probability=0.5, isolation_to_trace_delay=Exponential(1.0)); start_after_cases=50)
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
end

# ── Keyword convenience constructor ──────────────────────────────────

function Scheduled(intervention::AbstractIntervention;
        start_time::Union{Float64, Nothing} = nothing,
        end_time::Union{Float64, Nothing} = nothing,
        start_after_cases::Union{Int, Nothing} = nothing)
    conditions = Function[]
    start_time !== nothing && push!(conditions,
        s -> s.max_infection_time >= start_time)
    end_time !== nothing && push!(conditions,
        s -> s.max_infection_time <= end_time)
    start_after_cases !== nothing && push!(conditions,
        s -> s.cumulative_cases >= start_after_cases)

    isempty(conditions) && error("Scheduled requires at least one condition")

    condition = if length(conditions) == 1
        conditions[1]
    else
        s -> all(c -> c(s), conditions)
    end
    t = start_time === nothing ? 0.0 : start_time
    return Scheduled(intervention, condition, t)
end

# Predicate constructor: no individual-level reset
function Scheduled(intervention::AbstractIntervention, condition)
    Scheduled(intervention, condition, 0.0)
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

# `infectious_removal_time` is inherited ungated: on the continuous-time models
# a Scheduled removes a case when the wrapped intervention does. The loop
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
    persistent_competing_risks(w.intervention)
end

function competing_risk(s::Scheduled, parent, contact, state)
    (persistent_competing_risks(s.intervention) || is_active(s, state)) || return nothing
    return competing_risk(s.intervention, parent, contact, state)
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
