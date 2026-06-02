"""
Wrap an intervention so it only runs when a condition on the simulation
state is met.  Individuals are always initialised (so fields exist before
the policy activates), but `resolve_individual!` and
`apply_post_transmission!` are skipped while the condition returns `false`.

# Time-based scheduling

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
Scheduled(Isolation(delay=Exponential(2.0)); start_time=14.0)
Scheduled(ContactTracing(probability=0.5); start_after_cases=50)
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
struct Scheduled{I <: AbstractIntervention, F} <: AbstractIntervention
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
function Scheduled(intervention::AbstractIntervention, condition::Function)
    Scheduled(intervention, condition, 0.0)
end

# ── Protocol delegation ──────────────────────────────────────────────

EpiBranchCore.is_active(s::Scheduled, state::SimulationState) = s.condition(state)

# Always initialise — fields must exist before the policy activates.
function EpiBranchCore.initialise_individual!(s::Scheduled, ind, state)
    initialise_individual!(s.intervention, ind, state)
end

function EpiBranchCore.resolve_individual!(s::Scheduled, ind, state)
    is_active(s, state) || return nothing
    resolve_individual!(s.intervention, ind, state)
    _maybe_reset!(s, ind)
    return nothing
end

function EpiBranchCore.apply_post_transmission!(s::Scheduled, state, new_contacts)
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

EpiBranchCore.required_fields(s::Scheduled) = required_fields(s.intervention)
function EpiBranchCore.intervention_time(s::Scheduled, ind::Individual)
    intervention_time(s.intervention, ind)
end
EpiBranchCore.reset!(s::Scheduled, ind::Individual) = reset!(s.intervention, ind)
function EpiBranchCore.competing_risk(s::Scheduled, parent, contact, state)
    is_active(s, state) ? competing_risk(s.intervention, parent, contact, state) : nothing
end
