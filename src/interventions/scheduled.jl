"""
Wrap an intervention so it only runs when a condition on the simulation
state is met.  Individuals are always initialised (so fields exist before
the policy activates), but `resolve_individual!` and
`apply_post_transmission!` are skipped while the condition returns `false`.

When `start_time` is provided, it is also forwarded to the inner
intervention's own `start_time` field (if it has one), so that
individual-level filtering on action time is handled by the framework's
[`_enforce_start_time!`](@ref) mechanism automatically.

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
"""
struct Scheduled{I <: AbstractIntervention, F} <: AbstractIntervention
    intervention::I
    condition::F
end

# ── Keyword convenience constructor ──────────────────────────────────

function Scheduled(intervention::AbstractIntervention;
        start_time::Union{Float64, Nothing} = nothing,
        end_time::Union{Float64, Nothing} = nothing,
        start_after_cases::Union{Int, Nothing} = nothing)
    # Forward start_time to inner intervention for individual-level filtering
    if start_time !== nothing
        intervention = _with_start_time(intervention, start_time)
    end

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
    return Scheduled(intervention, condition)
end

"""Reconstruct an intervention with `start_time` set, if the type supports it."""
function _with_start_time(intervention::T, t::Float64) where {T <: AbstractIntervention}
    hasfield(T, :start_time) || return intervention
    fields = Dict{Symbol, Any}(fn => getfield(intervention, fn) for fn in fieldnames(T))
    fields[:start_time] = t
    return T(; fields...)
end

# ── Protocol delegation ──────────────────────────────────────────────

is_active(s::Scheduled, state::SimulationState) = s.condition(state)

# Always initialise — fields must exist before the policy activates.
function initialise_individual!(s::Scheduled, ind, state)
    initialise_individual!(s.intervention, ind, state)
end

function resolve_individual!(s::Scheduled, ind, state)
    is_active(s, state) && resolve_individual!(s.intervention, ind, state)
    return nothing
end

function apply_post_transmission!(s::Scheduled, state, new_contacts)
    is_active(s, state) && apply_post_transmission!(s.intervention, state, new_contacts)
    return nothing
end

post_isolation_transmission(s::Scheduled) = post_isolation_transmission(s.intervention)
required_fields(s::Scheduled) = required_fields(s.intervention)
start_time(s::Scheduled) = start_time(s.intervention)
intervention_time(s::Scheduled, ind::Individual) = intervention_time(s.intervention, ind)
reset!(s::Scheduled, ind::Individual) = reset!(s.intervention, ind)
