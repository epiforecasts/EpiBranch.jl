"""
    AbstractStoppingRule

A rule that decides whether the simulation should terminate at the
current step. Subtypes implement
[`should_stop(rule, state)`](@ref) returning `Bool`; the engine stops
when *any* rule returns `true`. The default implementation returns
`false`, so user-defined rules only need to override the truthy cases.

Built-in rules:

- [`Extinction`](@ref) — stop when no active individuals remain
  (always included in `SimOpts` unless explicitly overridden).
- [`MaxCases`](@ref) — stop when cumulative cases reach a cap.
- [`MaxGenerations`](@ref) — stop after a maximum number of
  generations.
- [`MaxTime`](@ref) — stop when the maximum infection time crosses a
  threshold.

User extensions are a single method:

```julia
struct MaxChainLength <: AbstractStoppingRule
    n::Int
end
should_stop(r::MaxChainLength, state) =
    maximum(ind.generation for ind in state.individuals; init = 0) >= r.n
```
"""
abstract type AbstractStoppingRule end

# Default termination controls, named once so the public `simulate`
# signatures, the `SimOpts` constructor, and the ignored-control warning share
# a single source of truth (see `_warn_ignored_termination`).
const _DEFAULT_MAX_CASES = 10_000
const _DEFAULT_MAX_GENERATIONS = 100

"""Stop when the simulation has gone extinct (no active individuals)."""
struct Extinction <: AbstractStoppingRule end

"""Stop when `state.cumulative_cases >= n`."""
struct MaxCases <: AbstractStoppingRule
    n::Int
end

"""Stop when `state.current_generation >= n`."""
struct MaxGenerations <: AbstractStoppingRule
    n::Int
end

"""Stop when `state.max_infection_time >= t`."""
struct MaxTime <: AbstractStoppingRule
    t::Float64
end

"""
    should_stop(rule::AbstractStoppingRule, state::SimulationState) -> Bool

Whether this rule wants the simulation to terminate given the current
state. Default: `false`.
"""
should_stop(::AbstractStoppingRule, ::SimulationState) = false
should_stop(::Extinction, state::SimulationState) = state.extinct
should_stop(r::MaxCases, state::SimulationState) = state.cumulative_cases >= r.n
should_stop(r::MaxGenerations, state::SimulationState) = state.current_generation >= r.n
should_stop(r::MaxTime, state::SimulationState) = state.max_infection_time >= r.t

"""
Options controlling simulation termination and setup. Contains only
simulation control parameters — clinical and demographic properties
are set via `attributes` functions, [`AbstractClinicalTransition`](@ref)s,
and interventions.

Termination is controlled by `stopping_rules`, a vector of
[`AbstractStoppingRule`](@ref); the simulation stops at the first step
for which any rule returns `true`. The keyword constructor accepts
ergonomic shortcuts (`max_cases`, `max_generations`, `max_time`) that
build the corresponding rules and prepend [`Extinction`](@ref):

```julia
SimOpts(max_cases = 500)               # [Extinction(), MaxCases(500)]
SimOpts(max_generations = 20, max_time = 90.0)
SimOpts(stopping_rules = [MaxCases(1000), MyCustomRule()])
```

For finer control or custom rules, pass `stopping_rules` directly.
"""
struct SimOpts
    n_initial::Int
    initial_cases::Union{Nothing, Vector{Int}}
    stopping_rules::Vector{AbstractStoppingRule}
end

function SimOpts(;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing)
    initial_cases !== nothing && n_initial !== nothing &&
        throw(ArgumentError(
            "provide either initial_cases or n_initial, not both"))
    ids = initial_cases === nothing ? nothing : collect(Int, initial_cases)
    if ids !== nothing
        all(>(0), ids) || throw(ArgumentError("initial_cases must contain positive IDs"))
        allunique(ids) || throw(ArgumentError("initial_cases must contain distinct IDs"))
    end
    count = ids === nothing ? something(n_initial, 1) : length(ids)
    if stopping_rules !== nothing
        # Extinction is always included (per its docstring) unless the user
        # supplied their own, so a custom rule set can't loop forever on an
        # outbreak that goes extinct below the cap. `collect` makes a fresh
        # vector, leaving the caller's untouched.
        rules = collect(AbstractStoppingRule, stopping_rules)
        any(r -> r isa Extinction, rules) || pushfirst!(rules, Extinction())
        return SimOpts(count, ids, rules)
    end
    rules = AbstractStoppingRule[Extinction()]
    max_cases !== nothing && push!(rules, MaxCases(max_cases))
    max_generations !== nothing && push!(rules, MaxGenerations(max_generations))
    max_time !== nothing && push!(rules, MaxTime(Float64(max_time)))
    return SimOpts(count, ids, rules)
end

"""Extract the `MaxCases` cap from `opts` (or `typemax(Int)` if absent).
Used by analytical helpers that need to know the cap to flag outbreaks
that hit it before going extinct."""
function _case_cap(opts::SimOpts)
    for rule in opts.stopping_rules
        rule isa MaxCases && return rule.n
    end
    return typemax(Int)
end

# Preserve the positional constructor used by external simulation methods.
function SimOpts(n_initial::Int, rules::Vector{AbstractStoppingRule})
    SimOpts(n_initial, nothing, rules)
end

function _validate_initial_cases(model::TransmissionModel, opts::SimOpts)
    opts.initial_cases === nothing || throw(ArgumentError(
        "$(nameof(typeof(model))) does not support initial_cases"))
    return nothing
end

function _validate_initial_case_ids(opts::SimOpts, n, external_hazard)
    ids = opts.initial_cases
    ids === nothing && return nothing
    all(id -> id <= n, ids) || throw(ArgumentError(
        "initial_cases IDs must be in 1:$n"))
    _ext_active(external_hazard) && throw(ArgumentError(
        "initial_cases cannot yet be combined with external_hazard; use " *
        "chosen initial cases with external_hazard = 0"))
    return nothing
end

# Candidate times are local to a race; chosen IDs refer to the whole population.
function _seed_initial_cases!(best, members, ids)
    chosen = Set(ids)
    for (k, id) in enumerate(members)
        id in chosen && (best[k] = 0)
    end
    return nothing
end
