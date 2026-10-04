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

User extensions are a single method, qualified with `EpiBranch.` (or
reached via `import EpiBranch: should_stop`) so it adds to this function
rather than shadowing it with a new one of the same name, and with
`state` typed `::SimulationState` so it doesn't clash with the default
method above:

```julia
struct MaxChainLength <: AbstractStoppingRule
    n::Int
end
EpiBranch.should_stop(r::MaxChainLength, state::SimulationState) =
    maximum(ind.generation for ind in state.individuals; init = 0) >= r.n
```

The structure-driven (Sellke) models run to extinction or a time bound
rather than stepping through `should_stop` each generation; a rule that
should also be able to end such a run overrides
[`time_bound`](@ref EpiBranch.time_bound), as [`MaxTime`](@ref) does. When
reaching that bound is the whole of what the rule tests, it also declares
[`honoured_without_should_stop`](@ref EpiBranch.honoured_without_should_stop),
so such a run does not report it as ignored.
See the Extending guide for a worked example.
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
    time_bound(rule::AbstractStoppingRule) -> Real

The latest infection time at which `rule` could still want the simulation to
continue, or `Inf` if the rule places no bound on time. The continuous-time
(Sellke) models run over a fixed population to extinction or this bound,
rather than stepping through `should_stop` each generation, so they read this
trait instead of enumerating the known stopping-rule subtypes. Override it
alongside `should_stop` for a rule that, like [`MaxTime`](@ref), should be
able to end such a run; the default `Inf` leaves it unaffected. Default:
`Inf`.
"""
time_bound(::AbstractStoppingRule) = Inf
time_bound(r::MaxTime) = r.t

"""
    honoured_without_should_stop(rule::AbstractStoppingRule) -> Bool

Whether a run that never consults [`should_stop`](@ref) still applies `rule` in
full. The structure-driven (Sellke) models end at extinction or at a time bound
instead of stepping through `should_stop` each generation, so they apply a rule
that asks for nothing more: [`Extinction`](@ref), and [`MaxTime`](@ref) through
its [`time_bound`](@ref EpiBranch.time_bound). A rule that tests anything else
keeps the default `false` and such a run reports it as ignored, including a
rule that declares a time bound and tests a case count as well, since only its
bound is applied. Declaring a time bound is therefore not on its own grounds
for answering `true`. Default: `false`.
"""
honoured_without_should_stop(::AbstractStoppingRule) = false
honoured_without_should_stop(::Extinction) = true
honoured_without_should_stop(::MaxTime) = true

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
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing
    )
    initial_cases !== nothing && n_initial !== nothing &&
        throw(
        ArgumentError(
            "provide either initial_cases or n_initial, not both"
        )
    )
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
function SimOpts(n_initial, rules)
    return SimOpts(n_initial, nothing, rules)
end

function _validate_initial_cases(model::TransmissionModel, opts::SimOpts)
    opts.initial_cases === nothing || throw(
        ArgumentError(
            "$(nameof(typeof(model))) does not support initial_cases"
        )
    )
    return nothing
end

function _validate_initial_case_ids(opts::SimOpts, n)
    ids = opts.initial_cases
    ids === nothing && return nothing
    all(id -> id <= n, ids) || throw(
        ArgumentError(
            "initial_cases IDs must be in 1:$n"
        )
    )
    return nothing
end

# Candidate times are local to a race; chosen IDs refer to the whole population.
_seed_initial_cases!(best, members, ids) = _seed_initial_cases!(best, members, Set(ids))

function _seed_initial_cases!(best, members, chosen::AbstractSet)
    for (k, id) in enumerate(members)
        id in chosen && (best[k] = 0)
    end
    return nothing
end
