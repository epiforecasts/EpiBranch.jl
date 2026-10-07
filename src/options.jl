"""
    AbstractStoppingRule

A rule for ending a simulation. The run stops as soon as any of its rules
says so; pass them to [`simulate`](@ref) as `stopping_rules`, or use the
`max_cases`, `max_generations` and `max_time` keywords, which create the
first three of these:

- [`MaxCases`](@ref): stop once the outbreak reaches a number of cases.
- [`MaxGenerations`](@ref): stop after a number of generations.
- [`MaxTime`](@ref): stop once infections reach a time, in days.
- [`Extinction`](@ref): stop when transmission has died out. Always included,
  even when you give your own `stopping_rules`.

To write your own rule, define a new type and a method
`EpiBranch.should_stop(rule::MyRule, state::SimulationState)` (the
`EpiBranch.` prefix and the `::SimulationState` are both needed):

```julia
struct MaxChainLength <: AbstractStoppingRule
    n::Int
end
EpiBranch.should_stop(r::MaxChainLength, state::SimulationState) =
    maximum(ind.generation for ind in state.individuals; init = 0) >= r.n
```

The homogeneous, network and household models run until transmission dies
out or a time limit, and do not check `should_stop`. A rule that should also
end those runs defines [`time_bound`](@ref EpiBranch.time_bound), as
[`MaxTime`](@ref) does; if the time limit is all the rule checks, it also
defines [`honoured_without_should_stop`](@ref EpiBranch.honoured_without_should_stop)
so those runs do not warn that it was ignored. The Extending guide has a
worked example.
"""
abstract type AbstractStoppingRule end

# Default termination controls, named once so the public `simulate`
# signatures, the `SimOpts` constructor, and the ignored-control warning share
# a single source of truth (see `_warn_ignored_termination`).
const _DEFAULT_MAX_CASES = 10_000
const _DEFAULT_MAX_GENERATIONS = 100

"""Stop when transmission has died out (no one is left who can still infect
others)."""
struct Extinction <: AbstractStoppingRule end

"""Stop once the outbreak has at least `n` cases in total."""
struct MaxCases <: AbstractStoppingRule
    n::Int
end

"""Stop once `n` generations have been simulated."""
struct MaxGenerations <: AbstractStoppingRule
    n::Int
end

"""Stop once the latest infection is at or after time `t` (days since the
start of the outbreak)."""
struct MaxTime <: AbstractStoppingRule
    t::Float64
end

"""
    should_stop(rule::AbstractStoppingRule, state::SimulationState) -> Bool

Whether `rule` ends the simulation at this point of the outbreak. Define a
method of this function for a new stopping rule. Default: `false`.
"""
should_stop(::AbstractStoppingRule, ::SimulationState) = false
should_stop(::Extinction, state::SimulationState) = state.extinct
should_stop(r::MaxCases, state::SimulationState) = state.cumulative_cases >= r.n
should_stop(r::MaxGenerations, state::SimulationState) = state.current_generation >= r.n
should_stop(r::MaxTime, state::SimulationState) = state.max_infection_time >= r.t

"""
    time_bound(rule::AbstractStoppingRule) -> Real

The time limit (days) a stopping rule sets, or `Inf` if it sets none (the
default). The homogeneous, network and household models do not check
[`should_stop`](@ref); they run until transmission dies out or until this
time. Define it alongside `should_stop` for a rule that, like
[`MaxTime`](@ref), should also end those runs.
"""
time_bound(::AbstractStoppingRule) = Inf
time_bound(r::MaxTime) = r.t

"""
    honoured_without_should_stop(rule::AbstractStoppingRule) -> Bool

Whether the homogeneous, network and household models apply `rule` in full.
They do not check [`should_stop`](@ref) and stop only when transmission dies
out or at a time limit, so they fully apply only [`Extinction`](@ref) and
[`MaxTime`](@ref) (through its [`time_bound`](@ref EpiBranch.time_bound)).
Any other rule keeps the default `false`, and those models warn that it was
ignored. That includes a rule with a time limit that also checks something
else, such as a case count, since only its time limit is applied.
"""
honoured_without_should_stop(::AbstractStoppingRule) = false
honoured_without_should_stop(::Extinction) = true
honoured_without_should_stop(::MaxTime) = true

"""
    SimOpts(; n_initial, initial_cases, max_cases, max_generations, max_time, stopping_rules)

Settings for how a simulation starts (`n_initial` index cases, or the
`initial_cases` IDs) and when it stops. [`simulate`](@ref) takes the same
keywords and builds this for you. Natural history, population
characteristics and interventions are set with a [`ModelSpec`](@ref).

The run stops at the first generation at which any of the
`stopping_rules` ([`AbstractStoppingRule`](@ref)) says so, and always when
transmission dies out ([`Extinction`](@ref)). `max_cases`,
`max_generations` and `max_time` (days) are shortcuts that create the
matching rules; they are ignored when `stopping_rules` is given.

```julia
SimOpts(max_cases = 500)               # [Extinction(), MaxCases(500)]
SimOpts(max_generations = 20, max_time = 90.0)
SimOpts(stopping_rules = [MaxCases(1000), MyCustomRule()])
```

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

"""The case cap set by `opts` (`typemax(Int)` if none), used to flag
outbreaks that reached the cap before dying out."""
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
