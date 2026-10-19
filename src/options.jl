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

The mass-action, network and household models run a continuous-time race
rather than stepping by generation; they poll `should_stop` after each
settled case, for any rule whose effect is not already captured by
[`time_bound`](@ref EpiBranch.time_bound) and
[`generation_bound`](@ref EpiBranch.generation_bound) alone, as
[`MaxTime`](@ref) and [`MaxGenerations`](@ref) are. A rule bounded this way
also defines [`honoured_without_should_stop`](@ref EpiBranch.honoured_without_should_stop).
The `max_cases` and `max_generations` keyword *defaults* have no effect on
these models, so a run over a large fixed population is not capped unless you
ask: give an explicit value, or a `stopping_rules` entry, to end such a run
early. The Extending EpiBranch guide has a worked example in its section on
stopping rules.
"""
abstract type AbstractStoppingRule end

# Default termination controls, named once so the public `simulate`
# signatures, the `SimOpts` constructor, and `_drop_default_shortcuts` share a
# single source of truth.
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

"""Stop a branch once its own infection is at or after time `t` (days since
the start of the outbreak), without drawing it any further contacts. The
generation-based engine (such as [`BranchingProcess`](@ref)) keeps other
branches running while they are still before `t`, which keeps cases before
the limit from being lost even when one branch reaches well past it first.
The mass-action, network and household models settle cases in time order
rather than generation order, so nothing after `t` could have been reached
first; ending the whole run at `t` is enough for them."""
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
default). The mass-action, network and household models end the whole run at
the earliest such limit among its rules, before any candidate infection past
it is drawn. The generation-based engine (such as [`BranchingProcess`](@ref))
instead cuts a branch off from making further contacts once its own infection
reaches the limit, keeping other branches running while they are still before
it. Define it alongside `should_stop` for a rule that, like [`MaxTime`](@ref),
should also get this treatment on those models.
"""
time_bound(::AbstractStoppingRule) = Inf
time_bound(r::MaxTime) = r.t

"""
    generation_bound(rule::AbstractStoppingRule) -> Int

The generation limit a stopping rule sets, or `typemax(Int)` if it sets none
(the default). The mass-action, network and household models settle cases in
time order, not generation order, so `state.current_generation` cannot be
trusted to climb monotonically as [`should_stop`](@ref) is polled; those
models instead drop a settled case's own onward proposals once its generation
reaches this bound. Define it alongside `should_stop` for a rule that, like
[`MaxGenerations`](@ref), should also cap such a run's depth.
"""
generation_bound(::AbstractStoppingRule) = typemax(Int)
generation_bound(r::MaxGenerations) = r.n

"""
    honoured_without_should_stop(rule::AbstractStoppingRule) -> Bool

Whether the mass-action, network and household models apply `rule` in full
through [`time_bound`](@ref EpiBranch.time_bound) and
[`generation_bound`](@ref EpiBranch.generation_bound) alone, with no need to
poll [`should_stop`](@ref) after each settled case. True for
[`Extinction`](@ref), [`MaxTime`](@ref) and [`MaxGenerations`](@ref). Any
other rule keeps the default `false`, and those models poll `should_stop` for
it instead — including a rule that bounds time or generation depth but also
tests something else, such as a case count, since only the bound is captured
without asking.

A rule with a finite [`time_bound`](@ref EpiBranch.time_bound) that also
returns `true` here skips `should_stop` on the generation-based engine (such
as [`BranchingProcess`](@ref)) too: that engine never polls `should_stop` for
it, relying instead on the per-branch time cut-off described under
`time_bound` to end the run.
"""
honoured_without_should_stop(::AbstractStoppingRule) = false
honoured_without_should_stop(::Extinction) = true
honoured_without_should_stop(::MaxTime) = true
honoured_without_should_stop(::MaxGenerations) = true

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
SimOpts(max_cases = 500)   # [Extinction(), MaxCases(500), MaxGenerations(100)]
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
