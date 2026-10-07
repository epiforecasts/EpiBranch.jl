# ── Infectiousness window ────────────────────────────────────────────

"""
    Infectiousness(offspring; from = :infection, until = (), kernel = NoGenerationTime())

When, and how much, a case transmits by one route. `offspring` is the
distribution of the number of people infected by this route; transmission
starts when the case reaches state `from` and stops at the earliest of the
`until` states. Give a [`BranchingProcess`](@ref) several of these for
different routes, such as community and funeral transmission of Ebola.

- `offspring`: a distribution of secondary cases, such as `NegBin(R, k)`
  (mean `R`, dispersion `k`), or a function `(rng, ind) -> n` returning the
  number infected by a given case (one count per type in a multi-type
  model). The function may also take the outbreak state as a third argument
  (see [`draw_offspring`](@ref EpiBranch.draw_offspring)).
- `from`: the state at which transmission starts, `:infection` by default.
  Any other name refers to a state set by a step of the natural history, such
  as `:infectious` or `:died`; transmission starts only once the case reaches
  it.
- `until`: the states at which transmission stops, for example
  `(:recovered, :died)` for community transmission or `(:buried,)` for a
  funeral. The earliest one reached ends transmission, and infections that
  would have happened later do not. Empty by default. Isolation is set with
  the [`Isolation`](@ref) intervention, not here.
- `kernel`: the time in days from `from` to each infection: a distribution,
  a function of the individual returning a distribution, `(ind) -> ...`, or
  `NoGenerationTime()` to place every infection at the `from` time. With the
  default `from = :infection` and no `until`, this is the generation time.
  With `until` it is the contact interval (the time to a contact that would
  infect if nothing stopped it), and the generation time follows from which
  comes first, the contact or the end of transmission, so do not also
  shorten it by hand.

# Examples
```julia
using EpiBranch, Distributions

progression = [
    Transition(:infectious, from = :infection, delay = Gamma(4.0, 2.0)),
    Transition(:died, from = :infectious, delay = Gamma(4.0, 2.0),
        probability = 0.6, terminal = true),
    Transition(:recovered, from = :infectious, delay = Gamma(5.0, 2.0),
        terminal = true),
    Transition(:buried, from = :died, delay = 2.0),
]
community = Infectiousness(NegBin(1.2, 0.5);
    from = :infectious, until = (:recovered, :died), kernel = Exponential(4.0))
funeral = Infectiousness(Poisson(0.5);
    from = :died, until = (:buried,), kernel = Uniform(0.0, 2.0))
model = ModelSpec(BranchingProcess(community, funeral); progression)
```
"""
struct Infectiousness{O, F, U, K}
    offspring::O
    from::F
    until::U
    kernel::K
end
function Infectiousness(
        offspring; from = :infection, until = (),
        kernel = NoGenerationTime()
    )
    return Infectiousness(offspring, from, until, kernel)
end

# ── BranchingProcess type and constructors ──────────────────────────

"""
    BranchingProcess(offspring, generation_time; population_size)
    BranchingProcess(offspring)
    BranchingProcess(windows::Infectiousness...)

A stochastic branching process: each case independently infects a random
number of others, drawn from the offspring distribution `offspring`, and the
outbreak grows as a tree from the index cases. A negative binomial,
`NegBin(R, k)` with mean `R` and dispersion `k`, is the usual choice; smaller
`k` means more superspreading.

`generation_time` is the distribution of the time in days from a case's
infection to the infection of each of its secondary cases. Leave it out to
study only chain sizes and lengths, without timing. `population_size` limits
the number of people who can be infected (unlimited by default).

The transmission model describes only who infects whom and when. Add the
natural history, interventions, population characteristics and reporting
with a [`ModelSpec`](@ref).

# Examples

```julia
# R = 2.5, k = 0.16; generation time log-normal with log-mean 1.6 and
# log-sd 0.5 (mean about 5.6 days)
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

# no timing: chain sizes and lengths only
BranchingProcess(NegBin(0.8, 0.5))

# two types (e.g. children and adults): M[i, j] is the mean number of
# type-i cases infected by one type-j case
M = [1.2 0.4; 0.3 0.9]
BranchingProcess(M, R -> NegBin(R, 0.16), LogNormal(1.6, 0.5))

# add isolation with a ModelSpec
ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)])
```

For transmission by several routes, or starting and stopping at points in
the natural history, build the process from [`Infectiousness`](@ref)
windows. The forms above make a single one that starts at infection.
"""
struct BranchingProcess{W <: Tuple, P, L} <: TransmissionModel
    infectiousness::W
    population_size::P
    n_types::Int
    type_labels::L
end

# Cross-tier check, run when a process is composed with a progression in a
# [`ModelSpec`](@ref): warn when an infectiousness window opens at a `from`
# state the progression never produces, so the window would silently never
# open. `from` may legitimately be a state written by an attribute or
# intervention rather than a progression transition, so this is a warning.
# A no-op for models without infectiousness windows.
_validate_process_windows(::TransmissionModel, progression) = nothing
function _validate_process_windows(m::BranchingProcess, progression)
    return _validate_windows(m.infectiousness, progression)
end
function _validate_windows(windows, progression)
    produced = Set{Symbol}((:infection,))
    for t in progression
        hasproperty(t, :state) && push!(produced, t.state::Symbol)
    end
    for w in windows
        w.from isa Symbol || continue
        (w.from === :infection || w.from in produced) && continue
        @warn "Infectiousness window has from = :$(w.from), which no progression " *
            "transition produces; the window only opens if something sets " *
            ":$(Symbol(w.from, :_time)) (an attribute, intervention, or transition)."
    end
    return nothing
end

# Shared by the continuous-time structure-driven models (`HomogeneousProcess`,
# `NetworkProcess`, `HouseholdProcess`, `RouteWindow`): a case's window closes
# at the earliest `Symbol(s, :_time)` for `s in until` (`_window_close` in
# sellke.jl). A progression's terminal transition writes its own
# `state => true`/`state_time` pair regardless of whether `state` is in
# `until`, so a state missing from `until` is simply never consulted: a case
# reaching it keeps generating exposure proposals as if still infectious.
# `terminal_target` (defined per transition type, alongside `is_terminal`)
# gives the state label without needing an individual to resolve a time from —
# `Transition` reads it off `.state`, `Death`/`Recovery` are hardcoded to
# :died/:recovered. A terminal transition that does not implement it stays at
# the default `nothing`, so it is not checkable here and stays silently
# exempt — documented on `terminal_target`'s own docstring, since a custom
# terminal transition must opt in to be covered. `from`, when it names a
# terminal state itself (e.g. a funeral `RouteWindow` with `from = :died`),
# is excluded too: a window that only opens once a case reaches that state
# cannot sensibly be asked to also close on it.
function _uncovered_terminal_states(until::Tuple, progression; from = nothing)
    covered = Set{Symbol}(until)
    from isa Symbol && push!(covered, from)
    states = Symbol[]
    for t in progression
        is_terminal(t) || continue
        target = terminal_target(t)
        target === nothing && continue
        target in covered || push!(states, target)
    end
    return unique(states)
end

# Warn once (per `ModelSpec`) when `until` does not cover every terminal state
# the progression can reach, so the silent runaway is discoverable instead of
# only showing up as an implausibly large outbreak. `route` labels the warning
# when `until` belongs to one window among several (a `RouteWindow`); `from`
# is that window's own opening state, excluded from the check (see above).
function _warn_uncovered_terminal_states(
        until::Tuple, progression;
        route = nothing, from = nothing
    )
    states = _uncovered_terminal_states(until, progression; from)
    isempty(states) && return nothing
    on_route = route === nothing ? "" : " on route :$route"
    @warn "Progression has a terminal transition to " *
        "$(join((":" * String(s) for s in states), ", ")), which `until`" *
        "$on_route $until does not list. A case reaching it never has its " *
        "infectious/exposure window closed, and keeps generating exposure " *
        "proposals indefinitely. Add it to `until` if it should end " *
        "transmission."
    return nothing
end

# Natural history, interventions, attributes and observation are not carried by
# the process — they are composed onto it with a [`ModelSpec`](@ref). The
# shared accessors (in model_inputs.jl and model_spec.jl) resolve to empty
# defaults for a bare process, so `_progvec` stays here for the spec to use.
_progvec(p) = convert(Vector{AbstractClinicalTransition}, p)

population_size(m::BranchingProcess) = m.population_size
n_types(m::BranchingProcess) = m.n_types

function single_type_offspring(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(
        ArgumentError(
            "Analytical helpers need a single infectiousness window (this model has " *
                "$(length(m.infectiousness))). The offspring law across several windows is a " *
                "fate-mixture with no closed form, so use simulation for multi-window models."
        )
    )
    return _single_type(m.infectiousness[1].offspring)
end

# The single-type law of an offspring specification, or an error for the kinds
# that have none: an offspring function here, and `MultiTypeOffspring` in
# multi_type_offspring.jl.
_single_type(off) = off
function _single_type(::Function)
    throw(
        ArgumentError(
            "This function only works with single-type models (not multi-type function offspring)"
        )
    )
end

# The contact interval of a single-window model (used by analytical
# helpers that assume one generation-time distribution).
function _single_kernel(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(
        ArgumentError(
            "this analytical helper needs a single infectiousness window; this model has $(length(m.infectiousness))"
        )
    )
    return m.infectiousness[1].kernel
end

_offspring_label(off::Distribution) = string(typeof(off))
_offspring_label(off) = "Function"

function Base.show(io::IO, m::BranchingProcess)
    pop_str = m.population_size isa NoPopulation ? "unlimited" : string(m.population_size)
    return if length(m.infectiousness) == 1
        w = m.infectiousness[1]
        off_str = _offspring_label(w.offspring)
        gt_str = w.kernel isa NoGenerationTime ? "none" :
            w.kernel isa Distribution ? string(typeof(w.kernel)) : "Function"
        print(
            io,
            "BranchingProcess(offspring=$(off_str), generation_time=$(gt_str), population_size=$(pop_str))"
        )
    else
        print(
            io,
            "BranchingProcess($(length(m.infectiousness)) infectiousness windows, population_size=$(pop_str))"
        )
    end
end

# Single-type with a contact interval (one default window).
function BranchingProcess(
        offspring::Distribution, gt;
        population_size::Union{Int, NoPopulation} = NoPopulation()
    )
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),), population_size, 1,
        NoTypeLabels()
    )
end

# Single-type without a contact interval (pure chain statistics).
function BranchingProcess(
        offspring::Distribution;
        population_size::Union{Int, NoPopulation} = NoPopulation()
    )
    return BranchingProcess((Infectiousness(offspring),), population_size, 1, NoTypeLabels())
end

# Multi-type with an explicit offspring function.
function BranchingProcess(
        offspring, gt;
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels()
    )
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),), population_size, n_types,
        type_labels
    )
end

# Explicit windows: pass `Infectiousness` windows directly.
function BranchingProcess(
        windows::Tuple{Infectiousness, Vararg{Infectiousness}};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels()
    )
    return BranchingProcess(windows, population_size, n_types, type_labels)
end
function BranchingProcess(window::Infectiousness, windows::Infectiousness...; kwargs...)
    return BranchingProcess((window, windows...); kwargs...)
end

# ── Offspring generation ─────────────────────────────────────────────

"""
    generate_offspring(model::BranchingProcess, parent, state)

Draw how many people the case `parent` infects: one number, or one number
per type in a multi-type model. Only defined for a process with a single
[`Infectiousness`](@ref) window; the simulation handles several windows
through [`collect_exposures`](@ref).
"""
function generate_offspring(model::BranchingProcess, parent, state)
    length(model.infectiousness) == 1 || throw(
        ArgumentError(
            "generate_offspring is defined for a single infectiousness window; use collect_exposures for multi-window models"
        )
    )
    return draw_offspring(state.rng, model.infectiousness[1].offspring, parent, state)
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Draw the number of secondary cases of one case from an offspring
distribution."""
function draw_offspring(
        rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState
    )
    return rand(rng, offspring)
end

"""Draw the number of secondary cases of one case from an offspring
function, written either as `(rng, individual) -> n` or as
`(rng, individual, state) -> n`. The second form can read the state of the
whole outbreak, for example to reduce transmission once the case count
passes a threshold."""
function draw_offspring(rng, offspring, individual, state)
    if applicable(offspring, rng, individual, state)
        return offspring(rng, individual, state)
    end
    return offspring(rng, individual)
end
