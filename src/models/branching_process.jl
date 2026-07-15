# ── Infectiousness window ────────────────────────────────────────────

"""
    Infectiousness(offspring; from = :infection, until = (), kernel = NoGenerationTime())

A transmission window on a case's timeline: a source of `offspring`
contacts that becomes active at the `from` state, times each contact by
`kernel` measured from `from`, and is ended by the earliest of the
`until` states.

- `from` names a state (`:infection`, the default, or a state a
  transition wrote, e.g. `:infectious`, `:died`). The window contributes
  contacts only once that state has been reached.
- `until` is a tuple of state names whose earliest occurrence censors the
  window: `(:recovered, :died)` for community spread, `(:buried,)` for a
  funeral window. Each name `s` resolves to the infector's
  `Symbol(s, :_time)`. Empty by default (no censoring). Isolation censoring
  comes from the `Isolation` intervention, not from a state here.
- `kernel` is the contact interval, measured from `from`: a
  `Distribution`, a `Function (ind) -> Distribution`, or
  `NoGenerationTime()` (contacts land at the `from` time).
- `offspring` is the window's own draw: a `Distribution` (single-type), a
  `Function (rng, ind[, state]) -> Int` / `-> Vector{Int}` (multi-type),
  or any spec [`draw_offspring`](@ref) accepts.

Several windows on one [`BranchingProcess`](@ref) (community, funeral)
each carry their own offspring, timing, and censoring.
"""
struct Infectiousness{O, F, U, K}
    offspring::O
    from::F
    until::U
    kernel::K
end
function Infectiousness(offspring; from = :infection, until = (),
        kernel = NoGenerationTime())
    return Infectiousness(offspring, from, until, kernel)
end

# ── BranchingProcess type and constructors ──────────────────────────

"""
Stochastic branching process transmission model.

Each infected case independently generates a random number of secondary
cases — its offspring — and the outbreak is the branching tree that grows
from the seeds. The first positional argument is that offspring
distribution; a negative binomial is the usual choice, its overdispersion
standing in for superspreading. The second positional argument is the
contact interval (generation-time distribution); with the default single
window and no natural-history states it is the generation interval, exactly
as before.

The process is a pure transmission kernel: the modelling layers
(progression, interventions, attributes, observation) are attached with a
[`ModelSpec`](@ref), not on the constructor.

# Examples

```julia
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
BranchingProcess(NegBin(0.8, 0.5))  # no timing, pure chain statistics
BranchingProcess(M, R_j -> NegBin(R_j, 0.16), LogNormal(1.6, 0.5))  # multi-type

# attach a policy via a ModelSpec
ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0))])
```

Transmission is a tuple of [`Infectiousness`](@ref) windows; the
convenience constructors above build a single default window with
`from = :infection` and no censoring.
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
    _validate_windows(m.infectiousness, progression)
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

# Natural history, interventions, attributes and observation are not carried by
# the process — they are composed onto it with a [`ModelSpec`](@ref). The
# shared accessors (in model_inputs.jl and model_spec.jl) resolve to empty
# defaults for a bare process, so `_progvec` stays here for the spec to use.
_progvec(p) = convert(Vector{AbstractClinicalTransition}, p)

population_size(m::BranchingProcess) = m.population_size
n_types(m::BranchingProcess) = m.n_types

function single_type_offspring(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(ArgumentError(
        "Analytical helpers need a single infectiousness window (this model has " *
        "$(length(m.infectiousness))). The offspring law across several windows is a " *
        "fate-mixture with no closed form, so use simulation for multi-window models."))
    off = m.infectiousness[1].offspring
    off isa Function && throw(ArgumentError(
        "This function only works with single-type models (not multi-type function offspring)"))
    return off
end

# The contact interval of a single-window model (used by analytical
# helpers that assume one generation-time distribution).
function _single_kernel(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(ArgumentError(
        "this analytical helper needs a single infectiousness window; this model has $(length(m.infectiousness))"))
    return m.infectiousness[1].kernel
end

function Base.show(io::IO, m::BranchingProcess)
    pop_str = m.population_size isa NoPopulation ? "unlimited" : string(m.population_size)
    if length(m.infectiousness) == 1
        w = m.infectiousness[1]
        off_str = w.offspring isa Distribution ? string(typeof(w.offspring)) : "Function"
        gt_str = w.kernel isa NoGenerationTime ? "none" :
                 w.kernel isa Distribution ? string(typeof(w.kernel)) : "Function"
        print(io,
            "BranchingProcess(offspring=$(off_str), generation_time=$(gt_str), population_size=$(pop_str))")
    else
        print(io,
            "BranchingProcess($(length(m.infectiousness)) infectiousness windows, population_size=$(pop_str))")
    end
end

# Single-type with a contact interval (one default window).
function BranchingProcess(offspring::Distribution, gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size, 1,
        NoTypeLabels())
end

# Single-type without a contact interval (pure chain statistics).
function BranchingProcess(offspring::Distribution;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess((Infectiousness(offspring),), population_size, 1, NoTypeLabels())
end

# Multi-type with an explicit offspring function.
function BranchingProcess(offspring::Function, gt::Union{Distribution, Function};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size, n_types,
        type_labels)
end

# Explicit windows: pass `Infectiousness` windows directly.
function BranchingProcess(windows::Tuple{Infectiousness, Vararg{Infectiousness}};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    BranchingProcess(windows, population_size, n_types, type_labels)
end
function BranchingProcess(window::Infectiousness, windows::Infectiousness...; kwargs...)
    BranchingProcess((window, windows...); kwargs...)
end

"""
    BranchingProcess(offspring_matrix, dist_fn, generation_time; kwargs...)

Construct a multi-type branching process from an offspring matrix.
`M[i,j]` is the expected number of type-i offspring from a type-j parent.
`dist_fn` maps each type's R to an offspring distribution.
"""
function BranchingProcess(offspring_matrix::Matrix{Float64},
        dist_fn::Function,
        gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    n = size(offspring_matrix, 1)
    size(offspring_matrix, 2) == n || throw(ArgumentError(
        "offspring_matrix must be square, got $(size(offspring_matrix))"))

    R_by_type = vec(sum(offspring_matrix, dims = 1))
    alloc_probs = similar(offspring_matrix)
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0.0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end

    offspring_fn = function (rng::AbstractRNG, individual)
        pt = individual_type(individual)
        dist = dist_fn(R_by_type[pt])
        total = rand(rng, dist)
        total == 0 && return zeros(Int, n)
        return rand(rng, Multinomial(total, alloc_probs[:, pt]))
    end

    BranchingProcess((Infectiousness(offspring_fn; kernel = gt),), population_size, n,
        type_labels)
end

# ── Offspring generation ─────────────────────────────────────────────

"""
    generate_offspring(model::BranchingProcess, parent, state)

Return how many contacts `parent` makes this generation through the
model's single infectiousness window: a count (single-type) or a count
per type (multi-type). The engine's window-aware
[`collect_exposures`](@ref) is what actually drives multi-window models;
this is the single-window seam kept for external callers.
"""
function generate_offspring(model::BranchingProcess, parent, state)
    length(model.infectiousness) == 1 || throw(ArgumentError(
        "generate_offspring is defined for a single infectiousness window; use collect_exposures for multi-window models"))
    draw_offspring(state.rng, model.infectiousness[1].offspring, parent, state)
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type offspring draw."""
function draw_offspring(rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState)
    rand(rng, offspring)
end

"""Function-based offspring draw. The function may be called as
`(rng, individual)` or `(rng, individual, state)`; the latter form lets
the offspring rule read population-level state (e.g. cumulative cases
for time- or policy-dependent caps)."""
function draw_offspring(rng::AbstractRNG, offspring::Function,
        individual, state::SimulationState)
    if applicable(offspring, rng, individual, state)
        return offspring(rng, individual, state)
    end
    return offspring(rng, individual)
end
