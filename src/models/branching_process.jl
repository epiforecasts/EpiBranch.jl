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
  window — `(:recovered, :died, :isolated)` for community spread,
  `(:buried,)` for a funeral window. Empty by default (no censoring).
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

The second positional argument is the contact interval (generation-time
distribution); with the default single window and no natural-history
states it is the generation interval, exactly as before.

Every constructor also takes `interventions`, `attributes`, and
`observation` as keyword arguments, the inputs the model carries. They
default to none, so a model with no interventions, attributes or
observation reads exactly as above.

# Examples

```julia
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
BranchingProcess(NegBin(0.8, 0.5))  # no timing, pure chain statistics
BranchingProcess(M, R_j -> NegBin(R_j, 0.16), LogNormal(1.6, 0.5))  # multi-type

# attach interventions at construction
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5);
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0))])
```

Transmission is a tuple of [`Infectiousness`](@ref) windows; the
convenience constructors above build a single default window with
`from = :infection` and no censoring.
"""
struct BranchingProcess{W <: Tuple, P, L, A, O} <: TransmissionModel
    infectiousness::W
    population_size::P
    n_types::Int
    type_labels::L
    progression::Vector{AbstractClinicalTransition}
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O

    function BranchingProcess(infectiousness::W, population_size::P, n_types::Int,
            type_labels::L,
            progression::Vector{AbstractClinicalTransition};
            interventions = AbstractIntervention[],
            attributes::A = NoAttributes(),
            observation::O = NoObservation()) where {W <: Tuple, P, L, A, O}
        _validate_windows(infectiousness, progression)
        return new{W, P, L, A, O}(
            infectiousness, population_size, n_types, type_labels, progression,
            _intervention_vector(interventions), attributes, observation)
    end
end

# Warn when a window's `from` state is never produced — the window would
# silently never open. `from` may legitimately be a state written by an
# attribute or intervention (rather than a progression transition), so this
# is a warning, not an error. The co-location of windows and progression on
# the model is what lets us check this at all.
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

# The model's natural history: the timed clinical-state transitions a case
# moves through (latent, onset, severity, death/recovery, burial). The
# infectiousness windows and interventions key off the states it writes.
# Optional and empty by default; models without one resolve no transitions.
_progression(::TransmissionModel) = AbstractClinicalTransition[]
_progression(m::BranchingProcess) = m.progression
_progvec(p) = convert(Vector{AbstractClinicalTransition}, p)

# The model carries its interventions, attributes and observation; the
# shared accessors in model_inputs.jl read them from here.
interventions(m::BranchingProcess) = m.interventions
attributes(m::BranchingProcess) = m.attributes
observation(m::BranchingProcess) = m.observation

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
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        progression = AbstractClinicalTransition[],
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size, 1,
        NoTypeLabels(), _progvec(progression);
        interventions, attributes, observation)
end

# Single-type without a contact interval (pure chain statistics).
function BranchingProcess(offspring::Distribution;
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        progression = AbstractClinicalTransition[],
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    BranchingProcess((Infectiousness(offspring),), population_size, 1,
        NoTypeLabels(), _progvec(progression);
        interventions, attributes, observation)
end

# Multi-type with an explicit offspring function.
function BranchingProcess(offspring::Function, gt::Union{Distribution, Function};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels(),
        progression = AbstractClinicalTransition[],
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size, n_types,
        type_labels, _progvec(progression);
        interventions, attributes, observation)
end

# Explicit windows: pass `Infectiousness` windows directly.
function BranchingProcess(windows::Tuple{Infectiousness, Vararg{Infectiousness}};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels(),
        progression = AbstractClinicalTransition[],
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    BranchingProcess(windows, population_size, n_types, type_labels,
        _progvec(progression);
        interventions, attributes, observation)
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
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels(),
        progression = AbstractClinicalTransition[],
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
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
        type_labels, _progvec(progression);
        interventions, attributes, observation)
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
