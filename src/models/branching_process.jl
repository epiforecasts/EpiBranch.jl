# ── BranchingProcess type and constructors ──────────────────────────

"""
Stochastic branching process transmission model.

# Examples

```julia
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
BranchingProcess(NegBin(0.8, 0.5))  # no timing, pure chain statistics
BranchingProcess(M, R_j -> NegBin(R_j, 0.16), LogNormal(1.6, 0.5))  # multi-type
```
"""
struct BranchingProcess{O, G, P, L} <: TransmissionModel
    offspring::O
    generation_time::G
    population_size::P
    n_types::Int
    type_labels::L
end

population_size(m::BranchingProcess) = m.population_size
n_types(m::BranchingProcess) = m.n_types

function Base.show(io::IO, m::BranchingProcess)
    off_str = m.offspring isa Distribution ? string(typeof(m.offspring)) : "Function"
    gt_str = m.generation_time isa NoGenerationTime ? "none" :
             m.generation_time isa Distribution ? string(typeof(m.generation_time)) :
             "Function"
    pop_str = m.population_size isa NoPopulation ? "unlimited" : string(m.population_size)
    print(io,
        "BranchingProcess(offspring=$(off_str), generation_time=$(gt_str), population_size=$(pop_str))")
end

# Single-type with generation time
function BranchingProcess(offspring::Distribution, gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess(
        offspring, gt, population_size, 1, NoTypeLabels())
end

# Single-type without generation time (pure chain statistics)
function BranchingProcess(offspring::Distribution;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess(offspring, NoGenerationTime(), population_size,
        1, NoTypeLabels())
end

# Multi-type with explicit offspring function
function BranchingProcess(offspring::Function, gt::Union{Distribution, Function};
        n_types::Int = 1, population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    BranchingProcess(
        offspring, gt, population_size, n_types, type_labels)
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

    BranchingProcess(
        offspring_fn, gt, population_size, n, type_labels)
end

# ── Offspring generation ─────────────────────────────────────────────

"""
    generate_offspring(model::BranchingProcess, parent, state)

Return how many contacts `parent` makes this generation: a single count
(single-type) or a count per type (multi-type). This is the
offspring-driven model interface — the engine creates that many
candidate contacts, assigns each an infection time from the model's
`generation_time`, and resolves competing risks. The model itself
assigns no timing, constructs no `Individual`s, and never sees
interventions.
"""
function generate_offspring(model::BranchingProcess, parent, state)
    draw_offspring(state.rng, model.offspring, parent, state)
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
