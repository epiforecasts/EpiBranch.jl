# ── Generation time specification ────────────────────────────────────

"""
    GenerationTimeSpec

The generation time field of a transmission model. Either:
- A `Distribution` (fixed, same for all individuals), or
- A function `(incubation_period::Float64) -> Distribution` that returns an
  individual-specific generation time distribution based on their incubation period.
"""
const GenerationTimeSpec = Union{Distribution, Function}

"""
    get_generation_time(gt, individual)

Return the generation time distribution for a specific individual.
"""
get_generation_time(gt::Distribution, individual) = gt

function get_generation_time(gt::Function, individual)
    inc_period = get(individual.state, :onset_time, NaN) - individual.infection_time
    if isnan(inc_period) || inc_period <= 0.0
        inc_period = 5.0
    end
    gt(inc_period)
end

# ── Offspring specification ─────────────────────────────────────────

"""
    OffspringSpec

The offspring field of a transmission model. Either:
- A `Distribution` (single-type: draw a count), or
- A function `(rng::AbstractRNG, parent_type::Int) -> Vector{Int}`
  (multi-type: returns counts per type)
"""
const OffspringSpec = Union{Distribution, Function}

# ── Transmission models ─────────────────────────────────────────────
abstract type TransmissionModel end

"""
    BranchingProcess(offspring, generation_time; population_size=nothing)

Stochastic branching process transmission model.

Single-type:
    BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

Multi-type from offspring matrix + distribution family:
    BranchingProcess(M, R_j -> NegBin(R_j, 0.16), LogNormal(1.6, 0.5))

Multi-type with custom offspring function:
    BranchingProcess((rng, j) -> my_draw(rng, j), LogNormal(1.6, 0.5); n_types=4)

- `offspring`: `Distribution` (single-type) or `Function` (multi-type)
- `generation_time`: fixed `Distribution` or function of incubation period
- `population_size`: if set, enables susceptible depletion
- `n_types`: number of types (1 for single-type, inferred from matrix if provided)
- `type_labels`: optional labels for types (e.g. age group names)
"""
struct BranchingProcess <: TransmissionModel
    offspring::OffspringSpec
    generation_time::GenerationTimeSpec
    population_size::Union{Int, Nothing}
    n_types::Int
    type_labels::Union{Vector{String}, Nothing}
end

# Single-type constructor (existing API unchanged)
BranchingProcess(offspring::Distribution, gt::GenerationTimeSpec;
                 population_size::Union{Int, Nothing}=nothing) =
    BranchingProcess(offspring, gt, population_size, 1, nothing)

# Multi-type with explicit offspring function
BranchingProcess(offspring::Function, gt::GenerationTimeSpec;
                 n_types::Int, population_size::Union{Int, Nothing}=nothing,
                 type_labels::Union{Vector{String}, Nothing}=nothing) =
    BranchingProcess(offspring, gt, population_size, n_types, type_labels)

"""
    BranchingProcess(offspring_matrix, dist_fn, generation_time; kwargs...)

Construct a multi-type branching process from an offspring matrix.

- `offspring_matrix`: `Matrix{Float64}` where `M[i,j]` = expected type-i
  offspring from a type-j parent. Column sums give type-specific R values.
- `dist_fn`: function `(R_j::Float64) -> Distribution` that constructs an
  offspring distribution for a parent with mean `R_j`
- `generation_time`: as above
"""
function BranchingProcess(offspring_matrix::Matrix{Float64},
                          dist_fn::Function,
                          gt::GenerationTimeSpec;
                          population_size::Union{Int, Nothing}=nothing,
                          type_labels::Union{Vector{String}, Nothing}=nothing)
    n = size(offspring_matrix, 1)
    size(offspring_matrix, 2) == n || throw(ArgumentError(
        "offspring_matrix must be square, got $(size(offspring_matrix))"))

    # Precompute column sums (type-specific R) and allocation probabilities
    R_by_type = vec(sum(offspring_matrix, dims=1))
    alloc_probs = similar(offspring_matrix)
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0.0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end

    # Build offspring function
    offspring_fn = function (rng::AbstractRNG, parent_type::Int)
        dist = dist_fn(R_by_type[parent_type])
        total = rand(rng, dist)
        # Allocate to types via multinomial
        probs = alloc_probs[:, parent_type]
        counts = zeros(Int, n)
        for _ in 1:total
            # Sample type from categorical distribution
            u = rand(rng)
            cumsum = 0.0
            for i in 1:n
                cumsum += probs[i]
                if u <= cumsum
                    counts[i] += 1
                    break
                end
            end
        end
        return counts
    end

    BranchingProcess(offspring_fn, gt, population_size, n, type_labels)
end

# ── Individual state ────────────────────────────────────────────────

"""
    Individual

Represents a single contact in the transmission tree (infected or not).

Core fields used by the engine:
- `id`, `parent_id`, `generation`, `chain_id`: transmission tree structure
- `infection_time`: time of infection/contact (Float64, days)
- `susceptibility`: modifier on probability of being infected (default 1.0)
- `infectiousness`: modifier on onward transmission (default 1.0)
- `secondary_case_ids`: filled during simulation

The `state` dict holds all other properties — intervention state (isolated,
traced, etc.), clinical state (onset_time, asymptomatic), demographics (age,
sex), type, and any user-defined fields. Interventions initialise their own
fields via `initialise_individual!`.
"""
mutable struct Individual
    id::Int
    parent_id::Int
    generation::Int
    chain_id::Int
    infection_time::Float64
    susceptibility::Float64
    infectiousness::Float64
    secondary_case_ids::Vector{Int}
    state::Dict{Symbol, Any}
end

function Individual(; id::Int, parent_id::Int=0, generation::Int=0,
                    chain_id::Int=1, infection_time::Float64=0.0,
                    susceptibility::Float64=1.0, infectiousness::Float64=1.0,
                    state::Dict{Symbol, Any}=Dict{Symbol, Any}())
    Individual(id, parent_id, generation, chain_id, infection_time,
               susceptibility, infectiousness, Int[], state)
end

# ── State accessors ──────────────────────────────────────────────────
# Clean API for reading intervention/clinical state with safe defaults.

"""Symptom onset time (Float64, NaN if asymptomatic or not set)."""
onset_time(ind::Individual) = get(ind.state, :onset_time, NaN)::Float64

"""Whether the individual is isolated."""
is_isolated(ind) = get(ind.state, :isolated, false)::Bool

"""Time of isolation (Float64, Inf if not isolated)."""
isolation_time(ind) = get(ind.state, :isolation_time, Inf)::Float64

"""Whether the individual was traced via contact tracing."""
is_traced(ind) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated."""
is_vaccinated(ind) = get(ind.state, :vaccinated, false)::Bool

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind) = get(ind.state, :asymptomatic, false)::Bool

"""Whether the individual tested positive."""
is_test_positive(ind) = get(ind.state, :test_positive, true)::Bool

"""Whether the individual was successfully infected (vs contact only)."""
is_infected(ind) = get(ind.state, :infected, true)::Bool

"""Type index for multi-type branching processes (default 1)."""
individual_type(ind) = get(ind.state, :type, 1)::Int

"""Mark an individual as isolated at the given time."""
function set_isolated!(ind, time::Float64)
    ind.state[:isolated] = true
    ind.state[:isolation_time] = time
end

# ── Simulation state ───────────────────────────────────────────────

"""
    SimulationState

Holds the state of a running or completed simulation: all individuals
(infected and non-infected contacts), active case indices, generation
counter, RNG, and clinical/population parameters.
"""
mutable struct SimulationState
    individuals::Vector{Individual}
    active_ids::Vector{Int}
    current_generation::Int
    rng::AbstractRNG
    cumulative_cases::Int
    extinct::Bool
    incubation_period::Union{Distribution, Nothing}
    prob_asymptomatic::Float64
    asymptomatic_R_scaling::Float64
    test_sensitivity::Float64
    latent_period::Float64
    population_size::Union{Int, Nothing}
end
