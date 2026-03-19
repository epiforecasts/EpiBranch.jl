# ── Generation time specification ────────────────────────────────────

"""
    get_generation_time(gt, individual)

Return the generation time distribution for a specific individual.
For a `Distribution`, it is returned unchanged. For a `Function`, it is
called with the individual's incubation period.
"""
get_generation_time(gt::Distribution, individual) = gt

function get_generation_time(gt::Function, individual)
    inc_period = get(individual.state, :onset_time, NaN) - individual.infection_time
    if isnan(inc_period) || inc_period <= 0.0
        inc_period = 5.0
    end
    gt(inc_period)
end

# ── Transmission models ─────────────────────────────────────────────
abstract type TransmissionModel end

"""Interface methods with defaults for any TransmissionModel."""
population_size(::TransmissionModel) = nothing
latent_period(::TransmissionModel) = 0.0
n_types(::TransmissionModel) = 1

"""
    BranchingProcess(offspring, generation_time; population_size=nothing)

Stochastic branching process transmission model.

    BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
    BranchingProcess(NegBin(0.8, 0.5))  # no timing, pure chain statistics
    BranchingProcess(M, R_j -> NegBin(R_j, 0.16), LogNormal(1.6, 0.5))  # multi-type
"""
struct BranchingProcess <: TransmissionModel
    offspring::Union{Distribution, Function}
    generation_time::Union{Distribution, Function, Nothing}
    population_size::Union{Int, Nothing}
    latent_period::Float64
    n_types::Int
    type_labels::Union{Vector{String}, Nothing}
end

population_size(m::BranchingProcess) = m.population_size
latent_period(m::BranchingProcess) = m.latent_period
n_types(m::BranchingProcess) = m.n_types

# Single-type with generation time
BranchingProcess(offspring::Distribution, gt::Union{Distribution, Function};
                 population_size::Union{Int, Nothing}=nothing,
                 latent_period::Real=0.0) =
    BranchingProcess(offspring, gt, population_size, Float64(latent_period), 1, nothing)

# Single-type without generation time (pure chain statistics)
BranchingProcess(offspring::Distribution;
                 population_size::Union{Int, Nothing}=nothing) =
    BranchingProcess(offspring, nothing, population_size, 0.0, 1, nothing)

# Multi-type with explicit offspring function
BranchingProcess(offspring::Function, gt::Union{Distribution, Function};
                 n_types::Int, population_size::Union{Int, Nothing}=nothing,
                 latent_period::Real=0.0,
                 type_labels::Union{Vector{String}, Nothing}=nothing) =
    BranchingProcess(offspring, gt, population_size, Float64(latent_period), n_types, type_labels)

"""
    BranchingProcess(offspring_matrix, dist_fn, generation_time; kwargs...)

Construct a multi-type branching process from an offspring matrix.
`M[i,j]` is the expected number of type-i offspring from a type-j parent.
`dist_fn` maps each type's R to an offspring distribution.
"""
function BranchingProcess(offspring_matrix::Matrix{Float64},
                          dist_fn::Function,
                          gt::Union{Distribution, Function};
                          population_size::Union{Int, Nothing}=nothing,
                          latent_period::Real=0.0,
                          type_labels::Union{Vector{String}, Nothing}=nothing)
    n = size(offspring_matrix, 1)
    size(offspring_matrix, 2) == n || throw(ArgumentError(
        "offspring_matrix must be square, got $(size(offspring_matrix))"))

    R_by_type = vec(sum(offspring_matrix, dims=1))
    alloc_probs = similar(offspring_matrix)
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0.0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end

    # Issue 11: use Multinomial from Distributions.jl instead of hand-rolled loop
    offspring_fn = function (rng::AbstractRNG, parent_type::Int)
        dist = dist_fn(R_by_type[parent_type])
        total = rand(rng, dist)
        total == 0 && return zeros(Int, n)
        return rand(rng, Multinomial(total, alloc_probs[:, parent_type]))
    end

    BranchingProcess(offspring_fn, gt, population_size, Float64(latent_period), n, type_labels)
end

# ── Individual state ────────────────────────────────────────────────

"""
    Individual

A single contact in the transmission tree (infected or not).

Core fields (used by the engine): `id`, `parent_id`, `generation`,
`chain_id`, `infection_time`, `susceptibility`, `infectiousness`,
`secondary_case_ids`.

The `state` dict holds everything else: intervention state, clinical
state, demographics, and user-defined fields.

Note: `id` is the 1-based index into `state.individuals`. This invariant
is relied on for O(1) parent lookups.
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

State of a running or completed simulation.
"""
mutable struct SimulationState{R <: AbstractRNG}
    individuals::Vector{Individual}
    active_ids::Vector{Int}
    current_generation::Int
    rng::R
    cumulative_cases::Int
    extinct::Bool
    population_size::Union{Int, Nothing}
    latent_period::Float64
    max_infection_time::Float64
    attributes::Union{Function, Nothing}
end
