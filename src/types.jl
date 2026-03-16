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

# ── Transmission models ─────────────────────────────────────────────
abstract type TransmissionModel end

"""
    BranchingProcess(offspring, generation_time; population_size=nothing)

Stochastic branching process transmission model.

- `offspring`: offspring distribution (e.g. `NegBin(2.5, 0.16)`)
- `generation_time`: fixed `Distribution` or function of incubation period
- `population_size`: if set, enables susceptible depletion (finite population).
  Each potential offspring survives with probability (N - cumulative_cases) / N.
"""
struct BranchingProcess <: TransmissionModel
    offspring::Distribution
    generation_time::GenerationTimeSpec
    population_size::Union{Int, Nothing}
end

BranchingProcess(offspring::Distribution, gt::GenerationTimeSpec;
                 population_size::Union{Int, Nothing}=nothing) =
    BranchingProcess(offspring, gt, population_size)

# ── Individual state ────────────────────────────────────────────────

"""
    Individual

Represents a single case in the transmission tree.

Core fields used by the engine:
- `id`, `parent_id`, `generation`, `chain_id`: transmission tree structure
- `infection_time`: time of infection (Float64, days)
- `susceptibility`: modifier on probability of being infected (default 1.0)
- `infectiousness`: modifier on onward transmission (default 1.0)
- `secondary_case_ids`: filled during simulation

The `state` dict holds all other properties — intervention state (isolated,
traced, etc.), clinical state (onset_time, asymptomatic), demographics (age,
sex), and any user-defined fields. Interventions initialise their own fields
via `initialise_individual!`.
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
# Interventions and output functions use these rather than raw dict access.

onset_time(ind::Individual) = get(ind.state, :onset_time, NaN)::Float64
is_isolated(ind) = get(ind.state, :isolated, false)::Bool
isolation_time(ind) = get(ind.state, :isolation_time, Inf)::Float64
is_traced(ind) = get(ind.state, :traced, false)::Bool
is_quarantined(ind) = get(ind.state, :quarantined, false)::Bool
is_vaccinated(ind) = get(ind.state, :vaccinated, false)::Bool
is_asymptomatic(ind) = get(ind.state, :asymptomatic, false)::Bool
is_test_positive(ind) = get(ind.state, :test_positive, true)::Bool

function set_isolated!(ind, time::Float64)
    ind.state[:isolated] = true
    ind.state[:isolation_time] = time
end

# ── Simulation state ───────────────────────────────────────────────
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
