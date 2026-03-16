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
    inc_period = individual.onset_time - individual.infection_time
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
mutable struct Individual
    id::Int
    parent_id::Int                    # 0 for index cases
    generation::Int
    chain_id::Int
    infection_time::Float64
    onset_time::Float64
    asymptomatic::Bool
    test_positive::Bool
    # Transmission modifiers (set by interventions)
    susceptibility::Float64           # 1.0 = fully susceptible, 0.0 = immune
    infectiousness::Float64           # 1.0 = fully infectious
    # Intervention state
    isolated::Bool
    isolation_time::Float64           # Inf if not isolated
    traced::Bool
    quarantined::Bool
    vaccinated::Bool
    vaccination_time::Float64         # Inf if not vaccinated
    # Filled post-hoc by output pipeline
    secondary_case_ids::Vector{Int}
end

function Individual(; id::Int, parent_id::Int=0, generation::Int=0,
                    chain_id::Int=1, infection_time::Float64=0.0,
                    onset_time::Float64=0.0, asymptomatic::Bool=false,
                    test_positive::Bool=true,
                    susceptibility::Float64=1.0, infectiousness::Float64=1.0)
    Individual(id, parent_id, generation, chain_id, infection_time, onset_time,
               asymptomatic, test_positive,
               susceptibility, infectiousness,
               false, Inf, false, false, false, Inf, Int[])
end

# ── Simulation state ───────────────────────────────────────────────
mutable struct SimulationState
    individuals::Vector{Individual}
    active_ids::Vector{Int}           # indices of individuals who haven't transmitted yet
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
