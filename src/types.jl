# ── Sentinel types ──────────────────────────────────────────────────
# These replace Union{T, Nothing} patterns throughout the codebase,
# enabling dispatch instead of runtime nothing-checks.

"""Sentinel indicating no population size constraint (infinite population)."""
struct NoPopulation end

"""Sentinel indicating no attributes function is provided."""
struct NoAttributes end

"""Sentinel indicating no type labels for multi-type models."""
struct NoTypeLabels end

"""Sentinel indicating no delay distribution is provided."""
struct NoDelay end

"""Sentinel indicating no age-specific case fatality rate is provided."""
struct NoCFR end

"""Sentinel indicating no age distribution is provided."""
struct NoAgeDistribution end

"""Sentinel indicating no outcome options are provided."""
struct NoOutcomes end

"""Sentinel indicating no demographic options are provided."""
struct NoDemographics end

"""Sentinel indicating no case cap for extinction/containment checks."""
struct NoCases end

# ── Generation time specification ────────────────────────────────────

"""
    get_generation_time(gt, individual)

Return the generation time distribution for a specific individual.
For a `Distribution`, the distribution is returned unchanged. For a `Function`,
it is called with the individual's incubation period.
"""
get_generation_time(gt::Distribution, individual) = gt

function get_generation_time(gt::Function, individual)
    onset = get(individual.state, :onset_time, NaN)
    inc_period = onset - individual.infection_time
    if isnan(inc_period) || inc_period <= 0.0
        @debug "Missing or non-positive incubation period (e.g. asymptomatic individual); using 5.0 days" maxlog=1
        inc_period = 5.0
    end
    gt(inc_period)
end

# ── Transmission models ─────────────────────────────────────────────
abstract type TransmissionModel end

"""Interface methods with defaults for any TransmissionModel."""
population_size(::TransmissionModel) = NoPopulation()

"""Extract the offspring specification from a single-type model.

Returns whatever the model stores in `offspring` (typically a
`Distribution`, but can also be any type for which
`chain_size_distribution` is defined — e.g. `ClusterMixed`). Callers
that need a `Distribution` specifically should check the return type.
Throws only for multi-type (function-based) offspring, which this
accessor cannot sensibly return.
"""
function _single_type_offspring(model::TransmissionModel)
    off = model.offspring
    off isa Function && throw(ArgumentError(
        "This function only works with single-type models (not multi-type function offspring)"))
    return off
end
latent_period(::TransmissionModel) = 0.0
n_types(::TransmissionModel) = 1

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
    latent_period::Float64
    n_types::Int
    type_labels::L
end

population_size(m::BranchingProcess) = m.population_size
latent_period(m::BranchingProcess) = m.latent_period
n_types(m::BranchingProcess) = m.n_types

function Base.show(io::IO, m::BranchingProcess)
    off_str = m.offspring isa Distribution ? string(typeof(m.offspring)) : "Function"
    gt_str = m.generation_time === nothing ? "nothing" :
             m.generation_time isa Distribution ? string(typeof(m.generation_time)) :
             "Function"
    pop_str = m.population_size isa NoPopulation ? "unlimited" : string(m.population_size)
    print(io,
        "BranchingProcess(offspring=$(off_str), generation_time=$(gt_str), population_size=$(pop_str))")
end

# Single-type with generation time
function BranchingProcess(offspring::Distribution, gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        latent_period::Real = 0.0)
    BranchingProcess(
        offspring, gt, population_size, Float64(latent_period), 1, NoTypeLabels())
end

# Single-type without generation time (pure chain statistics)
function BranchingProcess(offspring::Distribution;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess(offspring, nothing, population_size, 0.0, 1, NoTypeLabels())
end

# Multi-type with explicit offspring function
function BranchingProcess(offspring::Function, gt::Union{Distribution, Function};
        n_types::Int, population_size::Union{Int, NoPopulation} = NoPopulation(),
        latent_period::Real = 0.0,
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    BranchingProcess(
        offspring, gt, population_size, Float64(latent_period), n_types, type_labels)
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
        latent_period::Real = 0.0,
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
        offspring_fn, gt, population_size, float(latent_period), n, type_labels)
end

# ── Individual state ────────────────────────────────────────────────

"""
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

function Base.show(io::IO, ind::Individual)
    infected_str = is_infected(ind) ? "infected" : "contact-only"
    isolated_str = is_isolated(ind) ? ", isolated" : ""
    print(io,
        "Individual(id=$(ind.id), gen=$(ind.generation), chain=$(ind.chain_id), t=$(round(ind.infection_time, digits=1)), $(infected_str)$(isolated_str))")
end

function Individual(; id::Int, parent_id::Int = 0, generation::Int = 0,
        chain_id::Int = 1, infection_time::Float64 = 0.0,
        susceptibility::Float64 = 1.0, infectiousness::Float64 = 1.0,
        state::Dict{Symbol, Any} = Dict{Symbol, Any}())
    Individual(id, parent_id, generation, chain_id, infection_time,
        susceptibility, infectiousness, Int[], state)
end

# ── State accessors ──────────────────────────────────────────────────

"""Symptom onset time (Float64, NaN if asymptomatic or not set)."""
onset_time(ind::Individual) = get(ind.state, :onset_time, NaN)::Float64

"""Whether the individual is isolated."""
is_isolated(ind::Individual) = get(ind.state, :isolated, false)::Bool

"""Time of isolation (Float64, Inf if not isolated)."""
isolation_time(ind::Individual) = get(ind.state, :isolation_time, Inf)::Float64

"""Whether the individual was traced via contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated."""
is_vaccinated(ind::Individual) = get(ind.state, :vaccinated, false)::Bool

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""Whether the individual tested positive."""
is_test_positive(ind::Individual) = get(ind.state, :test_positive, false)::Bool

"""Whether the individual was successfully infected (vs contact only)."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""Type index for multi-type branching processes (default 1)."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

"""Mark an individual as isolated at the given time."""
function set_isolated!(ind::Individual, time::Float64)
    ind.state[:isolated] = true
    ind.state[:isolation_time] = time
end

# ── Simulation state ───────────────────────────────────────────────

"""
State of a running or completed simulation.
"""
mutable struct SimulationState{R <: AbstractRNG, P, A}
    individuals::Vector{Individual}
    active_ids::Vector{Int}
    current_generation::Int
    rng::R
    cumulative_cases::Int
    extinct::Bool
    population_size::P
    latent_period::Float64
    max_infection_time::Float64
    attributes::A
end

function Base.show(io::IO, s::SimulationState)
    status = s.extinct ? "extinct" : "active"
    print(io,
        "SimulationState(cases=$(s.cumulative_cases), individuals=$(length(s.individuals)), gen=$(s.current_generation), $(status))")
end
