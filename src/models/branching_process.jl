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

# ── Generation time specification ────────────────────────────────────

"""
    get_generation_time(gt, individual)

Return the generation time distribution for a specific individual.

For a `Distribution`, everyone shares the same distribution. For a
`Function`, the engine calls it with the individual and uses the
`Distribution` it returns, so the generation time can read anything in
`individual.state`: the incubation period, or any per-individual
quantity an attributes function has stored. That is how the generation
time and symptom onset can come from one per-individual draw instead of
two independent ones. Use [`incubation_period`](@ref) to read the
incubation period inside such a function.
"""
get_generation_time(gt::Distribution, individual) = gt
get_generation_time(ngt::NoGenerationTime, individual) = ngt
get_generation_time(gt::Function, individual) = gt(individual)

# ── Neighbour generation ─────────────────────────────────────────────

"""
    contacts_of(model::BranchingProcess, parent, state)

A branching process is a network grown lazily as a tree: each infectious
`parent` reaches fresh, never-before-seen contacts. The offspring
distribution gives how many; each is created with [`make_contact!`](@ref)
at a generation-time-shifted infection time and returned as a
`(contact, infection_time)` pair. `:infected` is left undecided — the
engine resolves it via [`competing_risk`](@ref) after `contacts_of`
returns (both eventual infections and contacts whose transmission is
blocked are produced). See the [Design](@ref "Simulation, mutation, and
automatic differentiation") section for implications on automatic
differentiation.
"""
function contacts_of(model::BranchingProcess, parent::Individual,
        state::SimulationState)
    offspring_result = draw_offspring(state.rng, model.offspring, parent, state)
    gt_dist = get_generation_time(model.generation_time, parent)
    return _create_contacts(offspring_result, parent, state, gt_dist)
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

# ── Contact creation ─────────────────────────────────────────────────

"""Compute a contact's infection time from the parent's infection time
and the generation-time draw. Dispatches on the generation-time spec:
[`NoGenerationTime`](@ref) (no timing) returns the parent's time;
otherwise samples and adds. The generation time distribution should
already encode any biological constraint (e.g. a minimum latent
period); to enforce a lower bound use `truncated(gt_dist, lower, Inf)`
or a shifted distribution."""
_infection_time(::NoGenerationTime, parent, state) = parent.infection_time
function _infection_time(gt_dist::Distribution, parent, state)
    return parent.infection_time + rand(state.rng, gt_dist)
end

"""Single-type contacts: each offspring becomes a fresh contact, returned
as a `(contact, infection_time)` pair."""
function _create_contacts(n_contacts::Int, parent, state, gt_dist)
    result = Tuple{Individual, Float64}[]
    for _ in 1:n_contacts
        inf_time = _infection_time(gt_dist, parent, state)
        push!(result, (make_contact!(state, parent, inf_time), inf_time))
    end
    return result
end

"""Multi-type contacts: `counts[i]` fresh contacts of type `i`, each
returned as a `(contact, infection_time)` pair."""
function _create_contacts(counts::Vector{Int}, parent, state, gt_dist)
    result = Tuple{Individual, Float64}[]
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            inf_time = _infection_time(gt_dist, parent, state)
            push!(result, (make_contact!(state, parent, inf_time; type_idx), inf_time))
        end
    end
    return result
end
