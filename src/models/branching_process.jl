"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process. Produces the new
contacts (both eventual infections and contacts whose transmission
will be blocked) without deciding `:infected` — the engine resolves
that via [`competing_risk`](@ref) after `step!` returns. See the
[Design](@ref "Simulation, mutation, and automatic differentiation")
section for implications on automatic differentiation.
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_contacts = Individual[]
    next_id = length(state.individuals) + 1

    for idx in state.active_ids
        individual = state.individuals[idx]

        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        offspring_result = _draw_offspring(state.rng, model.offspring, individual, state)

        gt_dist = model.generation_time === nothing ? nothing :
                  get_generation_time(model.generation_time, individual)

        next_id = _create_contacts!(new_contacts,
            offspring_result, individual, state,
            gt_dist, interventions, next_id)
    end

    return new_contacts
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type offspring draw."""
function _draw_offspring(rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState)
    rand(rng, offspring)
end

"""Function-based offspring draw. The function may be called as
`(rng, individual)` or `(rng, individual, state)`; the latter form lets
the offspring rule read population-level state (e.g. cumulative cases
for time- or policy-dependent caps)."""
function _draw_offspring(rng::AbstractRNG, offspring::Function,
        individual, state::SimulationState)
    if applicable(offspring, rng, individual, state)
        return offspring(rng, individual, state)
    end
    return offspring(rng, individual)
end

# ── Contact creation ─────────────────────────────────────────────────

function _make_one_contact!(new_contacts, parent, state,
        gt_dist, interventions, next_id;
        type_idx::Union{Int, NoTypeLabels} = NoTypeLabels())
    if gt_dist === nothing
        inf_time = parent.infection_time
    else
        gt = rand(state.rng, gt_dist)
        gt = max(gt, state.latent_period)
        inf_time = parent.infection_time + gt
    end

    contact = _create_individual(state, parent.id, parent.chain_id,
        next_id, inf_time, interventions)
    _set_type!(contact, type_idx)

    # `:infected` defaults to false in `_create_individual` and is
    # set to true (if applicable) by the engine's competing-risks
    # resolution after `step!` returns.

    push!(parent.secondary_case_ids, next_id)
    push!(new_contacts, contact)
    return next_id + 1
end

_set_type!(contact, ::NoTypeLabels) = nothing
_set_type!(contact, idx::Int) = (contact.state[:type] = idx)

"""Single-type contacts."""
function _create_contacts!(new_contacts,
        n_contacts::Int, parent, state,
        gt_dist, interventions, next_id)
    for _ in 1:n_contacts
        next_id = _make_one_contact!(new_contacts, parent, state,
            gt_dist, interventions, next_id)
    end
    return next_id
end

"""Multi-type contacts."""
function _create_contacts!(new_contacts,
        counts::Vector{Int}, parent, state,
        gt_dist, interventions, next_id)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            next_id = _make_one_contact!(new_contacts, parent, state,
                gt_dist, interventions, next_id; type_idx)
        end
    end
    return next_id
end
