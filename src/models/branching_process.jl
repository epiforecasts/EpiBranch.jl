"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process. Returns the new
individuals produced this step (both infected and uninfected contacts);
the engine handles appending and bookkeeping. See the
[Design](@ref "Simulation, mutation, and automatic differentiation")
section for implications on automatic differentiation.
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_contacts = Individual[]
    next_id = length(state.individuals) + 1

    pop_suscept = _susceptible_fraction(state)
    pop_suscept <= 0.0 && return new_contacts

    for idx in state.active_ids
        individual = state.individuals[idx]

        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        offspring_result = _draw_offspring(state.rng, model.offspring, individual, state)

        gt_dist = model.generation_time === nothing ? nothing :
                  get_generation_time(model.generation_time, individual)
        residual = _post_isolation_transmission(interventions)

        next_id = _create_contacts!(new_contacts,
            offspring_result, individual, state,
            gt_dist, pop_suscept, residual,
            interventions, next_id)
    end

    for intervention in interventions
        apply_post_transmission!(intervention, state, new_contacts)
    end

    return new_contacts
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type offspring draw."""
function _draw_offspring(rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState)
    rand(rng, offspring)
end

"""Function-based offspring draw. The function receives the RNG and individual."""
function _draw_offspring(rng::AbstractRNG, offspring::Function,
        individual, state::SimulationState)
    return offspring(rng, individual)
end

# ── Contact creation ─────────────────────────────────────────────────

function _make_one_contact!(new_contacts, parent, state,
        gt_dist, pop_suscept, residual,
        interventions, next_id;
        type_idx::Union{Int, NoTypeLabels} = NoTypeLabels())
    if gt_dist === nothing
        gt = 0.0
        inf_time = parent.infection_time
    else
        gt = rand(state.rng, gt_dist)
        gt = max(gt, state.latent_period)
        inf_time = parent.infection_time + gt
    end

    contact = _create_individual(state, parent.id, parent.chain_id,
        next_id, inf_time, interventions)
    _set_type!(contact, type_idx)

    infected = _resolve_infection(state.rng, parent, contact,
        gt, pop_suscept, residual)
    contact.state[:infected] = infected

    # Clinical transitions are resolved by the engine after step!
    # returns (see _resolve_new_transitions! in simulation.jl).

    push!(parent.secondary_case_ids, next_id)
    push!(new_contacts, contact)
    return next_id + 1
end

_set_type!(contact, ::NoTypeLabels) = nothing
_set_type!(contact, idx::Int) = (contact.state[:type] = idx)

"""Single-type contacts."""
function _create_contacts!(new_contacts,
        n_contacts::Int, parent, state,
        gt_dist, pop_suscept, residual,
        interventions, next_id)
    for _ in 1:n_contacts
        next_id = _make_one_contact!(new_contacts, parent, state,
            gt_dist, pop_suscept, residual,
            interventions, next_id)
    end
    return next_id
end

"""Multi-type contacts."""
function _create_contacts!(new_contacts,
        counts::Vector{Int}, parent, state,
        gt_dist, pop_suscept, residual,
        interventions, next_id)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            next_id = _make_one_contact!(new_contacts, parent, state,
                gt_dist, pop_suscept, residual,
                interventions, next_id; type_idx)
        end
    end
    return next_id
end

# ── Infection resolution ─────────────────────────────────────────────

"""Determine whether a contact is successfully infected via competing risks."""
function _resolve_infection(rng::AbstractRNG, parent, contact,
        generation_time::Float64, pop_suscept::Float64,
        post_isolation_transmission::Float64)
    pop_suscept < 1.0 && rand(rng) > pop_suscept && return false
    contact.susceptibility < 1.0 && rand(rng) > contact.susceptibility && return false
    parent.infectiousness < 1.0 && rand(rng) > parent.infectiousness && return false

    if is_isolated(parent)
        iso_t = isolation_time(parent)
        transmission_time = parent.infection_time + generation_time
        if transmission_time >= iso_t
            (post_isolation_transmission <= 0.0 ||
             rand(rng) > post_isolation_transmission) && return false
        end
    end

    return true
end
