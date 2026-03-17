"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process.

Supports single-type (offspring::Distribution) and multi-type
(offspring::Function returning Vector{Int}) offspring.
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_contacts = Individual[]
    new_infected_indices = Int[]
    next_id = length(state.individuals) + 1

    pop_suscept = _susceptible_fraction(state)
    if pop_suscept <= 0.0
        state.extinct = true
        state.active_ids = Int[]
        return state
    end

    for idx in state.active_ids
        individual = state.individuals[idx]

        # Resolve interventions on the parent
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Draw offspring — returns Int (single-type) or Vector{Int} (multi-type)
        offspring_result = _draw_offspring(state.rng, model.offspring, individual, state)

        # Create contacts and resolve infection
        gt_dist = get_generation_time(model.generation_time, individual)
        residual = _residual_fraction(interventions)

        next_id = _create_contacts!(new_contacts, new_infected_indices,
                                     offspring_result, individual, state,
                                     gt_dist, pop_suscept, residual,
                                     interventions, next_id)
    end

    # Apply post-transmission interventions to ALL contacts
    for intervention in interventions
        apply_post_transmission!(intervention, state, new_contacts)
    end

    # Update state — all contacts stored, only infected are active
    append!(state.individuals, new_contacts)
    n_infected = length(new_infected_indices)
    state.cumulative_cases += n_infected
    state.current_generation += 1

    if n_infected == 0
        state.extinct = true
        state.active_ids = Int[]
    else
        base_idx = length(state.individuals) - length(new_contacts)
        state.active_ids = [base_idx + findfirst(==(nid), [c.id for c in new_contacts])
                           for nid in new_infected_indices]
    end

    return state
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type: draw from distribution."""
function _draw_offspring(rng::AbstractRNG, offspring::Distribution,
                         individual, state::SimulationState)
    return rand(rng, offspring)
end

"""Multi-type: call offspring function with parent type."""
function _draw_offspring(rng::AbstractRNG, offspring::Function,
                         individual, state::SimulationState)
    parent_type = individual_type(individual)
    return offspring(rng, parent_type)
end

# ── Contact creation ─────────────────────────────────────────────────

"""Single-type contacts: n_contacts is an Int, no type assignment."""
function _create_contacts!(new_contacts, new_infected_indices,
                           n_contacts::Int, parent, state,
                           gt_dist, pop_suscept, residual,
                           interventions, next_id)
    for _ in 1:n_contacts
        gt = rand(state.rng, gt_dist)
        gt = max(gt, state.latent_period)
        inf_time = parent.infection_time + gt

        contact = _create_individual(state, parent.id, parent.chain_id,
                                          next_id, inf_time, interventions)

        infected = _resolve_infection(state.rng, parent, contact,
                                       gt, pop_suscept, residual)
        contact.state[:infected] = infected

        push!(parent.secondary_case_ids, next_id)
        push!(new_contacts, contact)
        infected && push!(new_infected_indices, next_id)
        next_id += 1
    end
    return next_id
end

"""Multi-type contacts: counts is a Vector{Int}, each contact gets a type."""
function _create_contacts!(new_contacts, new_infected_indices,
                           counts::Vector{Int}, parent, state,
                           gt_dist, pop_suscept, residual,
                           interventions, next_id)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            gt = rand(state.rng, gt_dist)
            gt = max(gt, state.latent_period)
            inf_time = parent.infection_time + gt

            contact = _create_individual(state, parent.id, parent.chain_id,
                                          next_id, inf_time, interventions)
            contact.state[:type] = type_idx

            infected = _resolve_infection(state.rng, parent, contact,
                                           gt, pop_suscept, residual)
            contact.state[:infected] = infected

            push!(parent.secondary_case_ids, next_id)
            push!(new_contacts, contact)
            infected && push!(new_infected_indices, next_id)
            next_id += 1
        end
    end
    return next_id
end

# ── Infection resolution ─────────────────────────────────────────────

"""
Determine whether a contact is successfully infected via competing risks.
"""
function _resolve_infection(rng::AbstractRNG, parent, contact,
                             generation_time::Float64, pop_suscept::Float64,
                             residual_transmission::Float64)
    # Population-level susceptible depletion
    if pop_suscept < 1.0 && rand(rng) > pop_suscept
        return false
    end

    # Individual susceptibility (vaccination, prior immunity)
    if contact.susceptibility < 1.0 && rand(rng) > contact.susceptibility
        return false
    end

    # Parent infectiousness modifier
    if parent.infectiousness < 1.0 && rand(rng) > parent.infectiousness
        return false
    end

    # Competing risk: did transmission happen before parent was isolated?
    if is_isolated(parent)
        iso_t = isolation_time(parent)
        transmission_time = parent.infection_time + generation_time

        if transmission_time >= iso_t
            if residual_transmission <= 0.0 || rand(rng) > residual_transmission
                return false
            end
        end
    end

    return true
end
