"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process.
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_contacts = Individual[]
    new_infected_ids = Int[]
    next_id = length(state.individuals) + 1

    pop_suscept = _susceptible_fraction(state)
    if pop_suscept <= 0.0
        state.extinct = true
        state.active_ids = Int[]
        return state
    end

    for idx in state.active_ids
        individual = state.individuals[idx]

        for intervention in interventions
            resolve_individual!(intervention, individual, state)
            _enforce_start_time!(intervention, individual)
        end

        offspring_result = _draw_offspring(state.rng, model.offspring, individual, state)

        gt_dist = model.generation_time === nothing ? nothing :
            get_generation_time(model.generation_time, individual)
        residual = _residual_transmission(interventions)

        next_id = _create_contacts!(new_contacts, new_infected_ids,
                                     offspring_result, individual, state,
                                     gt_dist, pop_suscept, residual,
                                     interventions, next_id)
    end

    for intervention in interventions
        apply_post_transmission!(intervention, state, new_contacts)
        for contact in new_contacts
            _enforce_start_time!(intervention, contact)
        end
    end

    append!(state.individuals, new_contacts)
    n_infected = length(new_infected_ids)
    state.cumulative_cases += n_infected
    state.current_generation += 1

    for c in new_contacts
        if is_infected(c) && c.infection_time > state.max_infection_time
            state.max_infection_time = c.infection_time
        end
    end

    if n_infected == 0
        state.extinct = true
        state.active_ids = Int[]
    else
        # IDs are 1-based indices, so active_ids ARE the infected IDs
        state.active_ids = copy(new_infected_ids)
    end

    return state
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type offspring draw."""
_draw_offspring(rng::AbstractRNG, offspring::Distribution,
                individual, state::SimulationState) = rand(rng, offspring)

"""Function-based offspring draw. The function receives the RNG and individual."""
function _draw_offspring(rng::AbstractRNG, offspring::Function,
                         individual, state::SimulationState)
    return offspring(rng, individual)
end

# ── Contact creation ─────────────────────────────────────────────────

function _make_one_contact!(new_contacts, new_infected_ids, parent, state,
                             gt_dist, pop_suscept, residual,
                             interventions, next_id;
                             type_idx::Union{Int, Nothing}=nothing)
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
    type_idx !== nothing && (contact.state[:type] = type_idx)

    infected = _resolve_infection(state.rng, parent, contact,
                                   gt, pop_suscept, residual)
    contact.state[:infected] = infected

    push!(parent.secondary_case_ids, next_id)
    push!(new_contacts, contact)
    infected && push!(new_infected_ids, next_id)
    return next_id + 1
end

"""Single-type contacts."""
function _create_contacts!(new_contacts, new_infected_ids,
                           n_contacts::Int, parent, state,
                           gt_dist, pop_suscept, residual,
                           interventions, next_id)
    for _ in 1:n_contacts
        next_id = _make_one_contact!(new_contacts, new_infected_ids, parent, state,
                                      gt_dist, pop_suscept, residual,
                                      interventions, next_id)
    end
    return next_id
end

"""Multi-type contacts."""
function _create_contacts!(new_contacts, new_infected_ids,
                           counts::Vector{Int}, parent, state,
                           gt_dist, pop_suscept, residual,
                           interventions, next_id)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            next_id = _make_one_contact!(new_contacts, new_infected_ids, parent, state,
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
                             residual_transmission::Float64)
    pop_suscept < 1.0 && rand(rng) > pop_suscept && return false
    contact.susceptibility < 1.0 && rand(rng) > contact.susceptibility && return false
    parent.infectiousness < 1.0 && rand(rng) > parent.infectiousness && return false

    if is_isolated(parent)
        iso_t = isolation_time(parent)
        transmission_time = parent.infection_time + generation_time
        if transmission_time >= iso_t
            (residual_transmission <= 0.0 || rand(rng) > residual_transmission) && return false
        end
    end

    return true
end
