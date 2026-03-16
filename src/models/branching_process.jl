"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process.

The offspring draw is fully decoupled from timing and interventions:

1. Resolve interventions on each active (infecting) individual
2. Draw ALL potential contacts from the offspring distribution
3. Create contact individuals with generation times
4. Competing risks: determine which contacts are infected
   (generation time < isolation time of parent, susceptibility check)
5. Apply post-transmission interventions to ALL contacts
6. Only infected contacts become active in the next generation

All contacts (infected and non-infected) are stored in state.individuals
for tracking intervention effort, contacts tables, etc.
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

        # Step 1: resolve interventions on the parent (sets isolation_time etc.)
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Step 2: draw ALL potential contacts
        n_contacts = _draw_offspring(state.rng, model.offspring, individual, state)

        # Step 3 & 4: create contacts and determine infection via competing risks
        gt_dist = get_generation_time(model.generation_time, individual)
        residual = _residual_fraction(interventions)

        for _ in 1:n_contacts
            # Sample generation time (potential time of transmission)
            gt = rand(state.rng, gt_dist)
            gt = max(gt, state.latent_period)
            inf_time = individual.infection_time + gt

            # Create the contact individual
            contact = _create_individual(state, individual.id, individual.chain_id,
                                          next_id, inf_time, interventions)

            # Competing risk: was transmission successful?
            infected = _resolve_infection(state.rng, individual, contact,
                                           gt, pop_suscept, residual)
            contact.state[:infected] = infected

            push!(individual.secondary_case_ids, next_id)
            push!(new_contacts, contact)

            if infected
                push!(new_infected_indices, next_id)
            end

            next_id += 1
        end
    end

    # Step 5: apply post-transmission interventions to ALL contacts
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
        # Map next_ids to indices in state.individuals
        base_idx = length(state.individuals) - length(new_contacts)
        state.active_ids = [base_idx + findfirst(==(nid), [c.id for c in new_contacts])
                           for nid in new_infected_indices]
    end

    return state
end

# ── Internal helpers ─────────────────────────────────────────────────

"""
Draw offspring count, applying asymptomatic scaling if relevant.
Dispatches on offspring type for future multi-type support.
"""
function _draw_offspring(rng::AbstractRNG, offspring::Distribution,
                         individual, state::SimulationState)
    n = rand(rng, offspring)
    if is_asymptomatic(individual) && state.asymptomatic_R_scaling < 1.0
        n = rand(rng, Binomial(n, state.asymptomatic_R_scaling))
    end
    return n
end

"""
Determine whether a contact is successfully infected, based on
competing risks (generation time vs isolation/intervention timing)
and susceptibility.

Returns true if infected, false if the contact was made but transmission
did not occur.
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
            # Transmission after isolation — only succeeds with residual probability
            if residual_transmission <= 0.0 || rand(rng) > residual_transmission
                return false
            end
        end
    end

    return true
end
