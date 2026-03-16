"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process using hazard-based transmission.
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_individuals = Individual[]
    next_id = state.cumulative_cases + 1

    pop_suscept = _susceptible_fraction(state)
    if pop_suscept <= 0.0
        state.extinct = true
        state.active_ids = Int[]
        return state
    end

    for idx in state.active_ids
        individual = state.individuals[idx]

        # Resolve interventions for this individual
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Get generation time distribution
        gt_dist = get_generation_time(model.generation_time, individual)

        # Compute effective transmission fraction
        frac = transmission_fraction(individual, gt_dist, interventions)
        frac *= individual.infectiousness

        # Draw potential offspring, scale for asymptomatic
        n_potential = rand(state.rng, model.offspring)
        if is_asymptomatic(individual) && state.asymptomatic_R_scaling < 1.0
            n_potential = rand(state.rng, Binomial(n_potential,
                                                    state.asymptomatic_R_scaling))
        end

        # Thin by transmission fraction (infectiousness side)
        n_offspring = _thin_offspring(state.rng, n_potential, frac)

        # Thin by susceptible fraction (population level)
        if pop_suscept < 1.0
            n_offspring = rand(state.rng, Binomial(n_offspring, pop_suscept))
        end

        # Create secondary cases
        for _ in 1:n_offspring
            gt = _sample_truncated_gt(state.rng, gt_dist, individual, frac)
            gt = max(gt, state.latent_period)
            inf_time = individual.infection_time + gt

            child = _create_individual(state, individual.id, individual.chain_id,
                                        next_id, inf_time, interventions)

            push!(individual.secondary_case_ids, next_id)
            push!(new_individuals, child)
            next_id += 1
        end
    end

    # Apply post-transmission interventions
    for intervention in interventions
        apply_post_transmission!(intervention, state, new_individuals)
    end

    # Update state
    append!(state.individuals, new_individuals)
    state.cumulative_cases += length(new_individuals)
    state.current_generation += 1

    if isempty(new_individuals)
        state.extinct = true
        state.active_ids = Int[]
    else
        start_idx = length(state.individuals) - length(new_individuals) + 1
        state.active_ids = collect(start_idx:length(state.individuals))
    end

    return state
end

# ── Internal helpers ─────────────────────────────────────────────────

function _thin_offspring(rng::AbstractRNG, n_potential::Int, frac::Float64)
    frac >= 1.0 && return n_potential
    frac <= 0.0 && return 0
    return rand(rng, Binomial(n_potential, frac))
end

function _sample_truncated_gt(rng::AbstractRNG, gen_time_dist::Distribution,
                               individual::Individual, frac::Float64)
    if frac >= 1.0
        return rand(rng, gen_time_dist)
    end
    u = rand(rng) * frac
    return quantile(gen_time_dist, u)
end
