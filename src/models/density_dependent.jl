"""
    step!(model::DensityDependent, state::SimulationState, interventions)

Process one generation of a density-dependent branching process.
Uses hazard-based transmission with additional binomial thinning by
susceptible fraction.
"""
function step!(model::DensityDependent, state::SimulationState, interventions)
    new_individuals = Individual[]
    next_id = state.cumulative_cases + 1

    # Susceptible fraction
    n_susceptible = model.population_size - state.cumulative_cases
    if n_susceptible <= 0
        state.extinct = true
        state.active_ids = Int[]
        return state
    end
    susceptible_fraction = n_susceptible / model.population_size

    for idx in state.active_ids
        individual = state.individuals[idx]

        # Resolve interventions
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Get individual-specific generation time
        gt_dist = get_generation_time(model.generation_time, individual)

        # Transmission fraction from hazard
        frac = transmission_fraction(individual, gt_dist, interventions)

        # Draw potential offspring, scale for asymptomatic, thin by frac
        n_potential = rand(state.rng, model.offspring)
        if individual.asymptomatic && state.asymptomatic_R_scaling < 1.0
            n_potential = rand(state.rng, Binomial(n_potential,
                                                    state.asymptomatic_R_scaling))
        end
        n_pre_iso = _thin_offspring(state.rng, n_potential, frac)

        # Further thin by susceptible fraction
        n_offspring = rand(state.rng, Binomial(n_pre_iso, susceptible_fraction))

        for _ in 1:n_offspring
            gt = _sample_truncated_gt(state.rng, gt_dist, individual, frac)
            gt = max(gt, state.latent_period)
            inf_time = individual.infection_time + gt

            onset_time = if state.incubation_period !== nothing
                inf_time + rand(state.rng, state.incubation_period)
            else
                NaN
            end

            child = _create_child(state, individual, next_id, inf_time, onset_time)

            push!(individual.secondary_case_ids, next_id)
            push!(new_individuals, child)
            next_id += 1
        end
    end

    # Post-transmission interventions
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
