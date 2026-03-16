"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process using hazard-based transmission.

For each active individual:
1. Resolve interventions (set isolation time, susceptibility, infectiousness)
2. Get individual-specific generation time distribution
3. Compute effective transmission: hazard truncation × infectiousness
4. Draw potential offspring, thin by transmission fraction
5. Thin by susceptible fraction (if finite population)
6. Create secondary cases, applying offspring susceptibility
7. Apply post-transmission interventions (contact tracing, vaccination)
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_individuals = Individual[]
    next_id = state.cumulative_cases + 1

    # Population-level susceptible fraction (1.0 for infinite population)
    pop_suscept = _susceptible_fraction(state)
    if pop_suscept <= 0.0
        state.extinct = true
        state.active_ids = Int[]
        return state
    end

    for idx in state.active_ids
        individual = state.individuals[idx]

        # Step 1: resolve interventions for this individual
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Step 2: get generation time distribution
        gt_dist = get_generation_time(model.generation_time, individual)

        # Step 3: compute effective transmission fraction
        frac = transmission_fraction(individual, gt_dist, interventions)
        frac *= individual.infectiousness

        # Step 4: draw potential offspring, scale for asymptomatic
        n_potential = rand(state.rng, model.offspring)
        if individual.asymptomatic && state.asymptomatic_R_scaling < 1.0
            n_potential = rand(state.rng, Binomial(n_potential,
                                                    state.asymptomatic_R_scaling))
        end

        # Thin by transmission fraction (infectiousness side)
        n_offspring = _thin_offspring(state.rng, n_potential, frac)

        # Step 5: thin by susceptible fraction (population level)
        if pop_suscept < 1.0
            n_offspring = rand(state.rng, Binomial(n_offspring, pop_suscept))
        end

        # Step 6: create secondary cases
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

    # Step 7: apply post-transmission interventions
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

"""
Thin n_potential offspring by transmission fraction. Each potential
offspring has probability `frac` of occurring before isolation.
"""
function _thin_offspring(rng::AbstractRNG, n_potential::Int, frac::Float64)
    frac >= 1.0 && return n_potential
    frac <= 0.0 && return 0
    return rand(rng, Binomial(n_potential, frac))
end

"""
Sample a generation time from the distribution, truncated to [0, t_iso - t_inf]
if the individual is isolated. Uses inverse CDF sampling for truncation.
"""
function _sample_truncated_gt(rng::AbstractRNG, gen_time_dist::Distribution,
                               individual::Individual, frac::Float64)
    if frac >= 1.0
        return rand(rng, gen_time_dist)
    end
    # Inverse CDF sampling: U ~ Uniform(0, frac), then quantile(gen_time_dist, U)
    u = rand(rng) * frac
    return quantile(gen_time_dist, u)
end
