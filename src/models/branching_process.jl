"""
    step!(model::BranchingProcess, state::SimulationState, interventions)

Process one generation of the branching process using hazard-based transmission.

For each active individual:
1. Resolve interventions (set isolation time etc.)
2. Get individual-specific generation time distribution
3. Compute transmission fraction from generation time CDF at isolation time
4. Draw potential offspring, thin by transmission fraction
5. Assign infection times from generation time distribution (truncated at isolation)
6. Apply post-transmission interventions (contact tracing)
"""
function step!(model::BranchingProcess, state::SimulationState, interventions)
    new_individuals = Individual[]
    next_id = state.cumulative_cases + 1

    for idx in state.active_ids
        individual = state.individuals[idx]

        # Step 1: resolve interventions for this individual
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end

        # Step 2: get generation time distribution for this individual
        gt_dist = get_generation_time(model.generation_time, individual)

        # Step 3: compute fraction of transmission before intervention
        frac = transmission_fraction(individual, gt_dist, interventions)

        # Step 4: draw potential offspring, scale for asymptomatic, thin by frac
        n_potential = rand(state.rng, model.offspring)
        if individual.asymptomatic && state.asymptomatic_R_scaling < 1.0
            n_potential = rand(state.rng, Binomial(n_potential,
                                                    state.asymptomatic_R_scaling))
        end
        n_offspring = _thin_offspring(state.rng, n_potential, frac)

        # Step 5: create secondary cases with generation times
        for _ in 1:n_offspring
            gt = _sample_truncated_gt(state.rng, gt_dist, individual, frac)
            # Enforce latent period
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

    # Step 6: apply post-transmission interventions (e.g. contact tracing)
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
