"""
    simulate(model::TransmissionModel; interventions=[], sim_opts=SimOpts(), rng=Random.default_rng())

Run a single outbreak simulation using the given transmission model.
Returns a `SimulationState` containing all individuals and outbreak metadata.
"""
function simulate(model::TransmissionModel;
                  interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                  sim_opts::SimOpts=SimOpts(),
                  rng::AbstractRNG=Random.default_rng())
    state = initialise_state(model, sim_opts, rng)

    while !should_terminate(state, sim_opts)
        step!(model, state, interventions)
    end

    return state
end

"""
    simulate_batch(model, n; kwargs...)

Run `n` independent outbreak simulations. Returns a vector of `SimulationState`.
"""
function simulate_batch(model::TransmissionModel, n::Int;
                        interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                        sim_opts::SimOpts=SimOpts(),
                        rng::AbstractRNG=Random.default_rng())
    [simulate(model; interventions, sim_opts, rng) for _ in 1:n]
end

# ── Internal helpers ───────────────────────────────────────────────

function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
                          rng::AbstractRNG)
    individuals = Individual[]

    for i in 1:sim_opts.n_initial
        is_asymp = rand(rng) < sim_opts.prob_asymptomatic
        test_pos = !is_asymp && rand(rng) < sim_opts.test_sensitivity

        onset = if !is_asymp && sim_opts.incubation_period !== nothing
            rand(rng, sim_opts.incubation_period)
        else
            NaN
        end

        push!(individuals, Individual(;
            id=i,
            parent_id=0,
            generation=0,
            chain_id=i,
            infection_time=0.0,
            onset_time=onset,
            asymptomatic=is_asymp,
            test_positive=test_pos,
        ))
    end

    SimulationState(
        individuals,
        collect(1:sim_opts.n_initial),
        0,
        rng,
        sim_opts.n_initial,
        false,
        sim_opts.incubation_period,
        sim_opts.prob_asymptomatic,
        sim_opts.asymptomatic_R_scaling,
        sim_opts.test_sensitivity,
        sim_opts.latent_period,
    )
end

function should_terminate(state::SimulationState, sim_opts::SimOpts)
    state.extinct && return true
    state.cumulative_cases >= sim_opts.max_cases && return true
    state.current_generation >= sim_opts.max_generations && return true

    # Time-based termination
    if isfinite(sim_opts.max_time) && !isempty(state.individuals)
        latest = maximum(ind.infection_time for ind in state.individuals)
        latest >= sim_opts.max_time && return true
    end

    return false
end

"""
    _create_child(state, parent, next_id, inf_time, onset_time)

Create a new Individual with asymptomatic status and test sensitivity
drawn from state parameters. Internal helper used by step! functions.
"""
function _create_child(state::SimulationState,
                       parent::Individual, next_id::Int,
                       inf_time::Float64, onset_time::Float64)
    is_asymp = rand(state.rng) < state.prob_asymptomatic
    test_pos = !is_asymp && rand(state.rng) < state.test_sensitivity

    # Asymptomatic cases have no onset
    actual_onset = is_asymp ? NaN : onset_time

    Individual(;
        id=next_id,
        parent_id=parent.id,
        generation=state.current_generation + 1,
        chain_id=parent.chain_id,
        infection_time=inf_time,
        onset_time=actual_onset,
        asymptomatic=is_asymp,
        test_positive=test_pos,
    )
end

"""
    simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                         max_attempts=10_000, kwargs...)

Run simulations until one produces an outbreak with total cases within `size_range`.
Uses rejection sampling. Returns the successful `SimulationState`.

Throws an error if no valid simulation is found within `max_attempts`.
"""
function simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                              max_attempts::Int=10_000,
                              interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                              sim_opts::SimOpts=SimOpts(),
                              rng::AbstractRNG=Random.default_rng())
    for _ in 1:max_attempts
        state = simulate(model; interventions, sim_opts, rng)
        state.cumulative_cases in size_range && return state
    end
    throw(ErrorException(
        "No simulation produced an outbreak of size $size_range within $max_attempts attempts"
    ))
end
