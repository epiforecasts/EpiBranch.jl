"""
    simulate(model::TransmissionModel; interventions=[], sim_opts=SimOpts(), rng=Random.default_rng())

Run a single outbreak simulation using the given transmission model.
Returns a `SimulationState` containing all individuals and outbreak metadata.
"""
function simulate(model::TransmissionModel;
                  interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                  sim_opts::SimOpts=SimOpts(),
                  rng::AbstractRNG=Random.default_rng())
    state = initialise_state(model, sim_opts, interventions, rng)

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

function _get_population_size(model::BranchingProcess)
    model.population_size
end

function _get_population_size(model::TransmissionModel)
    nothing
end

function _get_n_types(model::BranchingProcess)
    model.n_types
end

function _get_n_types(model::TransmissionModel)
    1
end

function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
                          interventions, rng::AbstractRNG)
    individuals = Individual[]

    # Build a temporary state for intervention initialisation of index cases
    temp_state = SimulationState(
        individuals,
        Int[],
        0,
        rng,
        0,
        false,
        sim_opts.incubation_period,
        sim_opts.prob_asymptomatic,
        sim_opts.asymptomatic_R_scaling,
        sim_opts.test_sensitivity,
        sim_opts.latent_period,
        _get_population_size(model),
    )

    n_types = _get_n_types(model)
    for i in 1:sim_opts.n_initial
        ind = _create_individual(temp_state, 0, i, i, 0.0, interventions)
        # For multi-type, assign index cases to types uniformly at random
        if n_types > 1
            ind.state[:type] = rand(rng, 1:n_types)
        end
        push!(individuals, ind)
    end

    temp_state.cumulative_cases = sim_opts.n_initial
    temp_state.active_ids = collect(1:sim_opts.n_initial)

    return temp_state
end

function should_terminate(state::SimulationState, sim_opts::SimOpts)
    state.extinct && return true
    state.cumulative_cases >= sim_opts.max_cases && return true
    state.current_generation >= sim_opts.max_generations && return true

    if isfinite(sim_opts.max_time) && !isempty(state.individuals)
        latest = maximum(ind.infection_time for ind in state.individuals)
        latest >= sim_opts.max_time && return true
    end

    return false
end

"""
    _susceptible_fraction(state)

Compute the fraction of the population still susceptible. Returns 1.0
for infinite population (no depletion).
"""
function _susceptible_fraction(state::SimulationState)
    state.population_size === nothing && return 1.0
    n_susceptible = state.population_size - state.cumulative_cases
    n_susceptible <= 0 && return 0.0
    return n_susceptible / state.population_size
end

"""
    _create_individual(state, parent_id, chain_id, next_id, inf_time, interventions)

Create a new Individual with clinical state and intervention-initialised fields.
"""
function _create_individual(state::SimulationState, parent_id::Int,
                            chain_id::Int, next_id::Int,
                            inf_time::Float64, interventions)
    is_asymp = rand(state.rng) < state.prob_asymptomatic
    test_pos = !is_asymp && rand(state.rng) < state.test_sensitivity

    onset = if !is_asymp && state.incubation_period !== nothing
        inf_time + rand(state.rng, state.incubation_period)
    else
        NaN
    end

    s = Dict{Symbol, Any}(
        :onset_time => onset,
        :asymptomatic => is_asymp,
        :test_positive => test_pos,
        :infected => true,
    )

    ind = Individual(;
        id=next_id,
        parent_id=parent_id,
        generation=state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id=chain_id,
        infection_time=inf_time,
        state=s,
    )

    # Let each intervention initialise its own fields
    for intervention in interventions
        initialise_individual!(intervention, ind, state)
    end

    return ind
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
