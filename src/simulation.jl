"""
    simulate(model::TransmissionModel; interventions=[], init=nothing,
             sim_opts=SimOpts(), rng=Random.default_rng())

Run a single outbreak simulation.

- `model`: transmission model (e.g. `BranchingProcess`)
- `interventions`: vector of `AbstractIntervention`
- `init`: function `(rng, individual) -> nothing` that sets attributes on
  new individuals (clinical state, demographics, etc.), or `nothing`
- `sim_opts`: simulation control (max_cases, max_generations, etc.)
- `rng`: random number generator
"""
function simulate(model::TransmissionModel;
                  interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                  init::Union{Function, Nothing}=nothing,
                  sim_opts::SimOpts=SimOpts(),
                  rng::AbstractRNG=Random.default_rng())
    state = initialise_state(model, sim_opts, interventions, init, rng)

    # Validate required fields after first individual is initialised
    if !isempty(state.individuals)
        _validate_required_fields(state.individuals[1], interventions)
    end

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
                        init::Union{Function, Nothing}=nothing,
                        sim_opts::SimOpts=SimOpts(),
                        rng::AbstractRNG=Random.default_rng())
    [simulate(model; interventions, init, sim_opts, rng) for _ in 1:n]
end

"""
    simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                         max_attempts=10_000, kwargs...)

Run simulations until one produces an outbreak within `size_range`.
"""
function simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                              max_attempts::Int=10_000,
                              interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                              init::Union{Function, Nothing}=nothing,
                              sim_opts::SimOpts=SimOpts(),
                              rng::AbstractRNG=Random.default_rng())
    for _ in 1:max_attempts
        state = simulate(model; interventions, init, sim_opts, rng)
        state.cumulative_cases in size_range && return state
    end
    throw(ErrorException(
        "No simulation produced an outbreak of size $size_range within $max_attempts attempts"
    ))
end

# ── Internal helpers ───────────────────────────────────────────────

function _get_population_size(model::BranchingProcess)
    model.population_size
end

function _get_population_size(model::TransmissionModel)
    nothing
end

function _get_latent_period(model::BranchingProcess)
    model.latent_period
end

function _get_latent_period(model::TransmissionModel)
    0.0
end

function _get_n_types(model::BranchingProcess)
    model.n_types
end

function _get_n_types(model::TransmissionModel)
    1
end

function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
                          interventions, init, rng::AbstractRNG)
    individuals = Individual[]

    temp_state = SimulationState(
        individuals,
        Int[],
        0,
        rng,
        0,
        false,
        _get_population_size(model),
        _get_latent_period(model),
        init,
    )

    n_types = _get_n_types(model)
    for i in 1:sim_opts.n_initial
        ind = _create_individual(temp_state, 0, i, i, 0.0, interventions)
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

Create a new Individual. The `state.init` function (if provided) sets
clinical/demographic attributes. Then each intervention's `initialise_individual!`
sets intervention state.
"""
function _create_individual(state::SimulationState, parent_id::Int,
                            chain_id::Int, next_id::Int,
                            inf_time::Float64, interventions)
    s = Dict{Symbol, Any}(:infected => true)

    ind = Individual(;
        id=next_id,
        parent_id=parent_id,
        generation=state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id=chain_id,
        infection_time=inf_time,
        state=s,
    )

    # User-provided init function sets clinical/demographic attributes
    if state.init !== nothing
        state.init(state.rng, ind)
    end

    # Each intervention initialises its own fields
    for intervention in interventions
        initialise_individual!(intervention, ind, state)
    end

    return ind
end

# ── Init function constructors ───────────────────────────────────────

"""
    clinical_presentation(; incubation_period, prob_asymptomatic=0.0,
                           test_sensitivity=1.0)

Return an init function that sets clinical attributes on individuals:
`:onset_time`, `:asymptomatic`, `:test_positive`.

These fields are required by [`Isolation`](@ref) and read by
[`ContactTracing`](@ref).
"""
function clinical_presentation(; incubation_period::Distribution,
                                 prob_asymptomatic::Real=0.0,
                                 test_sensitivity::Real=1.0)
    pa = Float64(prob_asymptomatic)
    ts = Float64(test_sensitivity)
    return function (rng, ind)
        is_asymp = rand(rng) < pa
        ind.state[:asymptomatic] = is_asymp
        ind.state[:test_positive] = !is_asymp && rand(rng) < ts
        ind.state[:onset_time] = if !is_asymp
            ind.infection_time + rand(rng, incubation_period)
        else
            NaN
        end
    end
end

"""
    demographics(; age_distribution=nothing, age_range=(0, 90), prob_female=0.5)

Return an init function that sets demographic attributes on individuals:
`:age`, `:sex`.
"""
function demographics(; age_distribution::Union{Distribution, Nothing}=nothing,
                        age_range::Tuple{Int, Int}=(0, 90),
                        prob_female::Real=0.5)
    pf = Float64(prob_female)
    return function (rng, ind)
        ind.state[:age] = if age_distribution !== nothing
            clamp(floor(Int, rand(rng, age_distribution)), age_range...)
        else
            rand(rng, age_range[1]:age_range[2])
        end
        ind.state[:sex] = rand(rng) < pf ? :female : :male
    end
end

"""
    compose(fs...)

Compose multiple init functions into one. Functions are called in order.

```julia
init = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15)),
)
```
"""
compose(fs...) = (rng, ind) -> for f in fs; f(rng, ind); end

# ── Intervention field validation ────────────────────────────────────

"""
    required_fields(intervention)

Return a vector of field names (Symbols) that this intervention requires
on individuals. Default: empty (no requirements).
"""
required_fields(::AbstractIntervention) = Symbol[]

"""Validate that all required fields are present on an individual."""
function _validate_required_fields(individual, interventions)
    for intervention in interventions
        for field in required_fields(intervention)
            if !haskey(individual.state, field)
                itype = typeof(intervention)
                hint = _field_hint(field)
                error(
                    "$itype requires field :$field on individuals. $hint"
                )
            end
        end
    end
end

function _field_hint(field::Symbol)
    hints = Dict(
        :onset_time => "Provide init = clinical_presentation(incubation_period = ...).",
        :asymptomatic => "Provide init = clinical_presentation(incubation_period = ...).",
        :test_positive => "Provide init = clinical_presentation(incubation_period = ...).",
        :age => "Provide init = demographics(age_distribution = ...).",
        :sex => "Provide init = demographics(...).",
    )
    return get(hints, field, "Set this field via an init function.")
end
