"""
    simulate(model::TransmissionModel; interventions=[], attributes=nothing,
             sim_opts=SimOpts(), rng=Random.default_rng())

Run a single outbreak simulation.
"""
function simulate(model::TransmissionModel;
                  interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                  attributes::Union{Function, NoAttributes}=NoAttributes(),
                  sim_opts::SimOpts=SimOpts(),
                  rng::AbstractRNG=Random.default_rng())
    state = initialise_state(model, sim_opts, interventions, attributes, rng)


    if !isempty(state.individuals)
        _validate_required_fields(state.individuals[1], interventions)
    end

    while !should_terminate(state, sim_opts)
        step!(model, state, interventions)
    end

    return state
end

"""
    simulate_batch(model, n; parallel=false, kwargs...)

Run `n` independent outbreak simulations.

When `parallel=true`, simulations are distributed across available threads
using independent RNG streams derived from the provided `rng`. Use
`julia --threads N` to enable multi-threading.
"""
function simulate_batch(model::TransmissionModel, n::Int;
                        interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                        attributes::Union{Function, NoAttributes}=NoAttributes(),
                        sim_opts::SimOpts=SimOpts(),
                        rng::AbstractRNG=Random.default_rng(),
                        parallel::Bool=false)
    if parallel && Threads.nthreads() > 1
        # Derive independent RNG streams for each simulation
        seeds = [rand(rng, UInt64) for _ in 1:n]
        results = Vector{SimulationState}(undef, n)
        Threads.@threads for i in 1:n
            local_rng = Random.Xoshiro(seeds[i])
            results[i] = simulate(model; interventions, attributes, sim_opts, rng=local_rng)
        end
        return results
    else
        return [simulate(model; interventions, attributes, sim_opts, rng) for _ in 1:n]
    end
end

"""
    simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                         max_attempts=10_000, kwargs...)

Run simulations until one produces an outbreak within `size_range`.
"""
function simulate_conditioned(model::TransmissionModel, size_range::UnitRange{Int};
                              max_attempts::Int=10_000,
                              interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                              attributes::Union{Function, NoAttributes}=NoAttributes(),
                              sim_opts::SimOpts=SimOpts(),
                              rng::AbstractRNG=Random.default_rng())
    for _ in 1:max_attempts
        state = simulate(model; interventions, attributes, sim_opts, rng)
        state.cumulative_cases in size_range && return state
    end
    throw(ErrorException(
        "No simulation produced an outbreak of size $size_range within $max_attempts attempts"
    ))
end

# ── Internal helpers ───────────────────────────────────────────────

function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
                          interventions, attributes, rng::AbstractRNG)
    individuals = Individual[]

    temp_state = SimulationState(
        individuals,
        Int[],
        0,
        rng,
        0,
        false,
        population_size(model),
        latent_period(model),
        0.0,
        attributes,
    )

    nt = n_types(model)
    for i in 1:sim_opts.n_initial
        ind = _create_individual(temp_state, 0, i, i, 0.0, interventions)
        if nt > 1
            ind.state[:type] = rand(rng, 1:nt)
        end
        push!(individuals, ind)
    end

    temp_state.cumulative_cases = sim_opts.n_initial
    temp_state.active_ids = collect(1:sim_opts.n_initial)

    return temp_state
end

# track max_infection_time incrementally instead of scanning
function should_terminate(state::SimulationState, sim_opts::SimOpts)
    state.extinct && return true
    state.cumulative_cases >= sim_opts.max_cases && return true
    state.current_generation >= sim_opts.max_generations && return true
    isfinite(sim_opts.max_time) && state.max_infection_time >= sim_opts.max_time && return true
    return false
end

"""Fraction of the population still susceptible (1.0 for infinite population)."""
_susceptible_fraction(state::SimulationState{<:Any, NoPopulation}) = 1.0

function _susceptible_fraction(state::SimulationState{<:Any, Int})
    n_susceptible = state.population_size - state.cumulative_cases
    n_susceptible <= 0 && return 0.0
    return n_susceptible / state.population_size
end

"""Create a new Individual with attributes and intervention state."""
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

    _apply_attributes!(state.attributes, state.rng, ind)

    for intervention in interventions
        initialise_individual!(intervention, ind, state)
    end

    return ind
end

"""Apply attributes function to an individual. No-op for NoAttributes."""
_apply_attributes!(::NoAttributes, rng, ind) = nothing
_apply_attributes!(f::Function, rng, ind) = f(rng, ind)

# ── Attributes function constructors ─────────────────────────────────

"""
    clinical_presentation(; incubation_period, prob_asymptomatic=0.0)

Return an attributes function. `:onset_time` and `:asymptomatic` are
set on each individual. Required by [`Isolation`](@ref).
"""
function clinical_presentation(; incubation_period::Distribution,
                                 prob_asymptomatic::Real=0.0)
    pa = float(prob_asymptomatic)
    return function (rng, ind)
        is_asymp = rand(rng) < pa
        ind.state[:asymptomatic] = is_asymp
        ind.state[:onset_time] = if !is_asymp
            ind.infection_time + rand(rng, incubation_period)
        else
            NaN
        end
    end
end

"""
    Disease(; incubation_period, prob_asymptomatic=0.0)

Convenience wrapper for specifying disease properties. Can be passed
directly as the `attributes` argument to [`simulate`](@ref).

Sets `:onset_time` and `:asymptomatic` on each individual. Equivalent
to `clinical_presentation(; incubation_period, prob_asymptomatic)` but
with a name that reflects what is being specified.

```julia
disease = Disease(incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3)
simulate(model; attributes = disease, interventions = [iso])
```
"""
function Disease(; incubation_period::Distribution, prob_asymptomatic::Real=0.0)
    clinical_presentation(; incubation_period, prob_asymptomatic)
end

"""
    demographics(; age_distribution=nothing, age_range=(0, 90), prob_female=0.5)

Return an attributes function. `:age` and `:sex` are set on each individual.
"""
function demographics(; age_distribution::Union{Distribution, NoAgeDistribution}=NoAgeDistribution(),
                        age_range::Tuple{Int, Int}=(0, 90),
                        prob_female::Real=0.5)
    pf = float(prob_female)
    return function (rng, ind)
        ind.state[:age] = _sample_age(rng, age_distribution, age_range)
        ind.state[:sex] = rand(rng) < pf ? :female : :male
    end
end

_sample_age(rng, ::NoAgeDistribution, age_range) = rand(rng, age_range[1]:age_range[2])
_sample_age(rng, dist::Distribution, age_range) = clamp(floor(Int, rand(rng, dist)), age_range...)

"""
    compose(fs...)

Compose multiple attributes functions into one, called in order.
"""
compose(fs...) = (rng, ind) -> for f in fs; f(rng, ind); end

# ── Intervention field validation ────────────────────────────────────

"""Fields that an intervention requires on individuals. Default: none."""
required_fields(::AbstractIntervention) = Symbol[]

"""Check that all required fields are present on an individual."""
function _validate_required_fields(individual, interventions)
    for intervention in interventions
        for field in required_fields(intervention)
            if !haskey(individual.state, field)
                itype = typeof(intervention)
                hint = _field_hint(field)
                error("$itype requires field :$field on individuals. $hint")
            end
        end
    end
end

function _field_hint(field::Symbol)
    hints = Dict(
        :onset_time => "Provide attributes = Disease(incubation_period = ...).",
        :asymptomatic => "Provide attributes = Disease(incubation_period = ...).",
        :age => "Provide attributes = demographics(age_distribution = ...).",
        :sex => "Provide attributes = demographics(...).",
    )
    return get(hints, field, "Set this field via an attributes function.")
end
