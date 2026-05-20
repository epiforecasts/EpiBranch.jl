"""
    simulate(model::TransmissionModel; interventions=[], transitions=[],
             attributes=nothing, sim_opts=SimOpts(),
             rng=Random.default_rng(),
             condition=nothing, max_attempts=10_000)

Run a single outbreak simulation.

`transitions` is a vector of [`AbstractClinicalTransition`](@ref)s
(e.g. [`Reporting`](@ref), [`Hospitalisation`](@ref), [`Death`](@ref),
[`Recovery`](@ref)) that act on each case's clinical timeline after
attributes and interventions have run. Build the vector explicitly —
each transition's `probability` and `delay` accept either constants or
`(rng, ind) -> value` functions, so age- or risk-conditional rates and
delays are configured per-transition with no special-cased defaults.

If `condition` is provided (a `UnitRange{Int}`), simulations are repeated
until one produces an outbreak whose cumulative cases fall within the range,
up to `max_attempts`.
"""
function simulate(model::TransmissionModel;
        interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
        transitions::Vector{<:AbstractClinicalTransition} = AbstractClinicalTransition[],
        attributes::Union{Function, NoAttributes} = NoAttributes(),
        sim_opts::SimOpts = SimOpts(),
        rng::AbstractRNG = Random.default_rng(),
        condition::Union{UnitRange{Int}, Nothing} = nothing,
        max_attempts::Int = 10_000)
    if condition !== nothing
        for _ in 1:max_attempts
            state = simulate(model; interventions, transitions,
                attributes, sim_opts, rng)
            state.cumulative_cases in condition && return state
        end
        throw(ErrorException(
            "No simulation produced an outbreak of size $condition within $max_attempts attempts"
        ))
    end

    state = initialise_state(
        model, sim_opts, interventions, transitions, attributes, rng)
    _resolve_new_transitions!(state, 0)

    while !should_terminate(state, sim_opts)
        pre = length(state.individuals)
        new_contacts = step!(model, state, interventions)
        _resolve_competing_risks!(state, new_contacts, interventions)
        _register_step!(state, new_contacts)
        _resolve_new_transitions!(state, pre)
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
        interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
        transitions::Vector{<:AbstractClinicalTransition} = AbstractClinicalTransition[],
        attributes::Union{Function, NoAttributes} = NoAttributes(),
        sim_opts::SimOpts = SimOpts(),
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false)
    if parallel && Threads.nthreads() > 1
        # Derive independent RNG streams for each simulation
        seeds = [rand(rng, UInt64) for _ in 1:n]
        results = Vector{SimulationState}(undef, n)
        Threads.@threads for i in 1:n
            local_rng = Random.Xoshiro(seeds[i])
            results[i] = simulate(model; interventions, transitions,
                attributes, sim_opts, rng = local_rng)
        end
        return results
    else
        return [simulate(model; interventions, transitions, attributes,
                    sim_opts, rng) for _ in 1:n]
    end
end

# ── Internal helpers ───────────────────────────────────────────────

function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
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
        convert(Vector{AbstractClinicalTransition}, transitions)
    )

    nt = n_types(model)
    for i in 1:sim_opts.n_initial
        ind = _create_individual(temp_state, 0, i, i, 0.0, interventions)
        if nt > 1
            ind.state[:type] = rand(rng, 1:nt)
        end
        # Validate required fields on the first individual, before any
        # transition resolution runs. Without this, a missing :onset_time
        # surfaces as an opaque error inside the transition closure rather
        # than as the engine's friendly "Reporting requires field :onset_time"
        # message.
        if i == 1
            _validate_required_fields(ind, interventions)
            _validate_required_fields(ind, transitions)
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
    isfinite(sim_opts.max_time) && state.max_infection_time >= sim_opts.max_time &&
        return true
    return false
end

"""Fraction of the population still susceptible (1.0 for infinite
population). The optional `extra_infected` accounts for contacts
already infected within the current step but not yet registered."""
function _susceptible_fraction(state::SimulationState{<:Any, NoPopulation},
        extra_infected::Int = 0)
    1.0
end

function _susceptible_fraction(state::SimulationState{<:Any, Int},
        extra_infected::Int = 0)
    n_susceptible = state.population_size - state.cumulative_cases - extra_infected
    n_susceptible <= 0 && return 0.0
    return n_susceptible / state.population_size
end

"""Create a new Individual with attributes and intervention state.
Clinical transitions are resolved separately by the engine, which
sweeps newly added individuals after each `step!` returns.
"""
function _create_individual(state::SimulationState, parent_id::Int,
        chain_id::Int, next_id::Int,
        inf_time::Float64, interventions)
    s = Dict{Symbol, Any}(:infected => true)

    ind = Individual(;
        id = next_id,
        parent_id = parent_id,
        generation = state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id = chain_id,
        infection_time = inf_time,
        state = s
    )

    _apply_attributes!(state.attributes, state.rng, ind)

    for intervention in interventions
        initialise_individual!(intervention, ind, state)
    end

    return ind
end

"""Run init, resolve, and terminal-arbitration for every clinical
transition on `state.transitions` against `individual`. Called from the
simulation loop after the individual's `:type` is set (multi-type) and
after `_create_individual` has set attributes and intervention state.
"""
function _resolve_transitions!(state::SimulationState, individual)
    transitions = state.transitions
    isempty(transitions) && return nothing
    for transition in transitions
        initialise_individual!(transition, individual, state)
    end
    for transition in transitions
        resolve_individual!(transition, individual, state)
    end
    _finalise_terminal!(individual, transitions)
    return nothing
end

"""Decide `:infected` for each newly produced contact by composing
competing risks. Built-in risks (per-individual susceptibility,
parent infectiousness, population-level susceptibility for
finite-population models) are applied first; any [`competing_risk`](@ref)
contributed by an intervention is then applied in stack order. A risk
that has fired by transmission time blocks transmission with its
`block_probability`; transmission succeeds iff no risk blocks it.

Population susceptibility is recomputed per contact so that, as
contacts within the same step get infected, the susceptible pool
shrinks accordingly. This prevents the cumulative case count from
overshooting a finite `population_size`."""
function _resolve_competing_risks!(state::SimulationState, new_contacts, interventions)
    infected_so_far = 0
    for contact in new_contacts
        infected = _decide_infected(state, contact, interventions, infected_so_far)
        contact.state[:infected] = infected
        infected && (infected_so_far += 1)
    end
    return nothing
end

function _decide_infected(state::SimulationState, contact::Individual,
        interventions, infected_so_far::Int)
    contact.parent_id == 0 && return true
    rng = state.rng
    parent = state.individuals[contact.parent_id]
    transmission_time = contact.infection_time

    # Built-in risks (time-agnostic Bernoulli draws). Population
    # susceptibility reflects infections accumulated this step.
    pop_suscept = _susceptible_fraction(state, infected_so_far)
    pop_suscept <= 0.0 && return false
    pop_suscept < 1.0 && rand(rng) > pop_suscept && return false
    contact.susceptibility < 1.0 && rand(rng) > contact.susceptibility && return false
    parent.infectiousness < 1.0 && rand(rng) > parent.infectiousness && return false

    # Intervention-contributed risks.
    for intervention in interventions
        risk = competing_risk(intervention, parent, contact, state)
        risk === nothing && continue
        event_t = _resolve_risk_value(risk.event_time, rng, parent, contact, state)
        event_t > transmission_time && continue
        prob = _resolve_risk_value(risk.block_probability, rng, parent, contact, state)
        prob <= 0.0 && continue
        prob >= 1.0 && return false
        rand(rng) < prob && return false
    end
    return true
end

"""Append the new contacts returned by `step!` to `state.individuals`
and update the bookkeeping fields: `cumulative_cases`,
`current_generation`, `max_infection_time`, `active_ids`, `extinct`."""
function _register_step!(state::SimulationState, new_contacts)
    append!(state.individuals, new_contacts)
    new_infected_ids = Int[]
    for ind in new_contacts
        is_infected(ind) || continue
        push!(new_infected_ids, ind.id)
        if ind.infection_time > state.max_infection_time
            state.max_infection_time = ind.infection_time
        end
    end
    state.cumulative_cases += length(new_infected_ids)
    state.current_generation += 1
    if isempty(new_infected_ids)
        state.extinct = true
        state.active_ids = Int[]
    else
        state.active_ids = new_infected_ids
    end
    return nothing
end

"""Sweep newly added infected individuals (those at indices
`from_index+1:end`) and run clinical transitions on each. Called by
[`simulate`](@ref) after `initialise_state` and after each `step!` so
that authors of custom [`TransmissionModel`](@ref) subtypes do not
need to invoke transition resolution from inside their `step!`."""
function _resolve_new_transitions!(state::SimulationState, from_index::Int)
    isempty(state.transitions) && return nothing
    @inbounds for i in (from_index + 1):length(state.individuals)
        ind = state.individuals[i]
        if get(ind.state, :infected, false)
            _resolve_transitions!(state, ind)
        end
    end
    return nothing
end

"""Apply attributes function to an individual. No-op for NoAttributes."""
_apply_attributes!(::NoAttributes, rng, ind) = nothing
_apply_attributes!(f::Function, rng, ind) = f(rng, ind)

# ── Attributes function constructors ─────────────────────────────────

"""
    clinical_presentation(; incubation_period, prob_asymptomatic = 0.0)

Return an attributes function that sets `:onset_time` and
`:asymptomatic` on each individual.

For symptomatic cases, `:onset_time = infection_time + rand(incubation_period)`.
For asymptomatic cases (drawn with probability `prob_asymptomatic`),
`:onset_time = NaN` and `:asymptomatic = true`. Required by
[`Isolation`](@ref) and used by [`linelist`](@ref) to populate
`date_onset`.

# Examples

Symptomatic-only with a log-normal incubation period:

```julia
attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
```

With 30% asymptomatic:

```julia
attributes = clinical_presentation(
    incubation_period = LogNormal(1.6, 0.5),
    prob_asymptomatic = 0.3,
)
```

Compose with other attributes (e.g. demographics and a per-contact
infection probability):

```julia
attributes = compose(
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
    (rng, ind) -> (ind.susceptibility = 0.3),
)
```

See also [`demographics`](@ref), [`compose`](@ref).
"""
function clinical_presentation(; incubation_period::Distribution,
        prob_asymptomatic::Real = 0.0)
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
    demographics(; age_distribution=nothing, age_range=(0, 90), prob_female=0.5)

Return an attributes function. `:age` and `:sex` are set on each individual.
"""
function demographics(;
        age_distribution::Union{Distribution, NoAgeDistribution} = NoAgeDistribution(),
        age_range::Tuple{Int, Int} = (0, 90),
        prob_female::Real = 0.5)
    pf = float(prob_female)
    return function (rng, ind)
        ind.state[:age] = _sample_age(rng, age_distribution, age_range)
        ind.state[:sex] = rand(rng) < pf ? :female : :male
    end
end

_sample_age(rng, ::NoAgeDistribution, age_range) = rand(rng, age_range[1]:age_range[2])
function _sample_age(rng, dist::Distribution, age_range)
    clamp(floor(Int, rand(rng, dist)), age_range...)
end

"""
    transmission_traits(; susceptibility = 1.0, infectiousness = 1.0)

Return an attributes function that sets `susceptibility` (per-contact
probability of infection given exposure) and `infectiousness` (parent-side
modifier on transmission) on each individual.

Each argument accepts:

- a `Real`: assigned directly to every individual.
- a `Distribution`: sampled per individual via `rand(rng, dist)`.
- a `Function` `(rng, ind) -> value`: called per individual; the
  returned value is assigned. Use this for attribute-dependent rules
  (e.g. age-conditional susceptibility) — place the builder after
  `demographics` in `compose` so `ind.state[:age]` is set first.

Both default to `1.0` (no Bernoulli filtering in the transmission model).

# Examples

Constant per-contact infection probability:

```julia
attributes = transmission_traits(susceptibility = 0.3)
```

Per-individual heterogeneity:

```julia
attributes = transmission_traits(
    susceptibility = Beta(2, 5),
    infectiousness = Beta(8, 2),
)
```

Age-conditional susceptibility (compose after `demographics`):

```julia
attributes = compose(
    demographics(age_distribution = Uniform(0, 90)),
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:age] >= 65 ? 0.8 : 0.3,
    ),
)
```

The closure form `(rng, ind) -> (ind.susceptibility = ...)` inside
[`compose`](@ref) remains available as an escape hatch for cases this
builder does not cover.

See also [`clinical_presentation`](@ref), [`demographics`](@ref),
[`compose`](@ref).
"""
function transmission_traits(;
        susceptibility::Union{Real, Distribution, Function} = 1.0,
        infectiousness::Union{Real, Distribution, Function} = 1.0)
    sus = _trait_sampler(susceptibility)
    inf = _trait_sampler(infectiousness)
    return function (rng, ind)
        ind.susceptibility = sus(rng, ind)
        ind.infectiousness = inf(rng, ind)
    end
end

_trait_sampler(x::Real) =
    let v = float(x)
        (rng, ind) -> v
    end
_trait_sampler(d::Distribution) = (rng, ind) -> float(rand(rng, d))
_trait_sampler(f::Function) = (rng, ind) -> float(f(rng, ind))

"""
    compose(fs...)

Compose multiple attributes functions into one, called in order on each
individual at creation time.

Each `f` must be a callable `(rng, ind) -> nothing` that mutates
`ind.state` and/or fields on `ind` (e.g. `ind.susceptibility`).
EpiBranch provides [`clinical_presentation`](@ref), [`demographics`](@ref),
and [`transmission_traits`](@ref) for the common patterns; for other
fields, pass a plain closure.

# Examples

Combine the standard builders:

```julia
attributes = compose(
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
    transmission_traits(susceptibility = 0.3),
)
```

Correlate susceptibility with age — `transmission_traits` accepts a
function, so the rule is part of the builder rather than a follow-up
closure:

```julia
attributes = compose(
    demographics(age_distribution = Uniform(0, 90)),
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:age] >= 65 ? 0.8 : 0.3,
    ),
)
```

For fields without a dedicated builder, the inline closure form is the
escape hatch:

```julia
attributes = compose(
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    (rng, ind) -> (ind.state[:risk_group] = rand(rng, [:low, :high])),
)
```

Pass to [`simulate`](@ref) or [`simulate_batch`](@ref) via the
`attributes` keyword.
"""
compose(fs...) = (rng, ind) -> for f in fs
    ;
    f(rng, ind);
end

# ── Intervention field validation ────────────────────────────────────

"""Fields that an intervention requires on individuals. Default: none."""
required_fields(::AbstractIntervention) = Symbol[]

"""Check that all required fields are present on an individual. Works for
any iterable of items that define `required_fields` — used for both
interventions and clinical transitions."""
function _validate_required_fields(individual, items)
    for item in items
        for field in required_fields(item)
            if !haskey(individual.state, field)
                itype = typeof(item)
                hint = _field_hint(field)
                error("$itype requires field :$field on individuals. $hint")
            end
        end
    end
end

function _field_hint(field::Symbol)
    hints = Dict(
        :onset_time => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :asymptomatic => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :age => "Provide attributes = demographics(age_distribution = ...).",
        :sex => "Provide attributes = demographics(...)."
    )
    return get(hints, field, "Set this field via an attributes function.")
end
