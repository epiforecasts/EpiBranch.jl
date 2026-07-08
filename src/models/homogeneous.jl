# ── HomogeneousProcess ───────────────────────────────────────────────
#
# A homogeneously-mixing closed population of fixed size N: every infectious
# individual exerts the same force of infection on every susceptible, so the
# outbreak is a finite, depleting pool with no structure beyond its size. It is
# simulated by the Sellke threshold construction (`_sellke_pool!`), which
# reproduces the exact stochastic SIR final-size law and yields infection times,
# not just the final size. Like the other structure-driven models it shares the
# natural-history timeline (`progression`/`Transition`), interventions,
# attributes and observation, and returns an EpiBranch `SimulationState` that
# `linelist` renders.

"""
    HomogeneousProcess(; transmission_rate = nothing, R0 = nothing, population_size,
                       infectious_period = nothing, latent_period = nothing,
                       progression = [], from = nothing,
                       until = (:recovered, :died, :isolated),
                       interventions = [], attributes = NoAttributes(),
                       observation = NoObservation())

A closed population of `population_size` individuals that mix homogeneously:
every infectious person exerts the same force of infection on every susceptible.
It is simulated by the Sellke threshold construction, which reproduces the exact
stochastic SIR final-size law. Set transmission either as `transmission_rate`
(the per-infective rate β, so β/N to each susceptible) or as `R0`, from which β
follows as `R0 / mean(infectious_period)`. Give one of the two; `R0` needs an
`infectious_period`.

The infectious timeline is a flexible `progression` of EpiBranch `Transition`s,
exactly as for `BranchingProcess`: `infectious_period` and `latent_period` are
sugar for the two common transitions (a latent period to `:infectious`, an
infectious period to a terminal `:recovered` removal), each accepting a scalar
(constant) or a `Distribution` (per-case). `from` is the state the infectious
window opens at (`:infectious` when the timeline has a latent period,
`:infection` otherwise); `until` names the removal states that close it.

How detailed the course of infection is up to you. With just an
`infectious_period` this is an SIR model; a `latent_period` makes it SEIR, and
onset, hospitalisation and death (as an alternative to recovery) come from the
`progression` and appear in the line list. The model holds to two assumptions: a
closed population, and one infection per person. Age or contact structure,
waning immunity and reinfection are things you write into a model by extending
it.

# Example

```julia
using EpiBranch, Distributions
model = HomogeneousProcess(; R0 = 2.0, population_size = 3000,
    infectious_period = Exponential(1.0))
state = simulate(model; n_initial = 5)
```
"""
struct HomogeneousProcess{A, O <: ObservationModel} <: TransmissionModel
    population_size::Int
    β::Float64
    from::Symbol
    until::Tuple
    progression::Vector{AbstractClinicalTransition}
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O
end

function HomogeneousProcess(; transmission_rate = nothing, R0 = nothing,
        population_size::Integer,
        infectious_period = nothing, latent_period = nothing,
        progression = AbstractClinicalTransition[],
        from = nothing,
        until = (:recovered, :died, :isolated),
        interventions = AbstractIntervention[],
        attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    (transmission_rate === nothing) == (R0 === nothing) && throw(ArgumentError(
        "provide exactly one of `transmission_rate` (β directly) or `R0`"))

    prog = convert(Vector{AbstractClinicalTransition}, progression)
    # Desugar the timeline shorthands when no explicit progression is given:
    # a latent period to :infectious, an infectious period to a recovery removal.
    if isempty(prog)
        latent_period === nothing ||
            push!(prog,
                Transition(:infectious; from = :infection,
                    delay = _homogeneous_delay(latent_period)))
        anchor = latent_period === nothing ? :infection : :infectious
        infectious_period === nothing ||
            push!(prog,
                Transition(:recovered; from = anchor,
                    delay = _homogeneous_delay(infectious_period), terminal = true))
    end

    # The infectious window opens at :infectious when a latent period produces
    # it, otherwise at infection.
    kfrom = from === nothing ?
            (any(t -> _homogeneous_writes(t, :infectious), prog) ? :infectious :
             :infection) : from

    # Resolve β from R0 via the mean infectious period, or take it directly.
    if R0 !== nothing
        infectious_period === nothing && throw(ArgumentError(
            "`R0` requires `infectious_period` to derive β = R0 / mean infectious period"))
        β = Float64(R0) / _homogeneous_mean(infectious_period)
    else
        β = Float64(transmission_rate)
    end

    return HomogeneousProcess(Int(population_size), β, kfrom, Tuple(until), prog,
        _intervention_vector(interventions), attributes, observation)
end

# A `Transition` delay is a `Distribution` or `(rng, ind) -> Real`; a scalar
# period becomes a constant-delay closure so the sugar accepts both.
_homogeneous_delay(d::Distribution) = d
_homogeneous_delay(x::Real) = (rng, ind) -> float(x)

# Mean of an infectious-period specification: a distribution's mean, or the
# scalar itself.
_homogeneous_mean(d::Distribution) = mean(d)
_homogeneous_mean(x::Real) = float(x)

# Does a clinical transition write the named state?
_homogeneous_writes(t, s::Symbol) = hasproperty(t, :state) && getfield(t, :state) === s

# The interventions/attributes/observation the model carries.
population_size(m::HomogeneousProcess) = m.population_size
interventions(m::HomogeneousProcess) = m.interventions
attributes(m::HomogeneousProcess) = m.attributes
observation(m::HomogeneousProcess) = m.observation

function Base.show(io::IO, m::HomogeneousProcess)
    print(io, "HomogeneousProcess(population_size=$(m.population_size), ",
        "β=$(round(m.β; digits = 4)), from=:$(m.from), ",
        "$(length(m.progression)) transitions)")
end

"""
    simulate(model::HomogeneousProcess; rng = default_rng(), n_initial = 1) -> SimulationState

Simulate `model` by the Sellke threshold construction over its fixed pool,
seeding `n_initial` index cases at time 0. Returns an EpiBranch
`SimulationState`; `linelist(state)` turns it into the one-row-per-case
DataFrame, carrying the infection, infectiousness, onset and recovery times the
model's `progression` stamps on each case.
"""
function simulate(model::HomogeneousProcess; rng::AbstractRNG = Random.default_rng(),
        n_initial::Integer = 1)
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))
    n_initial <= model.population_size ||
        throw(ArgumentError("n_initial cannot exceed population_size"))

    state = new_state(model, model.progression, attributes(model), rng)
    add_individuals!(state, model.population_size, interventions(model);
        setup = (ind, i) -> nothing)

    # The homogeneous pool is the one-type case of the structured Sellke pool:
    # no attributes name the mixing (`mixing_by` defaults to `()`), so every
    # individual has the empty type `()` and feels the same force β/N per
    # infective (`sum(values(counts))` = number currently infectious).
    _sellke_pool!(state, collect(1:model.population_size), rng;
        force = (type, counts) -> model.β / model.population_size * sum(values(counts)),
        n_initial = n_initial, from = model.from, until = model.until)

    # The pool loop writes per-individual state directly, so reconcile the
    # aggregate bookkeeping the engine would otherwise maintain.
    state.cumulative_cases = count(
        ind -> get(ind.state, :infected, false), state.individuals)
    state.max_infection_time = maximum(
        (ind.infection_time
        for ind in state.individuals if get(ind.state, :infected, false));
        init = 0.0)

    apply_observation!(observation(model), state, rng)
    return state
end
