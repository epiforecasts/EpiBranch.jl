# ── Sentinel types ──────────────────────────────────────────────────
# These replace Union{T, Nothing} patterns throughout the codebase,
# enabling dispatch instead of runtime nothing-checks.

"""Placeholder meaning no limit on the population size (an unbounded
population)."""
struct NoPopulation end

"""Placeholder meaning no population characteristics are drawn."""
struct NoAttributes end

"""Placeholder meaning the types of a multi-type model have no names."""
struct NoTypeLabels end

"""Placeholder meaning no age distribution is given (ages are uniform over
the age range)."""
struct NoAgeDistribution end

"""Placeholder meaning no case cap when deciding whether an outbreak was
contained."""
struct NoCases end

"""Placeholder meaning no generation time: a [`BranchingProcess`](@ref)
without timing, used for chain sizes and lengths only."""
struct NoGenerationTime end

# Abstract supertype for clinical-state transitions (defined here so
# `SimulationState` can hold a typed `transitions` vector without
# forward-reference issues). Implementations live in
# `src/transitions/`.
abstract type AbstractClinicalTransition end

# ── Transmission models ─────────────────────────────────────────────

"""
The parent type of all transmission models: [`BranchingProcess`](@ref),
[`HomogeneousProcess`](@ref), `NetworkProcess`,
`RoutedNetwork` and `HouseholdProcess`. Needed only to write
a new kind of model; the [Extending guide](@ref "Extending EpiBranch") lists
what a new model defines.
"""
abstract type TransmissionModel end

"""The number of people who can be infected in `model`, or
`NoPopulation()` if unlimited (the default)."""
population_size(::TransmissionModel) = NoPopulation()

"""
    single_type_offspring(model::TransmissionModel)

The offspring distribution of a single-type model: usually a distribution
such as `NegBin(R, k)`, or anything else [`chain_size_distribution`](@ref)
accepts, such as [`ClusterMixed`](@ref). Raises an error for a multi-type
model.

The closed-form results ([`extinction_probability`](@ref),
[`epidemic_probability`](@ref), [`probability_contain`](@ref),
[`proportion_transmission`](@ref), [`chain_size_distribution`](@ref)) read
the offspring distribution through this function, so a new transmission
model that defines it can use all of them.
"""
function single_type_offspring(model::TransmissionModel)
    hasproperty(model, :offspring) || throw(
        ArgumentError(
            "$(typeof(model)) has no `offspring` field — did you forget to specialise single_type_offspring for it?"
        )
    )
    off = model.offspring
    off isa Function && throw(
        ArgumentError(
            "This function only works with single-type models (not multi-type function offspring)"
        )
    )
    return off
end
n_types(::TransmissionModel) = 1

# The offspring specification the analytical helpers dispatch on. By default it
# comes from `single_type_offspring`, which a custom model defines;
# `BranchingProcess` returns its `MultiTypeOffspring` here when it has one.
_analytic_offspring(model::TransmissionModel) = single_type_offspring(model)

# ── Individual state ────────────────────────────────────────────────

"""
A past infection of a person who has since been reinfected, kept on the
[`Individual`](@ref). It records that infection's `infection_time`, its place
in the transmission tree (`parent_id`, `generation`, `chain_id`), the
secondary cases it caused, and the person's `state` as it was when that
infection ended (natural-history times, outcome, and what interventions
recorded).

See also [`susceptible_again_time`](@ref), the time from which a person can
be infected again, and [`close_episode!`](@ref).
"""
struct InfectionEpisode{T <: Real}
    infection_time::T
    parent_id::Int
    generation::Int
    chain_id::Int
    secondary_case_ids::Vector{Int}
    state::Dict{Symbol, Any}
end

"""
One person in a simulated outbreak: a case, or a contact who was exposed
but not infected.

- `id`: the person's number, which is also their position in
  `state.individuals`.
- `parent_id`: the `id` of their infector (0 for an index case).
- `generation`: 0 for index cases, 1 for the people they infect, and so on.
- `chain_id`: which index case's transmission chain they belong to.
- `infection_time`: when they were exposed, in days since the start of the
  outbreak.
- `susceptibility`, between 0 and 1: the probability that they are infected
  when exposed.
- `infectiousness`, between 0 and 1: the factor applied to their onward
  transmission once infected (each of their contacts is infected only with
  this probability).
- `secondary_case_ids`: the `id`s of the contacts they exposed, infected or
  not.
- `state`: everything else recorded about them, such as symptom onset
  (`:onset_time`), `:asymptomatic`, `:age`, `:sex`, whether they were
  `:isolated`, `:traced`, `:quarantined` or `:vaccinated`, `:test_positive`,
  their `:type` in a multi-type model, whether they were `:infected`, and any
  population characteristics you add.
- `episodes`: past infections, oldest first, for a person who has been
  infected more than once (see [`close_episode!`](@ref)). The fields above
  describe the current or latest infection. None of the built-in models
  reinfects anyone, so this is empty there.

In the continuous-time models ([`HomogeneousProcess`](@ref),
`NetworkProcess`, `RoutedNetwork`, `HouseholdProcess`),
`susceptibility` and `infectiousness` instead multiply the rate of
transmission. They then lower the chance of infection only over an
infectious period of limited length: if the infectious period never ends,
any positive value leads to infection eventually, only later on average, and
a contact certain to fall within the infectious period (a fixed delay such as
`Dirac(2.0)`) infects whatever the value.

# Setting fields at simulation time

Give `attributes` to a [`ModelSpec`](@ref); they are drawn for every new
person when they are created. Several can be listed, and are applied in
order:

```julia
attributes = [
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
    transmission_traits(susceptibility = 0.3, infectiousness = 0.9),
]
```

For anything else, add your own function of the random number generator and
the individual, `(rng, ind) -> ...`, to the list.

See also [`clinical_presentation`](@ref),
[`demographics`](@ref), [`transmission_traits`](@ref).
"""
mutable struct Individual{T <: Real}
    id::Int
    parent_id::Int
    generation::Int
    chain_id::Int
    infection_time::T
    susceptibility::T
    infectiousness::T
    secondary_case_ids::Vector{Int}
    state::Dict{Symbol, Any}
    episodes::Vector{InfectionEpisode{T}}
end

function Individual(;
        id::Int, parent_id::Int = 0, generation::Int = 0,
        chain_id::Int = 1, infection_time::Real = 0.0,
        susceptibility::Real = 1.0, infectiousness::Real = 1.0,
        state::Dict{Symbol, Any} = Dict{Symbol, Any}()
    )
    T = promote_type(
        typeof(infection_time), typeof(susceptibility),
        typeof(infectiousness)
    )
    return Individual{T}(
        id, parent_id, generation, chain_id, convert(T, infection_time),
        convert(T, susceptibility), convert(T, infectiousness), Int[], state,
        InfectionEpisode{T}[]
    )
end

"""
    InfectionEpisode(ind::Individual)

Record `ind`'s current infection (its infection time, place in the
transmission tree, secondary cases so far and a copy of `state`) as an
[`InfectionEpisode`](@ref), so it can be kept with [`close_episode!`](@ref)
before a reinfection replaces it.
"""
InfectionEpisode(ind::Individual{T}) where {T} = InfectionEpisode{T}(
    ind.infection_time, ind.parent_id, ind.generation, ind.chain_id,
    copy(ind.secondary_case_ids), copy(ind.state)
)

"""
    close_episode!(ind::Individual, episode::InfectionEpisode)

Keep a past infection of `ind` in `ind.episodes` when they are reinfected.
`episode` is usually recorded with [`InfectionEpisode`](@ref) before the new
infection replaces `ind`'s current fields. `ind.secondary_case_ids` is
emptied, so the new infection counts only its own secondary cases.
"""
function close_episode!(ind::Individual, episode::InfectionEpisode)
    push!(ind.episodes, episode)
    empty!(ind.secondary_case_ids)
    return ind
end

# ── Simulation state ───────────────────────────────────────────────

"""
One simulated outbreak, as returned by [`simulate`](@ref). Turn it into
tables with [`linelist`](@ref) (one row per case), [`contacts`](@ref) (one
row per exposure), [`chain_statistics`](@ref) or [`weekly_incidence`](@ref).

Useful fields:

- `individuals`: everyone in the outbreak, cases and uninfected contacts, as
  [`Individual`](@ref)s.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out when the run stopped.
- `current_generation`: the last generation simulated.

For extension authors: `transitions` holds the natural-history steps of the
model, and `scratch` is space in which an intervention can keep its own
working information for the run (as `Individual.state` is for each person),
named as the extending guide describes. The simulation never reads
`scratch`. `GroupVaccination` keeps its list of each group's members there.
"""
mutable struct SimulationState{T <: Real, R <: AbstractRNG, P, A}
    individuals::Vector{Individual{T}}
    active_ids::Vector{Int}
    current_generation::Int
    rng::R
    cumulative_cases::Int
    extinct::Bool
    population_size::P
    max_infection_time::T
    attributes::A
    transitions::Vector{AbstractClinicalTransition}
    scratch::Dict{Any, Any}
end

# Pre-existing callers construct a `SimulationState` without `scratch`; it
# always starts empty, so this fills it in rather than requiring every call
# site to name it.
function SimulationState(
        individuals, active_ids, current_generation, rng, cumulative_cases,
        extinct, population_size, max_infection_time, attributes, transitions
    )
    return SimulationState(
        individuals, active_ids, current_generation, rng, cumulative_cases,
        extinct, population_size, max_infection_time, attributes, transitions,
        Dict{Any, Any}()
    )
end

"""The number type used for times and rates in `state` (`Float64` by
default)."""
_timetype(::SimulationState{T}) where {T} = T

function Base.show(io::IO, s::SimulationState)
    status = s.extinct ? "extinct" : "active"
    return print(
        io,
        "SimulationState(cases=$(s.cumulative_cases), individuals=$(length(s.individuals)), gen=$(s.current_generation), $(status))"
    )
end
