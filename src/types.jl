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

"""
    ChainSizeLaw

Parent type of every chain-size distribution: [`Borel`](@ref EpiBranch.Borel),
[`GammaBorel`](@ref EpiBranch.GammaBorel), [`PoissonGammaChainSize`](@ref
EpiBranch.PoissonGammaChainSize), [`ChainSizeMixture`](@ref),
[`IndexChainSize`](@ref), [`TruncatedChainSize`](@ref) and
[`ThinnedChainSize`](@ref). One `loglikelihood(::ChainSizes, ::ChainSizeLaw)`
method covers all of them, so `observe(law, obs)` can be passed straight to
`loglikelihood` for any observation model, built in or user-defined, as long
as the law it returns subtypes `ChainSizeLaw` and has a `logpdf` method.
"""
abstract type ChainSizeLaw <: DiscreteUnivariateDistribution end

# Abstract supertype for clinical-state transitions (defined here so
# `SimulationState` can hold a typed `transitions` vector without
# forward-reference issues). Implementations live in
# `src/transitions/`.
abstract type AbstractClinicalTransition end

# Abstract supertype for a cut race's unfinished event queue (defined here,
# for the same forward-reference reason as `AbstractClinicalTransition`; the
# concrete type is `_RaceContinuation` in `sellke.jl`). `simulate`'s
# continuation method resumes where the race it describes left off, rather
# than redrawing contacts already decided.
abstract type AbstractRaceContinuation end

# ── Transmission models ─────────────────────────────────────────────

"""
The parent type of all transmission models: [`BranchingProcess`](@ref),
[`HomogeneousProcess`](@ref), [`MixingProcess`](@ref), `NetworkProcess`,
`RoutedNetwork` and `HouseholdProcess`. Needed only to write
a new kind of model; the Extending EpiBranch guide lists what a new model
defines.
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

The closed-form results read the offspring distribution through this
function, so a new transmission model that defines it can use all of them:
[`reproduction_number`](@ref), [`extinction_probability`](@ref),
[`epidemic_probability`](@ref), [`probability_contain`](@ref),
[`proportion_transmission`](@ref), [`proportion_cases_individual`](@ref),
[`proportion_cases_offspring`](@ref), [`proportion_cluster_size`](@ref) and
[`chain_size_distribution`](@ref).
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

"""
    has_single_type_law(model::TransmissionModel)

Whether `single_type_offspring(model)` returns an offspring specification
with a closed-form `chain_size_distribution` method, rather than
`single_type_offspring` throwing or the offspring having no such method.
`chain_size_distribution` and `loglikelihood(::ChainSizes, model)` check
this before taking the analytical fast path, rather than each catching the
errors `single_type_offspring` and `chain_size_distribution` throw for a
multi-type/multi-window model or an offspring with no closed form.

The default calls `single_type_offspring` and conservatively reports `false`
(fall back to simulation) for anything it throws an `ArgumentError` or
`MethodError` over, and likewise checks whether `chain_size_distribution`
has a method for the result, so a custom `TransmissionModel` need not
specialise this to get correct, if not always fastest, behaviour.

This `hasmethod` check is on the offspring's type, not the particular
instance: a [`GeneratingFunctionOffspring`](@ref) has the method whether or
not it has a pmf, so calling `chain_size_distribution` on the result can
still throw for one built from a bare generating function. The analytical
fast paths check for a missing pmf before constructing the law.
"""
function has_single_type_law(model::TransmissionModel)
    try
        off = single_type_offspring(model)
        return hasmethod(chain_size_distribution, Tuple{typeof(off)})
    catch e
        (e isa ArgumentError || e isa MethodError) && return false
        rethrow()
    end
end

"""
    offspring_law_unchanged(x) -> Bool

Whether `x` leaves a model's single-type offspring law exactly as the
offspring distribution draws it, with nothing reshaping how many of the
drawn contacts end up as cases. `loglikelihood(::ChainSizes, model)`,
[`chain_size_distribution`](@ref) and the closed-form incidence likelihood
take their analytical path, scoring the bare offspring distribution, only
once the process, every entry of `attributes` and every `progression`
transition each answer `true` here; any one of them answering `false` sends
the evaluation to simulation instead.

For an attribute or a progression transition, the default is `true`: most of
what either can hold — the natural history, population characteristics such
as age or a group label, reporting — has no bearing on the offspring law at
all. [`transmission_traits`](@ref) overrides it to `false` once
susceptibility or infectiousness departs from its neutral value of 1, since
either one thins how many of the drawn contacts actually become cases. A
`Tuple` or `AbstractVector` of attributes answers `true` only once every
entry does.

A bare `Function` or closure answers the conservative `false` instead,
whatever it does: an `attributes` entry can set `ind.susceptibility` or
`ind.infectiousness` directly rather than going through
[`transmission_traits`](@ref) (its docstring recommends exactly this for
rules it doesn't cover), and this trait has no way to tell that apart from
one that leaves them alone. [`demographics`](@ref) and [`groups`](@ref)
return a dedicated callable type rather than a closure so they keep the
`true` default; give a custom attributes function one too if it should.

A custom [`AbstractClinicalTransition`](@ref) answers the same conservative
`false`, for the same reason: `initialise_individual!` and
`resolve_individual!` receive the case directly and can set its
susceptibility or infectiousness just as a bare attributes function can. The
five built-in transitions ([`Transition`](@ref), [`Reporting`](@ref),
[`Hospitalisation`](@ref), [`Recovery`](@ref), [`Death`](@ref)) touch neither
field; each overrides it back to `true`, and one of your own that doesn't
touch them either should do the same.

For a [`TransmissionModel`](@ref) the default is the conservative `false`
instead, as for [`EpiBranch.infection_likelihood_compatible`](@ref): a new
model opts in once it has checked that nothing of its own — a finite
population, a window that can close early, or anything else — reshapes the
offspring law away from the bare distribution. [`BranchingProcess`](@ref)
opts in except for a finite `population_size` (fewer susceptible people than
an unlimited offspring law assumes) or an infectiousness window with a
non-empty `until` (transmission can stop before every drawn contact
happens).
"""
offspring_law_unchanged(::Any) = true
offspring_law_unchanged(::Function) = false
offspring_law_unchanged(::TransmissionModel) = false
offspring_law_unchanged(::AbstractClinicalTransition) = false
function offspring_law_unchanged(xs::Union{Tuple, AbstractVector})
    return all(offspring_law_unchanged, xs)
end

"""
    reporting_probability(transition) -> Union{Nothing, Real, Function}

The probability that `transition` reports a case, or `nothing` (the default)
when it is not a reporting step. The closed forms that thin a chain-size law
or an incidence series by under-reporting ask this of every `progression`
transition rather than matching [`Reporting`](@ref) by type. A reporting
step of your own (one with a probability that varies with age, or a testing
transition) reaches them by defining it. [`Reporting`](@ref) returns its
`probability` field.

A closed form needs a number and falls back to simulation for a function.
The time the report is measured from is read with
[`EpiBranch.reporting_from`](@ref), and a step must also record `:reported`
and `:reporting_time` as [`Reporting`](@ref) does for `simulate` to agree with
the closed form.
"""
reporting_probability(::Any) = nothing

"""
    reporting_from(transition)

The time a reporting step measures its delay from, in the form
taken by the `from` field of [`Reporting`](@ref): `:infection`, a recorded time such as
`:onset_time`, or a function of the individual. Only asked of a transition
that answers [`EpiBranch.reporting_probability`](@ref). The default is
`:onset_time`.
"""
reporting_from(::Any) = :onset_time

"""
    onset_law(attribute)

The law of a case's symptom onset that `attribute` sets, as a
`NamedTuple` `(incubation_period, prob_asymptomatic)`: the incubation-period
distribution and the share of cases that never develop symptoms. The default
is `nothing`, for an attribute that records no onset.

The closed forms that need the onset (the analytic effect of
[`Isolation`](@ref), and reporting timed from onset in the incidence
likelihood and the chain-size law) ask each of the model's attributes this
rather than matching the type of [`clinical_presentation`](@ref). An onset
builder of your own reaches them by defining it. A model must have exactly
one attribute that answers, and its `prob_asymptomatic` must be a number for
the closed forms to use it. `simulate` must set `:onset_time` and
`:asymptomatic` consistently with the law you declare.
"""
onset_law(::Any) = nothing

"""The number of types of case `model` has, or `1` for a single-type model
(the default). A multi-type model, such as a [`BranchingProcess`](@ref)
built from an offspring matrix, overrides this with its own number of
types; [`single_type_offspring`](@ref) raises an error for one."""
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
    state::IndividualState{T}
end

"""
One person in a simulation: for a [`BranchingProcess`](@ref), a case or a
contact they exposed; for models with a fixed population, anyone in it.

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
- `secondary_case_ids`: the `id`s of the people they exposed. A branching
  process lists the contacts it created, infected or not. One whose exposure
  did not happen within the run has a `NaN` infection time: it falls after the
  run stopped, or a ring created it for a member who was never infected. Most contacts a run stopped before reaching
  were never created and are not listed. A model written against `generate_offspring` lists every
  contact it created; one written against `contacts_of` lists an existing
  person only if it infected them. The fixed-population models below list
  only the people they infected.
- `state`: everything else recorded about them, such as symptom onset
  (`:onset_time`), `:asymptomatic`, `:age`, `:sex`, whether they were
  `:isolated`, `:traced`, `:quarantined` or `:vaccinated`, `:test_positive`,
  their `:type` in a multi-type model, whether they were `:infected`, and any
  population characteristics you add. It is read and written as a dictionary
  (see [`EpiBranch.IndividualState`](@ref)).
- `episodes`: past infections, oldest first, for a person who has been
  infected more than once (see [`close_episode!`](@ref)). Add one with
  `close_episode!`, because a `push!` onto `ind.episodes` is lost. The fields above
  describe the current or latest infection. None of the built-in models
  reinfects anyone, so this is empty there.

In the fixed-population models ([`HomogeneousProcess`](@ref),
[`MixingProcess`](@ref), `NetworkProcess`, `RoutedNetwork`,
`HouseholdProcess`),
`susceptibility` and `infectiousness` instead multiply the rate at which a
pair meets. They then lower the chance of infection only over an
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
    state::IndividualState{T}
    # Allocated by the first `close_episode!`: most runs never reinfect anyone,
    # and an empty vector per person is a heap object each.
    _episodes::Union{Nothing, Vector{InfectionEpisode{T}}}
end

# `ind.episodes` reads the archive through `_episodes`. A person with none gets
# a fresh empty vector, which keeps a `push!` onto `ind.episodes` from reaching
# anyone else but also loses it. Add an episode with `close_episode!`.
@inline function Base.getproperty(ind::Individual{T}, name::Symbol) where {T}
    name === :episodes || return getfield(ind, name)
    episodes = getfield(ind, :_episodes)
    return episodes === nothing ? InfectionEpisode{T}[] : episodes
end

@inline function Base.setproperty!(ind::Individual{T}, name::Symbol, value) where {T}
    if name === :episodes
        return setfield!(
            ind, :_episodes,
            isempty(value) ? nothing : convert(Vector{InfectionEpisode{T}}, value)
        )
    end
    return setfield!(ind, name, value)
end

Base.propertynames(::Individual, private::Bool = false) =
    (filter(!=(:_episodes), fieldnames(Individual))..., :episodes)

# Whether `ind` has archived an episode, without building the empty vector
# `ind.episodes` returns for someone who has none.
_has_episodes(ind::Individual) = getfield(ind, :_episodes) !== nothing
_n_episodes(ind::Individual) =
    (e = getfield(ind, :_episodes); e === nothing ? 0 : length(e))

function Individual(;
        id::Int, parent_id::Int = 0, generation::Int = 0,
        chain_id::Int = 1, infection_time::Real = 0.0,
        susceptibility::Real = 1.0, infectiousness::Real = 1.0,
        state::AbstractDict = IndividualState{Float64}()
    )
    T = promote_type(
        typeof(infection_time), typeof(susceptibility),
        typeof(infectiousness)
    )
    return Individual{T}(
        id, parent_id, generation, chain_id, convert(T, infection_time),
        convert(T, susceptibility), convert(T, infectiousness), Int[],
        IndividualState{T}(state), nothing
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
    copy(ind.secondary_case_ids), _detach_removal_history!(copy(ind.state))
)

"""
    close_episode!(ind::Individual, episode::InfectionEpisode)

Keep a past infection of `ind` in `ind.episodes` when they are reinfected.
`episode` is usually recorded with [`InfectionEpisode`](@ref) before the new
infection replaces `ind`'s current fields. `ind.secondary_case_ids` is
emptied, so the new infection counts only its own secondary cases.
"""
function close_episode!(ind::Individual{T}, episode::InfectionEpisode) where {T}
    episodes = getfield(ind, :_episodes)
    if episodes === nothing
        episodes = InfectionEpisode{T}[]
        setfield!(ind, :_episodes, episodes)
    end
    push!(episodes, episode)
    empty!(ind.secondary_case_ids)
    return ind
end

# The three things a lazily populated model (`NetworkProcess`, `RoutedNetwork`)
# needs to create one of its own not-yet-existing nodes: the placeholder every
# unrealised slot in `individuals` points to until then, shared by every slot
# in one state and never mutated, so checking for it is one pointer
# comparison; the interventions the node is created with; and anything the
# model's own `add_individuals!`-style `setup` would have recorded about it
# (a graph index, say). `nothing` on `SimulationState` for a model that
# builds its whole population upfront, which is every model but these two.
struct _LazyPopulation{T <: Real, F}
    placeholder::Individual{T}
    interventions::Any
    setup::F
end

# ── People shared between copies of a state ────────────────────────

# Replaces the plain vector in `state.individuals` when the state is copied.
# `people` holds the `Individual` objects. `owned[i]` records whether this
# state alone holds `people[i]`. Julia cannot distinguish reads from writes
# on a mutable struct. The first access to a shared person therefore replaces
# them with a private copy, used for all subsequent accesses. People who are
# not accessed remain shared between states. In a large population, this is
# most people over a short continuation.
struct _SharedPeople{T <: Real} <: AbstractVector{Individual{T}}
    people::Vector{Individual{T}}
    owned::BitVector
end

Base.size(v::_SharedPeople) = size(v.people)
Base.IndexStyle(::Type{<:_SharedPeople}) = IndexLinear()

@inline function Base.getindex(v::_SharedPeople, i::Int)
    @boundscheck checkbounds(v.people, i)
    @inbounds if !v.owned[i]
        v.people[i] = deepcopy(v.people[i])
        v.owned[i] = true
    end
    return @inbounds v.people[i]
end

@inline function Base.setindex!(v::_SharedPeople{T}, ind, i::Int) where {T}
    @boundscheck checkbounds(v.people, i)
    @inbounds v.people[i] = ind
    @inbounds v.owned[i] = true
    return v
end

function Base.push!(v::_SharedPeople, ind)
    push!(v.people, ind)
    push!(v.owned, true)
    return v
end

function Base.resize!(v::_SharedPeople, n::Integer)
    old = length(v.people)
    resize!(v.people, n)
    resize!(v.owned, n)
    n > old && (v.owned[(old + 1):n] .= true)
    return v
end

# ── Simulation state ───────────────────────────────────────────────

"""
One simulated outbreak, as returned by [`simulate`](@ref). Turn it into
tables with [`linelist`](@ref) (one row per case), [`contacts`](@ref) (one
row per exposure), [`chain_statistics`](@ref) or [`weekly_incidence`](@ref).

Useful fields:

- `individuals`: one [`Individual`](@ref) per person in the simulation: the
  cases and the contacts they exposed for a [`BranchingProcess`](@ref),
  everyone for models with a fixed population. `NetworkProcess` and
  `RoutedNetwork` create a person here only once the outbreak reaches them
  (see [`realise!`](@ref)); a person it never reached reads as never
  infected and untouched by any intervention.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out when the run stopped.
- `current_generation`: the last generation simulated.

For extension authors: `transitions` holds the natural-history steps of the
model, and `scratch` is space in which an intervention can keep its own
working information for the run (as `Individual.state` is for each person),
named as the Extending EpiBranch guide describes. The simulation never reads
`scratch`. `GroupVaccination` keeps its list of each group's members there.
`continuation` is the unfinished race [`simulate`](@ref)'s continuation
method resumes, `nothing` once the run has reached extinction or stopped for
some other reason; nothing outside the simulation engine reads or writes it.
`lazy` is set by [`EpiBranch.lazily_populate!`](@ref) for a model with
a population created as the race reaches it, `nothing` for one built
upfront; nothing outside the simulation engine reads or writes it either.
"""
mutable struct SimulationState{T <: Real, R <: AbstractRNG, P, A}
    individuals::Union{Vector{Individual{T}}, _SharedPeople{T}}
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
    continuation::Union{Nothing, AbstractRaceContinuation}
    lazy::Union{Nothing, _LazyPopulation}
end

# Pre-existing callers construct a `SimulationState` without `scratch`,
# `continuation` or `lazy`; all three start absent, so this fills them in
# rather than requiring every call site to name them.
function SimulationState(
        individuals, active_ids, current_generation, rng, cumulative_cases,
        extinct, population_size, max_infection_time, attributes, transitions
    )
    return SimulationState(
        individuals, active_ids, current_generation, rng, cumulative_cases,
        extinct, population_size, max_infection_time, attributes, transitions,
        Dict{Any, Any}(), nothing, nothing
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

# Mark every person in `people` as held by two states, except the placeholder
# of a lazily populated model. No state mutates the placeholder. It must retain
# its identity for `_realised!` to recognise it.
function _share!(people::Vector{Individual{T}}, placeholder) where {T}
    return _share!(_SharedPeople{T}(people, falses(length(people))), placeholder)
end
function _share!(shared::_SharedPeople, placeholder)
    for i in eachindex(shared.people)
        shared.owned[i] = shared.people[i] === placeholder
    end
    return shared
end

"""
    copy(state::SimulationState) -> SimulationState

An independent copy of `state`, cheaper than `deepcopy` for a large
population. The copy and the original share each person until accessed
through either state, when that state takes a private copy of the person.
People who are not accessed remain shared. The cost of copying people during
continuation is proportional to the number accessed, rather than the total
population. The random number generator, the pending events of an unfinished
race, the working information in `scratch` and the list of infectious cases
are copied in full. The natural-history `transitions`, the `attributes`
builder and the set-up of the lazily populated model are shared because they
remain unchanged during a run.

Continuing a run from the copy gives the same result as continuing from a
`deepcopy` of the same state. Changes to either state leave the other
unchanged, provided people are accessed through the state. A reference to an
`Individual` obtained from `state.individuals` before copying still refers
to the person in the original state. While that person remains shared,
changes through the reference affect both states. Access people through
`state.individuals` again after copying.
"""
function Base.copy(state::SimulationState)
    lazy = state.lazy
    placeholder = lazy === nothing ? nothing : lazy.placeholder
    original = state.individuals
    shared = _share!(original, placeholder)
    state.individuals = shared
    people = _SharedPeople(copy(shared.people), copy(shared.owned))
    return SimulationState(
        people, copy(state.active_ids), state.current_generation,
        deepcopy(state.rng), state.cumulative_cases, state.extinct,
        state.population_size, state.max_infection_time, state.attributes,
        copy(state.transitions), deepcopy(state.scratch),
        deepcopy(state.continuation), lazy
    )
end
