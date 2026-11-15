# ── Population ───────────────────────────────────────────────────────
#
# A model takes its population once. The engines dispatch on the kind of
# population they are given: none, one group or several, rather than on a
# keyword that is sometimes an `Int` and sometimes a vector.

"""
    Population(size)
    Population(sizes; groups = NoTypeLabels())

The people a model is run over, given once: `Population(5000)` is one group of
5000 people, and `Population([1500, 3500]; groups = ["child", "adult"])` is
one group per entry of `sizes`. Every model takes it as its `population`
keyword, and `population_size = 5000` stays as shorthand for `Population(5000)`.
Without a population, a [`BranchingProcess`](@ref) has no limit on the number
of people it can infect.

A group is depleted against its own size: each person is infected once, and a
case of one group leaves the others' susceptibles untouched. `groups` names
the groups, in the order of `sizes`:

- for a [`BranchingProcess`](@ref), labels for the types, as `type_labels`;
- for a [`MixingProcess`](@ref), the value of the `mixing_by` attributes of
  each group, one tuple per group (a bare value is taken as a one-element
  tuple). The process then writes that attribute itself, so that the groups
  have exactly these sizes, and an attribute builder that also writes it is an
  error. Leave the sizes out of the population (`Population(n)`) for groups
  with sizes that vary between runs, such as an attribute drawn with
  `rand(rng, 1:2)`.

`Population(n)` and `Population(sizes)` return a `SinglePopulation` and a
`GroupedPopulation`, the types the engines dispatch on. [`NoPopulation`](@ref)
is the third.

# Examples
```julia
Population(5000)
Population([1500, 3500]; groups = ["child", "adult"])
BranchingProcess([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5), Gamma(2.0, 3.0);
    population = Population([1500, 3500]))
```
"""
abstract type Population end

"""One group of `size` people: `Population(size)`."""
struct SinglePopulation <: Population
    size::Int
    function SinglePopulation(size::Integer)
        size >= 1 || throw(ArgumentError("population size must be ≥ 1"))
        return new(Int(size))
    end
end

"""One group per entry of `sizes`, optionally named by `groups`:
`Population(sizes; groups)`."""
struct GroupedPopulation{G} <: Population
    sizes::Vector{Int}
    groups::G
    function GroupedPopulation(sizes::AbstractVector{<:Integer}, groups)
        isempty(sizes) && throw(ArgumentError("a population needs at least one group"))
        all(>=(0), sizes) ||
            throw(ArgumentError("group sizes must not be negative, got $(collect(sizes))"))
        sum(sizes) >= 1 || throw(ArgumentError("the group sizes add up to 0"))
        if !(groups isa NoTypeLabels)
            length(groups) == length(sizes) || throw(
                ArgumentError(
                    "`groups` has $(length(groups)) entries for $(length(sizes)) group sizes"
                )
            )
            allunique(groups) || throw(ArgumentError("`groups` are not unique"))
            groups = collect(groups)
        end
        return new{typeof(groups)}(Vector{Int}(sizes), groups)
    end
end

Population(size::Integer) = SinglePopulation(size)
function Population(sizes::AbstractVector{<:Integer}; groups = NoTypeLabels())
    return GroupedPopulation(sizes, groups)
end

# What a model's `population_size` seam may return is turned into a population.
# `population_size` is the total, so an `Int` is one group and a model written
# before populations existed keeps working.
_as_population(::NoPopulation) = NoPopulation()
_as_population(p::Population) = p
_as_population(n::Integer) = SinglePopulation(n)

"""The people `model` is run over, as a [`Population`](@ref) or
`NoPopulation()` if unlimited. A model that stores a population defines this;
one that only defines `population_size` gets the population of that size."""
population(m::TransmissionModel) = _as_population(population_size(m))

total_size(::NoPopulation) = NoPopulation()
total_size(p::SinglePopulation) = p.size
total_size(p::GroupedPopulation) = sum(p.sizes)

# The shorthand `population_size = n` and the `population` keyword are two
# ways to give one fact; giving both is an error.
_population_input(::NoPopulation, ::NoPopulation) = NoPopulation()
_population_input(p::Population, ::NoPopulation) = p
_population_input(::NoPopulation, n::Integer) = SinglePopulation(n)
_population_input(::NoPopulation, p::Population) = p
function _population_input(population, population_size)
    throw(ArgumentError("give `population` or `population_size`, not both"))
end
function _population_input(::Population, ::Population)
    throw(ArgumentError("give `population` or `population_size`, not both"))
end

# The type labels of a `BranchingProcess` come from the population's `groups`
# or from `type_labels`, but not from both.
_group_labels(::Union{NoPopulation, SinglePopulation}) = NoTypeLabels()
function _group_labels(p::GroupedPopulation)
    return p.groups isa NoTypeLabels ? NoTypeLabels() : string.(p.groups)
end
_type_labels(population, type_labels) = _merge_labels(_group_labels(population), type_labels)
_merge_labels(::NoTypeLabels, type_labels) = type_labels
_merge_labels(from_groups, ::NoTypeLabels) = from_groups
_merge_labels(::NoTypeLabels, ::NoTypeLabels) = NoTypeLabels()
function _merge_labels(from_groups, type_labels)
    throw(ArgumentError("give the type labels as `type_labels` or as the population's `groups`, not both"))
end

# The cases so far in each group, kept by the engine for a grouped population
# and empty for any other.
_empty_group_cases(::Union{NoPopulation, SinglePopulation}) = Int[]
_empty_group_cases(p::GroupedPopulation) = zeros(Int, length(p.sizes))
_empty_group_cases(::Any) = Int[]

# A model over a closed population, which every person it simulates belongs to,
# needs one: the `population` keyword or the `population_size` shorthand.
function _closed_population(population, population_size)
    pop = _population_input(population, population_size)
    pop isa NoPopulation && throw(
        ArgumentError("a population is needed: give `population` or `population_size`")
    )
    return pop
end

_single_population(p::SinglePopulation, ::AbstractString) = p
function _single_population(::GroupedPopulation, model::AbstractString)
    throw(
        ArgumentError(
            "$model has one group, so its population is `Population(n)`; " *
                "use MixingProcess for sizes by group"
        )
    )
end

# ── Group membership ─────────────────────────────────────────────────
#
# A grouped population fixes how many people each group has, so the model
# writes the group attribute itself, before the population characteristics are
# drawn (an attribute can then read it) and in numbers that match the sizes
# exactly. Who belongs to which group is random.

struct _NoAssignment end
_prepare!(::_NoAssignment, ind, i) = nothing
_verify(::_NoAssignment, ind, i) = nothing

struct _GroupAssignment
    keys::Tuple
    groups::Vector{Tuple}   # the group of each person, by position
end

function _prepare!(a::_GroupAssignment, ind, i)
    for (k, v) in zip(a.keys, a.groups[i])
        ind.state[k] = v
    end
    return nothing
end

function _verify(a::_GroupAssignment, ind, i)
    for (k, v) in zip(a.keys, a.groups[i])
        isequal(get(ind.state, k, missing), v) && continue
        throw(
            ArgumentError(
                "the attributes write `:$k`, which the population assigns from its " *
                    "group sizes. Leave it out of the attributes, or leave the sizes out " *
                    "of the population (`Population(n)`) to draw the groups from them."
            )
        )
    end
    return nothing
end

_group_assignment(::Union{NoPopulation, SinglePopulation}, rng, keys) = _NoAssignment()
function _group_assignment(pop::GroupedPopulation, rng::AbstractRNG, keys::Tuple)
    isempty(keys) && throw(
        ArgumentError(
            "a population with sizes by group needs `mixing_by`, the attributes " *
                "that hold each person's group"
        )
    )
    labels = _group_tuples(pop.groups, length(pop.sizes), length(keys))
    people = Tuple[]
    for (label, size) in zip(labels, pop.sizes)
        append!(people, fill(label, size))
    end
    return _GroupAssignment(keys, shuffle(rng, people))
end

# The group of each size as a tuple of attribute values, one per `mixing_by`
# key: a bare value stands for a one-element tuple, and without `groups` group
# `i` is `(i,)`.
function _group_tuples(::NoTypeLabels, n::Int, nkeys::Int)
    nkeys == 1 || throw(
        ArgumentError("`mixing_by` has $nkeys attributes, so `groups` must give each group's values")
    )
    return [(i,) for i in 1:n]
end
function _group_tuples(groups, n::Int, nkeys::Int)
    tuples = Tuple[g isa Tuple ? g : (g,) for g in groups]
    all(t -> length(t) == nkeys, tuples) || throw(
        ArgumentError("each of the population's `groups` needs $nkeys values, one for each `mixing_by` attribute")
    )
    return tuples
end
