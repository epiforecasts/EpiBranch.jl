"""
    PerGroup(group => budget, ...)

A budget given for each group of a population rather than for everyone,
accepted wherever a capacity wrapper takes a budget:
`budget_per_period` of [`CapacityConstrained`](@ref) and `capacity` of
[`HeldCapacity`](@ref). Each `budget` is a number or, for
`budget_per_period`, a schedule such as [`Steps`](@ref) or [`Seasonal`](@ref).
A budget given as a plain number or schedule is one pool shared by every group.

The groups are those of the model's [`Population`](@ref) and nowhere else: the
`groups` of a grouped population, or `1, 2, ...` when it names none. A name
written as a `Symbol` matches the same name written as a string. Every group of
the population needs an entry, and an entry naming no group is an error, so a
misspelt group is never a scenario with no doses. A population with no groups
has nothing to name, so `PerGroup` is refused there.

Every model that reads a budget reads this one the same way.

# Examples
```julia
using EpiBranch, Distributions

# Separate dose series for the under 5s and the over 5s
vaccination = CapacityConstrained(
    GroupVaccination(group_key = :age, efficacy = 0.9);
    budget_per_period = PerGroup(
        :under5 => 10.0, :over5 => Steps([7.0], [0.0, 50.0])
    ),
    period = 1.0
)
process = MixingProcess(
    population = Population([300, 700]; groups = ["under5", "over5"]),
    mixing_by = (:age,), rate = [0.2 0.1; 0.1 0.2]
)
```
"""
struct PerGroup{V}
    groups::Vector{Any}
    budgets::Vector{V}
    function PerGroup(groups::Vector{Any}, budgets::Vector{V}) where {V}
        isempty(groups) && throw(ArgumentError("PerGroup needs at least one group"))
        length(groups) == length(budgets) ||
            throw(ArgumentError("PerGroup has $(length(groups)) groups for $(length(budgets)) budgets"))
        allunique(groups) || throw(
            ArgumentError("PerGroup names a group more than once: $(join(repr.(groups), ", "))")
        )
        return new{V}(groups, budgets)
    end
end

function PerGroup(entries::Pair...)
    isempty(entries) && throw(ArgumentError("PerGroup needs at least one group"))
    groups = Any[_group_name(first(e)) for e in entries]
    budgets = Any[last(e) for e in entries]
    return PerGroup(groups, budgets)
end
PerGroup(entries::AbstractDict) = PerGroup(collect(entries)...)

function Base.show(io::IO, pg::PerGroup)
    print(io, "PerGroup(")
    join(io, (string(repr(g), " => ", b) for (g, b) in zip(pg.groups, pg.budgets)), ", ")
    return print(io, ")")
end

# A group as the budget and the model each name it: symbols and strings are the
# same name, and a one-element tuple (a group of a `mixing_by` population with
# one attribute) is its element.
_group_name(g::Symbol) = String(g)
_group_name(g::AbstractString) = String(g)
_group_name(g::Tuple) = length(g) == 1 ? _group_name(only(g)) : map(_group_name, g)
_group_name(g) = g

# The names of the groups of `population`, in its order: the names it was
# given, or `1, 2, ...`.
function _group_names(population::GroupedPopulation)
    population.groups isa NoTypeLabels && return Any[i for i in eachindex(population.sizes)]
    return Any[_group_name(g) for g in population.groups]
end
function _group_names(population)
    throw(
        ArgumentError(
            "a budget per group (PerGroup) needs a population with groups: give the " *
                "model `population = Population(sizes)`"
        )
    )
end

# The budget of each group of `names`, in order. Fails when a group has no
# entry, or an entry names a group the population does not have.
function _group_budgets(pg::PerGroup, names::AbstractVector)
    missing_groups = [n for n in names if !any(isequal(n), pg.groups)]
    isempty(missing_groups) || throw(
        ArgumentError(
            "PerGroup has no budget for the group(s) " *
                "$(join(repr.(missing_groups), ", ")); give every group of the population one"
        )
    )
    unknown = [g for g in pg.groups if !any(isequal(g), names)]
    isempty(unknown) || throw(
        ArgumentError(
            "PerGroup names $(join(repr.(unknown), ", ")), which " *
                "$(length(unknown) == 1 ? "is" : "are") not among the population's groups " *
                "$(join(repr.(names), ", "))"
        )
    )
    return [pg.budgets[findfirst(isequal(n), pg.groups)] for n in names]
end
_group_budgets(pg::PerGroup, population::Population) = _group_budgets(pg, _group_names(population))
_group_budgets(pg::PerGroup, ::NoPopulation) = _group_names(NoPopulation())

# The budget of the group called `name`, as `_group_budgets` checked it.
function _budget_of(pg::PerGroup, name)
    i = findfirst(isequal(_group_name(name)), pg.groups)
    i === nothing && throw(
        ArgumentError("PerGroup has no budget for the group $(repr(name))")
    )
    return pg.budgets[i]
end

# A budget that is one pool for everyone has no group to read; a `PerGroup`
# reads its own.
_budget_of(budget, name) = budget

_has_schedule(::Real) = false
_has_schedule(pg::PerGroup) = any(_has_schedule, pg.budgets)
_has_schedule(_) = true
