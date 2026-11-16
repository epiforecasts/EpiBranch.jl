# What each model applies

One specification, a [`ModelSpec`](@ref) with its interventions, means the same thing to every model that reads it.
What differs is what a model keeps: individuals, counts of cases by type, or compartments by mixing group.
A model with no individual to offer an intervention to applies only an intervention that says what it does to a group, and refuses any other, in the same words as every other model that cannot apply it.

The three kinds of model are:

- **Individuals**: [`simulate`](@ref) on a model of people, such as [`BranchingProcess`](@ref), [`HomogeneousProcess`](@ref), [`MixingProcess`](@ref) or a network or household model. It applies every intervention.
- **Counts by type**: [`simulate_by_generation`](@ref) on a [`BranchingProcess`](@ref), which keeps the number of cases and susceptibles of each type.
- **Compartments by mixing group**: [`MixingMeanFieldProcess`](@ref), [`MixingLinearNoiseProcess`](@ref) and [`HybridProcess`](@ref), which keep susceptible, infectious, removed and vaccinated compartments for each mixing group.

## One budget, or one per group

A dose budget limits the number of people an intervention can reach.
Given as a number or a schedule, `budget_per_period` is one pool shared by every group, in every model.
A [`PerGroup`](@ref) gives each group of the model's [`Population`](@ref) its own budget, as a number or a schedule:

```julia
doses = PerGroup(
    :under5 => 10.0,
    :over5 => Steps([7.0, 14.0], [0.0, 50.0, 0.0]),
)
campaign = CapacityConstrained(
    GroupVaccination(efficacy = 0.9, group_key = :age);
    budget_per_period = doses, period = 1.0
)
process = MixingProcess(
    population = Population([300, 700]; groups = ["under5", "over5"]),
    mixing_by = (:age,), mixing = FrequencyDependent([0.5 0.1; 0.1 0.5])
)
```

The groups are those of the population and nowhere else: the `groups` of a grouped population, or `1, 2, ...` when it names none.
A group of the population missing from the mapping is an error, and so is a name that is no group of the population, so that a mistyped name never passes as a scenario with no doses.
To give a group no doses, give it a budget of `0`.
A population with no groups has none to name, and `PerGroup` is refused there.
`PerGroup` also gives the places of a [`HeldCapacity`](@ref) per group, and it works with every intervention that has a budget, including one written outside the package.

## Support by kind of model

| Specification | Individuals | Counts by type | Compartments by mixing group |
|:--------------|:------------|:---------------|:-----------------------------|
| `GroupVaccination` under `CapacityConstrained`, shared budget | yes | yes | yes |
| ... with a `PerGroup` budget | yes | yes | yes |
| ... under `Scheduled` with a `start_time` and `end_time` | yes | yes | yes |
| ... under `Scheduled` gated on cases, a trigger or a predicate | yes | refused | refused |
| `CapacityConstrained` with `on_refusal = DeferRefused()` | yes | refused | refused |
| `RingVaccination`, `MassVaccination` | yes | refused | refused |
| `Isolation`, `ContactTracing`, `HeldCapacity` and any other intervention, alone or under `CapacityConstrained` (shared or `PerGroup` budget) | yes | refused | refused |

A model that cannot apply a specification raises an `ArgumentError` that names what it cannot apply and ends with the same list of what it reads: a `GroupVaccination`, alone or under `Scheduled` (a `start_time` and `end_time` only) and `CapacityConstrained` (one per campaign, with a shared or `PerGroup` budget and `on_refusal = DropRefused()`).
A model that keeps counts or compartments also refuses, and names, any input it cannot represent, such as a distribution as `efficacy`, a `delay_to_immunity` or `waning`.
The docstring of each model lists them.

An intervention written outside the package opts in to counts and compartments by defining `EpiBranch.count_dosing`, which says what it does to a group.
Without it, every model of counts or compartments refuses it.

The test file `test/test_engine_support.jl` runs every kind of model against this table.
If what a model reads changes, the test fails until the table is updated.
