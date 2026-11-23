# Interventions with budgets and schedules

An intervention with a dose budget or a schedule behaves the same in every model that applies it.
This page describes a model by four components: the sharing of its budget, the timing of its schedule, the treatment of a person the budget turns away, and the groups it reads.
[`simulate`](@ref) on a model of individuals, such as [`BranchingProcess`](@ref), [`HomogeneousProcess`](@ref), [`MixingProcess`](@ref) or a network or household model, applies every intervention.
[`simulate_by_generation`](@ref), [`MixingMeanFieldProcess`](@ref), [`MixingLinearNoiseProcess`](@ref) and [`HybridProcess`](@ref) work on groups rather than people. They apply only the interventions listed under "What a model of groups applies". Any other intervention raises an `ArgumentError` naming it and pointing to `simulate` on a model of individuals.

## One budget, or one per group

A dose budget limits the reach of an intervention.
Given as a number or a schedule, `budget_per_period` is one pool shared by every group, in every model.
A [`PerGroup`](@ref) gives each group of the [`Population`](@ref) of the model its own budget, as a number or a schedule:

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

The groups are those of the population: the `groups` of a grouped population, or `1, 2, ...` when it names none.
A group of the population missing from the mapping is an error. An unknown name is also an error, so a mistyped name cannot pass as a scenario with no doses.
To give a group no doses, give it a budget of `0`.
A population with no groups has none to name. `PerGroup` raises an error there.
`PerGroup` also sets the places of a [`HeldCapacity`](@ref) per group. It works with every intervention with a budget, including one written outside the package.

## Schedule and refusal policy

A [`Scheduled`](@ref) with a `start_time` and `end_time` limits an intervention to a calendar window, in every model.
A `Scheduled` that starts after a number of cases, or that waits on a trigger or a predicate, depends on individuals. Only `simulate` on a model of individuals applies it.
By default `CapacityConstrained` drops a person the budget turns away (`on_refusal = DropRefused()`), in every model.
`on_refusal = DeferRefused()` books that person into a later period. Only `simulate` on a model of individuals does this.

## What a model of groups applies

`simulate_by_generation`, the mean-field and linear-noise approximations and `HybridProcess` apply a [`GroupVaccination`](@ref), alone or under `Scheduled` (a `start_time` and `end_time` only) and `CapacityConstrained` (one per campaign, with a shared or `PerGroup` budget and `on_refusal = DropRefused()`).
They also name any input they cannot represent, such as a distribution as `efficacy`, a `delay_to_immunity` or `waning`.
The docstring of each model lists them.
[`RingVaccination`](@ref), [`MassVaccination`](@ref), [`Isolation`](@ref), [`ContactTracing`](@ref), [`HeldCapacity`](@ref) and any other intervention need `simulate` on a model of individuals.

An intervention written outside the package is applied to groups by defining `EpiBranch.gives_group_doses` as `true` and the `EpiBranch.dose_*` facts of its dose; see its docstring.
Without it, it needs `simulate` on a model of individuals.

The test file `test/test_engine_support.jl` runs every kind of model against this page.
If a model reads a different input, the test fails until the page is updated.
