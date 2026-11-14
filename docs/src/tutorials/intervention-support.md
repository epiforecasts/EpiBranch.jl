# What each engine supports

A `ModelSpec` has the same meaning in every engine that reads it. An engine that cannot apply it refuses it in the same words.
This page lists which parts of the specification each engine applies, and shows how to give each group its own budget of doses.

The engines are:

- **Individual-based simulation**: `simulate` on a model of individuals, such as `BranchingProcess`, `HomogeneousProcess`, `MixingProcess`, or a network or household structure. It applies every intervention.
- **Generation counts**: [`simulate_by_generation`](@ref), which keeps counts of cases and susceptibles of each type rather than individuals.
- **Mixing approximations**: [`MixingMeanFieldProcess`](@ref) and [`MixingLinearNoiseProcess`](@ref), which keep compartments for each mixing group.
- **Hybrid**: [`HybridProcess`](@ref), which runs the individual-based simulation and hands over to a mixing approximation. It reads the same specification as the approximation.

## One budget, or one per group

A dose budget limits the number of people an intervention can reach.
Given as a number or a schedule, `budget_per_period` is shared by everyone in every engine, whatever the groups of the model.
A [`PerGroup`](@ref) gives each group its own budget, as a number or a schedule:

```julia
doses = PerGroup(
    :under5 => 10.0,
    :over5 => Steps([7.0, 14.0], [0.0, 50.0, 0.0]),
)
campaign = CapacityConstrained(
    GroupVaccination(efficacy = 0.9, group_key = :age_band);
    budget_per_period = doses, period = 1.0
)
```

The groups are the values of the `group_key` of a [`GroupVaccination`](@ref).
For an intervention without one, such as [`Isolation`](@ref) or [`ContactTracing`](@ref), `PerGroup` names the state key that holds a person's group: `PerGroup("north" => 5.0, "south" => 2.0; key = :district)`.
A group missing from the mapping raises an error, so a mistyped name cannot pass as a scenario. To give a group no doses, give it a budget of `0`.

The groups of `simulate_by_generation` are the types of the model, numbered from one, and its `group_key` is `:type`.
In the mixing approximations the groups are the values of the key the model mixes by.

## Support by engine

| Specification | Individual-based | Generation counts | Mixing approximations and hybrid |
|:--------------|:-----------------|:------------------|:---------------------------------|
| `GroupVaccination` under `CapacityConstrained`, shared budget | yes | yes | yes |
| ... with a `PerGroup` budget | yes | yes (key `:type`) | yes (the `group_key`) |
| ... with no budget (unlimited doses) | yes | refused: no number of doses | yes |
| ... under `Scheduled` with a `start_time` and `end_time` | yes | yes | yes |
| ... under `Scheduled` gated on cases, a trigger or a predicate | yes | refused | refused |
| `CapacityConstrained` with `on_refusal = DeferRefused()` | yes | refused | refused |
| `RingVaccination`, `MassVaccination` | yes | refused | refused |
| `Isolation`, `ContactTracing` and other interventions, alone or under `CapacityConstrained` (shared or `PerGroup` budget) | yes | refused | refused |
| More than one campaign | yes | yes, one per `GroupVaccination` | refused |

An engine that cannot apply a specification raises an `ArgumentError` that names what it cannot apply and ends with the same list of what it reads: a `GroupVaccination`, alone or under `Scheduled` (a `start_time` and `end_time` only) and `CapacityConstrained` (one per campaign, with a shared or `PerGroup` budget and `on_refusal = DropRefused()`).
The engines that read a campaign without simulating people also refuse, and name, any input that a count or a compartment cannot represent, such as a distribution as `efficacy`, a `delay_to_immunity` or `waning`. The docstring of each engine lists them.

The test file `test/test_engine_support.jl` runs each engine against this table. If what an engine reads changes, the test fails until the table is updated.
