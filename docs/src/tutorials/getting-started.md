# Getting started

The core workflow is covered here: defining a model, running simulations,
and extracting results.

## Defining a transmission model

At minimum, a branching process is defined by an offspring distribution
(how many secondary cases each case produces). If you also want timing
(which you need for interventions), add a generation time distribution.

```@example gettingstarted
using EpiBranch
using Distributions
using StableRNGs

model = BranchingProcess(
    NegBin(2.5, 0.16),         # R₀ = 2.5, dispersion k = 0.16
    LogNormal(1.6, 0.5)        # generation time
)
```

[`NegBin`](@ref) is a convenience constructor for the Negative Binomial,
parameterised by mean (R) and dispersion (k), matching the epidemiological
convention.

## Running a simulation

```@example gettingstarted
rng = StableRNG(123)
state = simulate(model;
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Cases: $(state.cumulative_cases), Extinct: $(state.extinct)")
```

A `SimulationState` is returned, containing all individuals and
outbreak metadata.

## Adding clinical parameters

To model symptom onset (needed for isolation-based interventions), provide
an incubation period distribution via [`clinical_presentation`](@ref):

```@example gettingstarted
rng = StableRNG(42)
state = simulate(model;
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)

# Check onset times are set
ind = state.individuals[1]
println("Infection: $(round(ind.infection_time, digits=1)), Onset: $(round(onset_time(ind), digits=1))")
```

## Adding interventions

You can combine interventions by passing them as a vector:

```@example gettingstarted
iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.5, delay = Exponential(1.5))

rng = StableRNG(42)
state = simulate(model;
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Cases: $(state.cumulative_cases)")
println("Isolated: $(count(is_isolated, state.individuals))")
println("Traced: $(count(is_traced, state.individuals))")
```

## Batch simulation

Run many replicates to estimate containment probability:

```@example gettingstarted
rng = StableRNG(42)
results = simulate_batch(model, 500;
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 5000),
    rng = rng,
)
println("Containment probability: $(round(containment_probability(results), digits=3))")
```

## Contacts vs cases

Every potential transmission is tracked. Contacts that weren't
successfully infected are stored alongside cases:

```@example gettingstarted
n_total = length(state.individuals)
n_infected = count(is_infected, state.individuals)
println("Total contacts: $n_total")
println("Infected (cases): $n_infected")
println("Not infected: $(n_total - n_infected)")
```

Intervention effort can be tracked this way. For more information, see [Interventions](interventions.md).

## Outputs

Simulation state can be converted to DataFrames with several output functions.

```@example gettingstarted
using DataFrames, Dates

# Line list (cases only)
ll = linelist(state; reference_date = Date(2024, 1, 1), rng = StableRNG(99))
println("Line list: $(nrow(ll)) rows, $(ncol(ll)) columns")
first(ll, 3)
```

```@example gettingstarted
# Contacts table (all contacts, with was_case flag)
ct_df = contacts(state; reference_date = Date(2024, 1, 1))
println("Contacts: $(nrow(ct_df)) ($(count(ct_df.was_case)) infected)")
```

```@example gettingstarted
# Chain statistics
cs = chain_statistics(state)
cs
```

```@example gettingstarted
# Effective R per generation
r_df = generation_R(state)
first(r_df, 5)
```

## Analytical functions

Some quantities have closed-form solutions — no simulation needed:

```@example gettingstarted
println("P(extinction): $(round(extinction_probability(model), digits=3))")
println("P(epidemic):   $(round(epidemic_probability(model), digits=3))")
println("Top 20% cause $(round(proportion_transmission(model; prop_cases=0.2) * 100, digits=1))% of transmission")
```

## Conditioned simulation

Generate outbreaks of a specific size via rejection sampling:

```@example gettingstarted
rng = StableRNG(42)
state = simulate(model;
    condition = 50:100,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)
println("Outbreak size: $(state.cumulative_cases) (target: 50-100)")
```

## Next steps

- [Interventions](interventions.md) — combining interventions and the competing risks framework
- [Multi-type models](multi-type.md) — age-structured and heterogeneous transmission
- [Line lists and contacts](linelist.md) — generating epidemiological data
- [Chain statistics and likelihood](chains.md) — inference from chain data
- [Analytical functions](analytical.md) — closed-form solutions
