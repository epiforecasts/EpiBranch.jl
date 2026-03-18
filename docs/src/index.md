# EpiBranch.jl

Stochastic branching process simulation for infectious disease epidemiology.

EpiBranch.jl brings together the functionality of several R packages into
a single Julia package with one shared simulation engine:

- **Outbreak simulation with interventions** (cf. [ringbp](https://github.com/epiforecasts/ringbp))
- **Line list and contact tracing data** (cf. [simulist](https://github.com/epiverse-trace/simulist))
- **Chain statistics and likelihood** (cf. [epichains](https://github.com/epiverse-trace/epichains))
- **Offspring distribution analytics** (cf. [superspreading](https://github.com/epiverse-trace/superspreading))

## Design

1. **Offspring draw is decoupled from timing and interventions.** Contacts
   are drawn from a branching process, with the generation time distribution
   used to assign timing. Interventions then act as competing risks on
   whether each contact is actually infected. See [Design](design.md).

2. **Interventions are composable.** Isolation, contact tracing, vaccination,
   and PEP all stack in a vector and operate through the same hooks.

3. **All contacts are tracked.** Every potential transmission event is stored
   (infected and non-infected), so you can compute effort metrics, build
   simulist-style contacts tables, and track who was traced or vaccinated.

4. **Likelihood evaluation uses the same engine.** You can estimate offspring
   parameters under interventions with the same simulation code used for
   forward simulation.

5. **Individuals are extensible.** Each individual carries a `state` dict
   where interventions and attributes functions set their own fields. You add
   demographics, risk groups, or anything else without modifying the engine.

## Quick start

```@example quickstart
using EpiBranch
using Distributions
using StableRNGs

# Define a model
model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

# Add interventions
iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.5, delay = Exponential(1.5))

# Simulate
rng = StableRNG(42)
results = simulate_batch(model, 500;
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 5000),
    rng = rng,
)

containment_probability(results)
```

## Installation

EpiBranch.jl is not yet registered. Install from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/epiforecasts/EpiBranch.jl")
```
