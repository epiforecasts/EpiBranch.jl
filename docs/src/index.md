# EpiBranch.jl

A unified framework for stochastic branching process simulation of infectious
disease outbreaks.

EpiBranch.jl combines the functionality of several R packages into a single
Julia package with a shared simulation engine:

- **Outbreak simulation with interventions** (cf. [ringbp](https://github.com/epiforecasts/ringbp))
- **Line list and contact tracing data** (cf. [simulist](https://github.com/epiverse-trace/simulist))
- **Chain statistics and likelihood** (cf. [epichains](https://github.com/epiverse-trace/epichains))
- **Offspring distribution analytics** (cf. [superspreading](https://github.com/epiverse-trace/superspreading))

## Key design principles

1. **Offspring draw decoupled from timing and interventions.** The branching
   process generates contacts; generation times assign timing; interventions
   act as competing risks. See [Design](design.md).

2. **Composable interventions.** Isolation, contact tracing, and future
   interventions (vaccination, PEP) stack in a vector and operate through
   the same mechanism.

3. **Full contact tracking.** Every potential transmission is stored — infected
   and non-infected — enabling effort metrics and simulist-style contacts tables.

4. **Simulation-based likelihood.** Estimate offspring parameters under
   interventions using the same engine that runs forward simulations.

5. **Extensible individuals.** The `state` dict on each individual holds
   any data you need — interventions define their own fields, users add
   demographics or custom state.

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
    sim_opts = SimOpts(
        max_cases = 5000,
        incubation_period = LogNormal(1.5, 0.5),
    ),
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
