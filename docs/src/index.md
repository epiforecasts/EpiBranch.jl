# EpiBranch.jl

**EpiBranch.jl** is a package for stochastic branching process simulation
of infectious disease outbreaks. Functionality from
several R packages (cf.
[ringbp](https://github.com/epiforecasts/ringbp),
[simulist](https://github.com/epiverse-trace/simulist),
[epichains](https://github.com/epiverse-trace/epichains),
[superspreading](https://github.com/epiverse-trace/superspreading)) is
brought together in a single package with one shared simulation engine.

The offspring draw is decoupled from timing and interventions: contacts
are drawn from a branching process, generation times are assigned
independently, and interventions act as competing risks on whether
each contact is actually infected. For more information on the design,
see [Design](design.md).

## Quick start

```@example quickstart
using EpiBranch
using Distributions
using StableRNGs

model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.5, delay = Exponential(1.5))

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

**EpiBranch.jl** is not yet registered. It can be installed from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/epiforecasts/EpiBranch.jl")
```
