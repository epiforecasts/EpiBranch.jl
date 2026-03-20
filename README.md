
# Branching process models for infectious disease epidemiology

<!-- badges: start -->
[![CI](https://github.com/epiforecasts/EpiBranch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/epiforecasts/EpiBranch.jl/actions/workflows/CI.yml)
<!-- badges: end -->

[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl) is a Julia package for stochastic branching process simulation of infectious disease outbreaks. It brings together the functionality of several R packages ([ringbp](https://github.com/epiforecasts/ringbp), [simulist](https://github.com/epiverse-trace/simulist), [epichains](https://github.com/epiverse-trace/epichains), [superspreading](https://github.com/epiverse-trace/superspreading), [pepbp](https://github.com/sophiemeakin/pepbp)) in a single package with one shared simulation engine.

## Installation

The package can be installed using

```julia
using Pkg
Pkg.add(url="https://github.com/epiforecasts/EpiBranch.jl")
```

## Quick start

```julia
using EpiBranch
using Distributions

model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.5, delay = Exponential(1.5))

results = simulate_batch(model, 500;
    interventions = [iso, ct],
    attributes = Disease(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 5000),
)

containment_probability(results)
```

## Documentation

For information on how to use the package, see the [documentation](https://epiforecasts.github.io/EpiBranch.jl/).
