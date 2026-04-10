
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

## Why Julia?

In R, branching process simulation, chain statistics, line list generation, intervention modelling, and offspring fitting live in separate CRAN packages with separate codebases. This means, for example, that likelihoods cannot be evaluated under interventions because the simulation engine and the likelihood code don't share a common interface.

Julia's multiple dispatch makes it natural to unify these into a single package where components compose freely:

- **Same `loglikelihood` function** works with offspring counts, chain sizes, or chain lengths — dispatch on the data type selects the right method
- **Same `fit` function** estimates offspring distribution parameters from any of those data types
- **Same simulation engine** runs with or without interventions, and the simulation-based likelihood reuses it directly — enabling likelihood evaluation under interventions
- **Stack interventions freely**: isolation, contact tracing, ring vaccination, and time-dependent scheduling all go in a vector and interact through competing risks on individual state
- **AD-compatible likelihoods** work with [Turing.jl](https://turinglang.org) for Bayesian inference — including parameter estimation under interventions, which is not possible when simulation and inference live in separate packages

### Performance

Benchmark scripts comparing EpiBranch.jl against R equivalents (epichains, ringbp) are in [`benchmarks/`](benchmarks/). See the [benchmarks page](https://epiforecasts.github.io/EpiBranch.jl/dev/benchmarks) in the documentation for details.

## Documentation

For information on how to use the package, see the [documentation](https://epiforecasts.github.io/EpiBranch.jl/).
