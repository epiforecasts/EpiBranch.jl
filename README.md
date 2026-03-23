
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

Benchmarks against the R [epichains](https://github.com/epiverse-trace/epichains) package (median times, same scenarios):

| Scenario | R | Julia | Speedup |
|---|---|---|---|
| 1000 chains, Poisson(0.9) | 23.9 ms | 2.0 ms | 12x |
| 1000 chains, NegBin(0.8, 0.5) | 11.8 ms | 1.2 ms | 10x |
| 1000 chains + generation time | 31.6 ms | 2.6 ms | 12x |
| Chain statistics | 0.48 ms | 0.27 ms | 1.8x |
| Analytical log-likelihood | 0.16 ms | 0.008 ms | 20x |

Benchmark scripts are in [`benchmarks/`](benchmarks/).

## Documentation

For information on how to use the package, see the [documentation](https://epiforecasts.github.io/EpiBranch.jl/).
