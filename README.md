
# Branching process models for infectious disease epidemiology <img src="docs/src/assets/logo.svg" align="right" height="139" alt="EpiBranch.jl logo" />

<!-- badges: start -->
[![CI](https://github.com/epiforecasts/EpiBranch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/epiforecasts/EpiBranch.jl/actions/workflows/CI.yml)
<!-- badges: end -->

[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl) is a Julia package for stochastic branching process simulation of infectious disease outbreaks. It unifies the functionality of several R packages ([ringbp](https://github.com/epiforecasts/ringbp), [simulist](https://github.com/epiverse-trace/simulist), [epichains](https://github.com/epiverse-trace/epichains), [superspreading](https://github.com/epiverse-trace/superspreading), [pepbp](https://github.com/sophiemeakin/pepbp)) in a single package with one shared simulation engine.

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
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 5000),
)

containment_probability(results)
```

## Why Julia?

The R packages this draws on (ringbp for simulation, epichains for chain statistics, superspreading for offspring fitting) don't talk to each other, so a couple of useful analyses turn into engineering jobs rather than analyses. Two examples people often want and can't easily do: evaluating a likelihood under intervention scenarios, and fitting a Bayesian model with interventions inside the loop.

Putting the pieces in one package lets you:

- Fit offspring distributions, chain sizes, or chain lengths with one `loglikelihood` call. The function picks the right method for whichever shape of data you give it.
- Estimate R and k under whichever intervention scenarios you care about. The likelihood comes from running the same simulator you'd use to generate forward outbreaks.
- Stack interventions in any combination (isolation, contact tracing, ring vaccination, time-varying policies) and have them interact correctly through competing risks on individual cases.
- Pass a model with interventions straight into [Turing.jl](https://turinglang.org) for a posterior over R, k, and any other parameters you're fitting, under whatever intervention scenario you specified.

### Performance

Benchmark scripts comparing EpiBranch.jl against R equivalents (epichains, ringbp) are in [`benchmarks/`](benchmarks/). See the [benchmarks page](https://epiforecasts.github.io/EpiBranch.jl/dev/benchmarks) in the documentation for details.

## Documentation

For information on how to use the package, see the [documentation](https://epiforecasts.github.io/EpiBranch.jl/).
