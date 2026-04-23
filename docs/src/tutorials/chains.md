# Chain statistics, likelihood, and fitting

Compute chain size and length statistics, evaluate likelihoods, and fit
offspring distributions — analytically where possible, simulation-based
otherwise.

## Chain statistics

```@example chains
using EpiBranch
using Distributions
using DataFrames
using StableRNGs

model = BranchingProcess(Poisson(0.9), Exponential(5.0))

rng = StableRNG(42)
state = simulate(model; sim_opts = SimOpts(n_initial = 20, max_cases = 10_000), rng = rng)

cs = chain_statistics(state)
first(cs, 10)
```

```@example chains
println("Mean size: $(round(mean(cs.size), digits=2)), Max: $(maximum(cs.size))")
println("Mean length: $(round(mean(cs.length), digits=2))")
```

## Analytical chain size distributions

For certain offspring distributions, the chain size distribution has a
closed form:

```@example chains
# Poisson → Borel
d_borel = chain_size_distribution(Poisson(0.8))
println("Borel(0.8) mean: $(round(mean(d_borel), digits=2))")

# NegBin → Lagrange inversion formula
d_nb = chain_size_distribution(NegBin(0.8, 0.5))
println("NegBin chain size P(1): $(round(pdf(d_nb, 1), digits=4))")
```

## Likelihood

Given observed chain sizes, evaluate the log-likelihood using
`loglikelihood` with data wrapped in
[`ChainSizes`](@ref):

```@example chains
data = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])

# Compare offspring parameters
for λ in [0.3, 0.5, 0.7, 0.9]
    ll = loglikelihood(data, Poisson(λ))
    println("Poisson($λ): LL = $(round(ll, digits=2))")
end
```

With Negative Binomial offspring:

```@example chains
ll = loglikelihood(data, NegBin(0.9, 0.5))
println("NegBin(R=0.9, k=0.5): LL = $(round(ll, digits=2))")
```

### Imperfect observation

Account for incomplete case ascertainment by wrapping a model in
[`PartiallyObserved`](@ref). Each case in a chain is detected
independently with the given probability:

```@example chains
data = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])
base = BranchingProcess(Poisson(0.9))
full = PartiallyObserved(base, 1.0)
partial = PartiallyObserved(base, 0.7)
println("Full observation:  $(round(loglikelihood(data, full), digits=2))")
println("70% observation:   $(round(loglikelihood(data, partial), digits=2))")
```

Call `PartiallyObserved` with just the detection probability and you
get back a function that wraps whatever model you apply it to, so you
can build the observation layer with Julia's pipe:

```@example chains
ll_pipe = loglikelihood(data, base |> PartiallyObserved(0.7))
println("Via pipe: $(round(ll_pipe, digits=2))")
```

Stacking the same wrapper compounds: two stages of per-case detection
at 0.5 give the same observed distribution as a single stage at 0.25.

```@example chains
ll_stacked = loglikelihood(data,
    base |> PartiallyObserved(0.5) |> PartiallyObserved(0.5))
ll_single = loglikelihood(data, base |> PartiallyObserved(0.25))
println("Stacked: $(round(ll_stacked, digits=2))")
println("Single:  $(round(ll_single, digits=2))")
```

### Offspring counts

If you have direct observations of secondary case counts (who infected
whom), wrap them in [`OffspringCounts`](@ref):

```@example chains
offspring_data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 5, 0])
ll = loglikelihood(offspring_data, Poisson(1.4))
println("Offspring LL: $(round(ll, digits=2))")
```

### Chain lengths

```@example chains
length_data = ChainLengths([0, 1, 0, 2, 1, 0, 0, 3, 0, 1])
ll = loglikelihood(length_data, Poisson(0.5))
println("Chain length LL: $(round(ll, digits=2))")
```

## Simulation-based likelihood

For models with interventions, use the simulation-based likelihood by
passing a [`BranchingProcess`](@ref) model instead of a distribution:

```@example chains
model = BranchingProcess(Poisson(2.0), Exponential(5.0))
iso = Isolation(delay = Exponential(2.0))

ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2]), model;
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    n_sim = 1000,
    rng = StableRNG(42),
)
println("LL with interventions: $(round(ll, digits=2))")
```

Forward simulation and likelihood evaluation share the same code path.
Likelihoods can therefore be evaluated under interventions — something
not possible when simulation and inference are in separate packages.

## Fitting

Use `fit` to find the maximum likelihood offspring distribution.
The same interface works for all data types:

```@example chains
# From offspring counts
offspring_data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 5, 0])
d = fit(Poisson, offspring_data)
println("Poisson MLE from offspring: R = $(round(mean(d), digits=2))")
```

```@example chains
d = fit(NegativeBinomial, offspring_data)
println("NegBin MLE from offspring: R = $(round(mean(d), digits=2)), k = $(round(d.r, digits=2))")
```

```@example chains
# From chain sizes (subcritical only)
rng = StableRNG(42)
model = BranchingProcess(Poisson(0.5), Exponential(5.0))
states = simulate_batch(model, 500; rng = rng)
sizes = Int[]
for s in states
    cs = chain_statistics(s)
    append!(sizes, cs.size)
end

d = fit(Poisson, ChainSizes(sizes))
println("Poisson MLE from chain sizes: R = $(round(mean(d), digits=2))")
```

### Bayesian inference with Turing.jl

The `loglikelihood` functions work directly with
[Turing.jl](https://turinglang.org) via `@addlogprob!`:

```julia
using Turing

@model function chain_model(data)
    R ~ LogNormal(-0.5, 1.0)
    Turing.@addlogprob! loglikelihood(ChainSizes(data), Poisson(R))
end
```
