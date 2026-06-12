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
state = simulate(model; n_initial = 20, max_cases = 10_000, rng = rng)

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

Account for incomplete case ascertainment by giving the model a
[`PerCaseObservation`](@ref): pass `observation = …` to the process
constructor. Each case is detected independently with the given
probability:

```@example chains
data = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])
full = BranchingProcess(Poisson(0.9); observation = PerCaseObservation(detection_prob = 1.0))
partial = BranchingProcess(Poisson(0.9); observation = PerCaseObservation(detection_prob = 0.7))
println("Full observation:  $(round(loglikelihood(data, full), digits=2))")
println("70% observation:   $(round(loglikelihood(data, partial), digits=2))")
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
passing a model instead of a distribution. Because the interventions and
attributes live on the model, the likelihood is evaluated under
exactly the process that produced the data, with nothing to pass twice:

```@example chains
iso = Isolation(onset_to_isolation_delay = Exponential(2.0))

model = BranchingProcess(Poisson(2.0), Exponential(5.0);
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
)

ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2]), model;
    n_sim = 1000,
    rng = StableRNG(42),
)
println("LL with interventions: $(round(ll, digits=2))")
```

Forward simulation and likelihood evaluation share the same code path.
Likelihoods can therefore be evaluated under interventions — something
not possible when simulation and inference are in separate packages — and
the same `model` drives both `simulate(model)` and `loglikelihood`.

## Fitting

The `loglikelihood` interface is the backend for both maximum-likelihood
and Bayesian fitting. For an MLE, maximise `loglikelihood` over the
parameter — with Optim.jl, Turing's `maximum_likelihood`, or, for a
single parameter, a simple grid:

```@example chains
rng = StableRNG(42)
truth = BranchingProcess(Poisson(0.5), Exponential(5.0))
states = simulate(truth, 500; rng = rng)
sizes = Int[]
for s in states
    cs = chain_statistics(s)
    append!(sizes, cs.size)
end
data = ChainSizes(sizes)

Rgrid = 0.05:0.01:0.95
R̂ = Rgrid[argmax([loglikelihood(data, Poisson(R)) for R in Rgrid])]
println("Poisson MLE from chain sizes: R = $(round(R̂, digits=2))")
```

### Bayesian inference with Turing.jl

Call [`chain_size_distribution`](@ref) (or
[`chain_length_distribution`](@ref), [`offspring_distribution`](@ref))
on the model and the data sits directly on the right-hand side of `~`:

```julia
using Turing

@model function chain_model(data)
    R ~ LogNormal(-0.5, 1.0)
    data ~ chain_size_distribution(BranchingProcess(Poisson(R)))
end
```

For a bare offspring law these return the analytical distribution
(`Borel`, `GammaBorel`, …) where one exists; when the model carries
interventions (or you pass `seeds`, `pi`, `n_sim`, …) they return a
wrapper that routes through the simulation-based `loglikelihood`. Either
way the model carries its own interventions and attributes, so the `~`
line stays clean.

```julia
data ~ chain_size_distribution(BranchingProcess(Poisson(R));
    seeds = seeds, pi = pi)
```
