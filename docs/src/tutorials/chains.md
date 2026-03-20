# Chain statistics and likelihood

Compute chain size and length statistics, and evaluate likelihoods for
offspring distribution estimation — analytically where possible,
simulation-based otherwise.

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

## Analytical likelihood

Given observed chain sizes, evaluate the log-likelihood:

```@example chains
data = [1, 1, 2, 1, 3, 1, 1, 5, 1, 2]

# Compare offspring parameters
for λ in [0.3, 0.5, 0.7, 0.9, 1.1]
    ll = chain_size_ll(data, Poisson(λ))
    println("Poisson($λ): LL = $(round(ll, digits=2))")
end
```

With Negative Binomial offspring:

```@example chains
ll = chain_size_ll(data, NegBin(0.9, 0.5))
println("NegBin(R=0.9, k=0.5): LL = $(round(ll, digits=2))")
```

### Imperfect observation

Account for incomplete case ascertainment:

```@example chains
ll_full = chain_size_ll(data, Poisson(0.9))
ll_partial = chain_size_ll(data, Poisson(0.9), 0.7)
println("Full observation:  $(round(ll_full, digits=2))")
println("70% observation:   $(round(ll_partial, digits=2))")
```

## Simulation-based likelihood

For offspring distributions without analytical chain size distributions,
or for models with interventions, use simulation-based likelihood:

```@example chains
# Any model works — including with interventions
model = BranchingProcess(Poisson(2.0), Exponential(5.0))
iso = Isolation(delay = Exponential(2.0))

ll = chain_size_ll(data, model;
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    n_sim = 1000,
    rng = StableRNG(42),
)
println("LL with interventions: $(round(ll, digits=2))")
```

Forward simulation and likelihood evaluation share the same code path.
Likelihoods can therefore be evaluated under interventions.

## Chain length likelihood

```@example chains
lengths = [0, 1, 0, 2, 1, 0, 0, 3, 0, 1]
ll = chain_length_ll(lengths, Poisson(0.5))
println("Chain length LL: $(round(ll, digits=2))")
```
