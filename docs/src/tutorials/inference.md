# Inference

EpiBranch.jl's `loglikelihood` functions are compatible with automatic
differentiation, so they work directly with [Turing.jl](https://turinglang.org)
for Bayesian inference. This tutorial shows end-to-end parameter estimation
from different data types.

!!! note
    Turing.jl is not a dependency of EpiBranch.jl. Install it separately
    with `Pkg.add("Turing")`.

```@example inference
using EpiBranch
using Distributions
using Turing
using StableRNGs
```

## From offspring counts

The simplest case: you observe how many secondary cases each case caused.

```@example inference
# Generate synthetic data: 50 observations from NegBin(R=0.8, k=0.5)
rng = StableRNG(42)
true_R, true_k = 0.8, 0.5
d_true = NegBin(true_R, true_k)
data = rand(rng, NegativeBinomial(d_true.r, d_true.p), 50)
println("Observed offspring counts: mean=$(round(mean(data), digits=2)), var=$(round(var(data), digits=2))")
```

### Maximum likelihood

```@example inference
d_mle = fit(NegativeBinomial, OffspringCounts(data))
println("MLE: R=$(round(mean(d_mle), digits=2)), k=$(round(d_mle.r, digits=2))")
```

### Bayesian estimation

```@example inference
@model function offspring_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ Exponential(1.0)
    Turing.@addlogprob! loglikelihood(OffspringCounts(data), NegBin(R, k))
end

chain = sample(offspring_model(data), NUTS(), 1000; progress=false)
println("Posterior R: $(round(mean(chain[:R]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:R]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:R]), 0.975), digits=2)))")
println("Posterior k: $(round(mean(chain[:k]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:k]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:k]), 0.975), digits=2)))")
```

## From chain sizes

When you observe final outbreak sizes but not who-infected-whom:

```@example inference
# Simulate chain sizes from a subcritical Poisson(0.7) process
rng = StableRNG(42)
true_R = 0.7
model = BranchingProcess(Poisson(true_R))
states = simulate_batch(model, 200; rng=rng)
sizes = Int[]
for s in states
    cs = chain_statistics(s)
    append!(sizes, cs.size)
end
println("Observed $(length(sizes)) chain sizes, mean=$(round(mean(sizes), digits=2))")
```

### Maximum likelihood

```@example inference
d_mle = fit(Poisson, ChainSizes(sizes))
println("MLE: R=$(round(mean(d_mle), digits=2))")
```

### Bayesian estimation

```@example inference
@model function chain_size_model(data)
    R ~ Beta(2, 2)  # prior on (0, 1) for subcritical
    Turing.@addlogprob! loglikelihood(ChainSizes(data), Poisson(R))
end

chain = sample(chain_size_model(sizes), NUTS(), 1000; progress=false)
println("True R = $true_R")
println("Posterior R: $(round(mean(chain[:R]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:R]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:R]), 0.975), digits=2)))")
```

## Comparing data types

The same `loglikelihood` interface works regardless of data type. This
makes it easy to combine different data sources in a single model or
compare estimates from different observation processes:

```@example inference
# Same underlying R, different observation processes
rng = StableRNG(42)
true_R = 0.6

# Direct offspring observations
offspring_data = rand(rng, Poisson(true_R), 100)

# Chain size observations
model = BranchingProcess(Poisson(true_R))
states = simulate_batch(model, 200; rng=StableRNG(99))
size_data = Int[]
for s in states
    cs = chain_statistics(s)
    append!(size_data, cs.size)
end

d_offspring = fit(Poisson, OffspringCounts(offspring_data))
d_chains = fit(Poisson, ChainSizes(size_data))
println("From offspring counts: R=$(round(mean(d_offspring), digits=2))")
println("From chain sizes:     R=$(round(mean(d_chains), digits=2))")
println("True:                 R=$true_R")
```
