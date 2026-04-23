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

## Inference under interventions

When you pass a `BranchingProcess` model with interventions to
`loglikelihood`, it uses simulation-based likelihood. Because this is
stochastic and not differentiable, you have to use a gradient-free
sampler like `MH()` instead of `NUTS()`:

```@example inference
# Generate "observed" chain sizes from a model WITH isolation
rng = StableRNG(42)
true_R = 2.0
true_model = BranchingProcess(Poisson(true_R), Exponential(5.0))
iso = Isolation(delay=Exponential(2.0))
clinical = clinical_presentation(incubation_period=LogNormal(1.5, 0.5))

observed_states = simulate_batch(true_model, 100;
    interventions=[iso], attributes=clinical,
    sim_opts=SimOpts(max_cases=500), rng=rng)
observed_sizes = Int[]
for s in observed_states
    cs = chain_statistics(s)
    append!(observed_sizes, cs.size)
end
println("Observed $(length(observed_sizes)) chain sizes under isolation")
println("Mean size: $(round(mean(observed_sizes), digits=1))")
```

Now estimate R from the observed data, accounting for the intervention.
The simulation-based likelihood automatically handles right-censoring:
simulations that hit the case cap contribute P(size >= cap) instead of
P(size = cap).

```@example inference
@model function intervention_model(data, iso, clinical)
    R ~ LogNormal(0.5, 0.5)
    model = BranchingProcess(Poisson(R), Exponential(5.0))
    Turing.@addlogprob! loglikelihood(
        ChainSizes(data), model;
        interventions=[iso], attributes=clinical,
        sim_opts=SimOpts(max_cases=500),
        n_sim=500, rng=StableRNG(hash(R))
    )
end

chain = sample(
    intervention_model(observed_sizes, iso, clinical),
    MH(), 2000; progress=false
)
println("True R = $true_R")
println("Posterior R: $(round(mean(chain[:R]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:R]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:R]), 0.975), digits=2)))")
```

The posterior recovers the true R despite the intervention and the
case cap truncating large outbreaks.

## Multi-seed and ongoing outbreaks

[`ChainSizes`](@ref) takes optional `seeds` and `concluded` vectors for
inference in the style of Endo, Abbott, Kucharski & Funk (2020,
[Wellcome Open Research 5:67](https://wellcomeopenresearch.org/articles/5-67)):
some clusters start from more than one imported case, and some are
still ongoing so the observed size is only a lower bound.

```@example inference
# Synthetic data: mixed single- and two-seed imports, some ongoing.
true_R, true_k = 0.6, 0.2
d_true = GammaBorel(true_k, true_R)

rng = StableRNG(7)
n = 50
seeds = rand(rng, [1, 1, 1, 2], n)
sizes = Int[]
concluded = Bool[]
for s in seeds
    x = sum(rand(rng, GammaBorel(true_k, true_R)) for _ in 1:s)
    # Mark about a fifth of clusters as still ongoing; pick a lower
    # bound below the true final size.
    ongoing = rand(rng) < 0.2
    push!(sizes, ongoing ? max(s, x - rand(rng, 0:max(x - s, 1))) : x)
    push!(concluded, !ongoing)
end

data = ChainSizes(sizes; seeds = seeds, concluded = concluded)
println("Clusters: $(length(sizes)) (seeds 1 / 2: $(count(==(1), seeds)) / $(count(==(2), seeds)))")
println("Ongoing:  $(count(==(false), concluded))")

@model function endo_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    Turing.@addlogprob! loglikelihood(data, NegativeBinomial(k, k / (k + R)))
end

chain = sample(endo_model(data), NUTS(), 1000; progress = false)
r_post = vec(chain[:R])
k_post = vec(chain[:k])
println("True R=$true_R, k=$true_k")
println("R: $(round(mean(r_post), digits=2)) (95% CI: " *
        "$(round(quantile(r_post, 0.025), digits=2))–" *
        "$(round(quantile(r_post, 0.975), digits=2)))")
println("k: $(round(mean(k_post), digits=2)) (95% CI: " *
        "$(round(quantile(k_post, 0.025), digits=2))–" *
        "$(round(quantile(k_post, 0.975), digits=2)))")
```

The likelihood handles the concluded and ongoing clusters through a
single call: concluded ones contribute `log P(X = x | s)`, ongoing
ones contribute `log P(X ≥ x | s)`, summed across observations.

## When to use `fit` vs Turing

- **`fit`**: MLE point estimate. Use for quick exploration or initial
  parameter guesses.
- **Turing + analytical likelihood**: full posterior with NUTS (when
  the analytical likelihood is available).
- **Turing + simulation-based likelihood**: when interventions or
  complex model features make analytical likelihood unavailable. Slower
  (many simulations per likelihood evaluation) but handles models that
  have no closed-form likelihood.
