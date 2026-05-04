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

### Maximum likelihood via Turing

For raw offspring counts EpiBranch does not provide a `fit` wrapper —
the same Turing model used for the posterior also gives the MLE via
`maximum_likelihood`:

```@example inference
@model function offspring_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ Exponential(1.0)
    Turing.@addlogprob! loglikelihood(OffspringCounts(data), NegBin(R, k))
end

mle = maximum_likelihood(offspring_model(data))
println("MLE: R=$(round(mle.values[:R], digits=2)), k=$(round(mle.values[:k], digits=2))")
```

### Bayesian estimation

```@example inference
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

R_offspring = mean(offspring_data)  # Poisson MLE = sample mean
d_chains = fit(Poisson, ChainSizes(size_data))
println("From offspring counts: R=$(round(R_offspring, digits=2))")
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

[`ChainSizes`](@ref) takes optional `seeds` and `concluded` vectors.
They matter in two situations:

- **Multi-seed clusters.** Some clusters start from more than one
  imported case, so the final size is the sum of contributions from
  multiple independent chains.
- **Real-time analysis.** If the data is a time-slice of an evolving
  outbreak (a live situation report, an in-progress study), some
  clusters are still generating new cases at the reporting cutoff.
  For those, the observed size is a lower bound and the likelihood
  needs the right-tail `P(X ≥ x | s)` instead of `P(X = x | s)`.
  Retrospective analyses of closed outbreaks don't need this.

This is the setup in Endo, Abbott, Kucharski & Funk (2020,
[Wellcome Open Research 5:67](https://wellcomeopenresearch.org/articles/5-67)),
where (R₀, k) for SARS-CoV-2 were estimated during the early pandemic
from the WHO 27 Feb 2020 situation report — most European clusters
were still active at that cutoff.

`concluded` is a per-cluster flag; the likelihood takes it at face
value and doesn't care how it was decided. Endo et al. used a 7-day
rule on WHO reports — a cluster was treated as ongoing if any case
had been reported in the last 7 days:

```julia
using Dates
# Derive `concluded` from raw case dates and a reporting cutoff.
concluded_by_window(latest_case_dates, cutoff; window_days = 7) =
    [cutoff - d >= Day(window_days) for d in latest_case_dates]
```

Other analyses might base `concluded` on phylogenetic closure, expert
judgement, negative-test rings, or anything else. The likelihood is
agnostic about the decision rule.

Synthetic data below uses the same kind of mix (some clusters
multi-seed, some ongoing):

```@example inference
true_R, true_k = 0.6, 0.2

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

@model function cluster_size_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    Turing.@addlogprob! loglikelihood(data, NegativeBinomial(k, k / (k + R)))
end

chain = sample(cluster_size_model(data), NUTS(), 1000; progress = false)
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

The likelihood handles the concluded and ongoing clusters in one
call: concluded ones contribute `log P(X = x | s)`, ongoing ones
contribute `log P(X ≥ x | s)`, summed across observations.

## Choosing an inference approach

Two largely independent questions:

1. **Point estimate or full posterior?** Turing supports both
   (`maximum_likelihood`, `maximum_a_posteriori`, or NUTS for the full
   posterior with quantified uncertainty). `fit` is a convenience
   wrapper for MLE only, in the `Distributions.fit` style — closed
   form where possible, score-equation or grid search otherwise.
   Useful for quick exploration or initialising a sampler.
2. **Is the analytical likelihood available?** With no interventions
   and a supported offspring/data combination, the analytical
   likelihood gives fast, exact evaluations. With interventions or
   model features that break the analytical form, the simulation-based
   likelihood takes over — same `loglikelihood` interface, but each
   call runs many simulations, so sampling is markedly slower.

The `loglikelihood` methods are the shared backend: `fit` minimises
`-loglikelihood` over a bracket; Turing models call `loglikelihood`
inside `@addlogprob!`. Switching between MLE, MAP, and posterior is a
question of which Turing entry point you call, not which package.
