# Inference

[`chain_size_distribution`](@ref), [`chain_length_distribution`](@ref),
and [`offspring_distribution`](@ref) turn a model into a
`Distribution` you can put on the right-hand side of
[Turing.jl](https://turinglang.org)'s `~`. With no extra arguments
they return the analytical form (`Borel`, `GammaBorel`, the bare
offspring `Distribution`) where one exists; with `seeds`, `pi`, a
[`ModelSpec`](@ref) composing interventions onto the process, or other
kwargs they return a wrapper that routes through the same
`loglikelihood` methods used for MLE. A wrapper preserves AD only when
the likelihood it routes through is itself differentiable, so NUTS works
for the analytical and closed-form paths but not where a wrapper routes
through a simulation-based intervention likelihood.

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
    data ~ offspring_distribution(BranchingProcess(NegBin(R, k)))
end

mle = maximum_likelihood(offspring_model(data))
mle_params = NamedTuple(mle.params)
println("MLE: R=$(round(mle_params.R, digits=2)), k=$(round(mle_params.k, digits=2))")
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
states = simulate(model, 200; rng=rng)
sizes = Int[]
for s in states
    cs = chain_statistics(s)
    append!(sizes, cs.size)
end
println("Observed $(length(sizes)) chain sizes, mean=$(round(mean(sizes), digits=2))")
```

### Maximum likelihood

Maximise `loglikelihood` over the parameter — here a one-parameter grid;
for harder problems use Optim.jl or Turing's `maximum_likelihood`:

```@example inference
data = ChainSizes(sizes)
Rgrid = 0.05:0.01:0.95
R_mle = Rgrid[argmax([loglikelihood(data, Poisson(R)) for R in Rgrid])]
println("MLE: R=$(round(R_mle, digits=2))")
```

### Bayesian estimation

```@example inference
@model function chain_size_model(data)
    R ~ Beta(2, 2)  # prior on (0, 1) for subcritical
    data ~ chain_size_distribution(BranchingProcess(Poisson(R)))
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
states = simulate(model, 200; rng=StableRNG(99))
size_data = Int[]
for s in states
    cs = chain_statistics(s)
    append!(size_data, cs.size)
end

R_offspring = mean(offspring_data)  # Poisson MLE = sample mean
size_d = ChainSizes(size_data)
Rgrid = 0.05:0.01:0.95
R_chains = Rgrid[argmax([loglikelihood(size_d, Poisson(R)) for R in Rgrid])]
println("From offspring counts: R=$(round(R_offspring, digits=2))")
println("From chain sizes:     R=$(round(R_chains, digits=2))")
println("True:                 R=$true_R")
```

## Inference under interventions

When a [`ModelSpec`](@ref) composes interventions onto the process,
`loglikelihood` uses the simulation-based likelihood. Because this is
stochastic and not
differentiable, you have to use a gradient-free sampler like `MH()`
instead of `NUTS()`:

```@example inference
# Generate "observed" chain sizes from a model WITH isolation
rng = StableRNG(42)
true_R = 2.0
iso = Isolation(onset_to_isolation_delay=Exponential(2.0))
clinical = clinical_presentation(incubation_period=LogNormal(1.5, 0.5))
true_model = ModelSpec(BranchingProcess(Poisson(true_R), Exponential(5.0));
    interventions=[iso], attributes=clinical)

observed_states = simulate(true_model, 100;
    max_cases=500, rng=rng)
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
    model = ModelSpec(BranchingProcess(Poisson(R), Exponential(5.0));
        interventions = [iso], attributes = clinical)
    data ~ chain_size_distribution(model;
        max_cases = 500,
        n_sim = 500, rng = StableRNG(hash(R)))
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

## Multi-seed clusters

[`ChainSizes`](@ref) takes an optional `seeds` vector for clusters
with multiple independent index cases. All clusters are treated as
concluded; the analytical multi-seed chain-size PMF handles them in
one call.

```@example inference
true_R, true_k = 0.6, 0.2

rng = StableRNG(7)
n = 50
seeds = rand(rng, [1, 1, 1, 2], n)
cluster_law = chain_size_distribution(NegBin(true_R, true_k))
sizes = [sum(rand(rng, cluster_law) for _ in 1:s) for s in seeds]

data = ChainSizes(sizes; seeds = seeds)
println("Clusters: $(length(sizes)) (seeds 1 / 2: " *
        "$(count(==(1), seeds)) / $(count(==(2), seeds)))")

@model function cluster_size_model(sizes, seeds)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    sizes ~ chain_size_distribution(BranchingProcess(NegativeBinomial(k, k / (k + R))); seeds = seeds)
end

chain = sample(cluster_size_model(sizes, seeds), NUTS(), 1000; progress = false)
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
`-loglikelihood` over a bracket; Turing models route through the
distribution wrappers, which call the same methods. Switching between
MLE, MAP, and posterior is a question of which Turing entry point you
call, not which package.

## Live diagnostics during long fits

`sample(...)` accepts a `callback=` kwarg (from AbstractMCMC) that
runs once per chain per step. Use it with
[TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl)
or any logger to stream per-iteration diagnostics while the fit runs.
