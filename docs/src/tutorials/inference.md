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

mle = maximum_likelihood(StableRNG(5), offspring_model(data))
mle_params = NamedTuple(mle.params)
println("MLE: R=$(round(mle_params.R, digits=2)), k=$(round(mle_params.k, digits=2))")
```

### Profile-likelihood intervals

[ProfileLikelihood.jl](https://github.com/SciML/ProfileLikelihood.jl) adds
confidence intervals to the maximum-likelihood estimate. For each parameter in
turn it fixes the parameter at a range of values, maximises the likelihood over
the others, and keeps the values where the log-likelihood lies within 1.92
(half the 95% quantile of a χ² distribution with one degree of freedom) of its
maximum. The objective is EpiBranch's `loglikelihood`. Working with
log R and log k keeps both parameters positive and spreads the profile evenly
over several orders of magnitude of k.

```@example inference
using ProfileLikelihood, OptimizationOptimJL

negbin_loglik(θ, data) = loglikelihood(data, NegBin(exp(θ[1]), exp(θ[2])))

function negbin_problem(data)
    return LikelihoodProblem(
        negbin_loglik, [0.0, 0.0];
        data = data,
        syms = [:log_R, :log_k],
        f_kwargs = (adtype = AutoForwardDiff(),),
        prob_kwargs = (lb = log.([0.01, 0.01]), ub = log.([10.0, 100.0]))
    )
end

function print_intervals(prob)
    # `mle` above holds the Turing estimate, so name the package's function in full
    sol = ProfileLikelihood.mle(prob, Optim.LBFGS())
    prof = profile(prob, sol; confidence_interval_method = :extrema)
    for (name, sym) in ((:R, :log_R), (:k, :log_k))
        ci = get_confidence_intervals(prof, sym)
        println("$name: $(round(exp(sol[sym]), digits = 2)) (95% CI: " *
                "$(round(exp(ci.lower), digits = 2))–$(round(exp(ci.upper), digits = 2)))")
    end
end

print_intervals(negbin_problem(OffspringCounts(data)))
```

Both intervals contain the values used to simulate the data (R = 0.8,
k = 0.5). The same objective works for chain sizes; only the data change:

```@example inference
# 200 chains from NegBin(R=0.6, k=0.2)
chain_sizes = rand(StableRNG(1), chain_size_distribution(NegBin(0.6, 0.2)), 200)
print_intervals(negbin_problem(ChainSizes(chain_sizes)))
```

The upper end of the interval for k often does not exist. As k grows the
negative binomial approaches a Poisson distribution and the likelihood levels
off. With few observations or little overdispersion the profile for k stays
within 1.92 of its maximum all the way up, and the data cannot rule out Poisson
offspring. The upper bound on log k in `ub` then caps the profile: set it where
the offspring distribution is close to Poisson (k = 100 above), and read an
interval whose upper end equals that bound as "k is at least the lower end".
With `confidence_interval_method = :extrema` the profile reports the bound in
this case; the default spline method can return the maximum-likelihood estimate
as the upper end instead.

A parametric bootstrap gives intervals too: simulate many datasets of the same
size from the fitted model with `simulate`, refit each one, and take quantiles
of the refitted estimates. It needs one fit per replicate, and many replicates
put k at the upper bound.

### Bayesian estimation

```@example inference
chain = sample(StableRNG(11), offspring_model(data), NUTS(), 1000; progress=false)
println("Posterior R: $(round(mean(chain[:R]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:R]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:R]), 0.975), digits=2)))")
println("Posterior k: $(round(mean(chain[:k]), digits=2)) " *
        "(95% CI: $(round(quantile(vec(chain[:k]), 0.025), digits=2))–" *
        "$(round(quantile(vec(chain[:k]), 0.975), digits=2)))")
```

### Case-level covariates

The number of secondary cases a case causes often depends on its own
characteristics: the setting of exposure, age, or time of infection. Passing a
vector of distributions, one per observation, evaluates each count against its
own offspring distribution instead of a single shared one:

```@example inference
rng_cov = StableRNG(7)
x = rand(rng_cov, 0:1, 100)  # e.g. household (1) vs community (0) exposure
β0_true, β1_true, k_true = -0.3, 0.8, 0.6
μ_true = exp.(β0_true .+ β1_true .* x)
y = [rand(rng_cov, NegativeBinomial(NegBin(m, k_true).r, NegBin(m, k_true).p)) for m in μ_true]

loglikelihood(OffspringCounts(y), NegBin.(μ_true, k_true))
```

`product_distribution` (from Distributions.jl) turns the same vector of
per-case distributions into a single `Distribution` you can put on the
right-hand side of Turing's `~`, so the covariate coefficients can be
fitted directly:

```@example inference
@model function offspring_covariate_model(x, y)
    β0 ~ Normal(0.0, 2.0)
    β1 ~ Normal(0.0, 2.0)
    k ~ Exponential(1.0)
    y ~ product_distribution(NegBin.(exp.(β0 .+ β1 .* x), k))
end

mle = maximum_likelihood(StableRNG(8), offspring_covariate_model(x, y))
mle_params = NamedTuple(mle.params)
println("MLE: β0=$(round(mle_params.β0, digits=2)), " *
        "β1=$(round(mle_params.β1, digits=2)), k=$(round(mle_params.k, digits=2))")
```

For data that list only cases with at least one secondary case, truncate each
distribution: `loglikelihood(OffspringCounts(y), truncated.(offspring, 1, Inf))`.

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

chain = sample(StableRNG(12), chain_size_model(sizes), NUTS(), 1000; progress=false)
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
iso = Isolation(onset_to_isolation_delay=Exponential(2.0), isolation_duration = 7.0)
clinical = clinical_presentation(incubation_period=LogNormal(1.5, 0.5))
true_model = ModelSpec(BranchingProcess(Poisson(true_R), Exponential(5.0));
    interventions=[iso], attributes=clinical)

observed_states = simulate(true_model, 100;
    max_cases=50, rng=rng)
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
        max_cases = 50,
        n_sim = 100, rng = StableRNG(hash(R)))
end

chain = sample(
    StableRNG(13), intervention_model(observed_sizes, iso, clinical),
    MH(), 1000; progress=false
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
    p = k / (k + R)
    # Reject proposals whose success probability saturates at a numerical boundary.
    if !isfinite(log(p)) || !isfinite(log1p(-p))
        Turing.@addlogprob! -Inf
        return
    end
    sizes ~ chain_size_distribution(BranchingProcess(NegativeBinomial(k, p)); seeds = seeds)
end

chain = sample(StableRNG(14), cluster_size_model(sizes, seeds), NUTS(), 1000; progress = false)
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

1. **Point estimate or full posterior?** For a maximum-likelihood point
   estimate, maximise `loglikelihood` over the parameter — with Turing's
   `maximum_likelihood`, with Optim.jl, or, for a single parameter, over a
   grid (as in the [chains tutorial](chains.md)). For a full posterior with
   quantified uncertainty, put the data on the right-hand side of `~` through
   the distribution wrappers and sample with NUTS; `maximum_a_posteriori`
   gives the MAP point.
2. **Is the analytical likelihood available?** With no interventions
   and a supported offspring/data combination, the analytical
   likelihood gives fast, exact evaluations. With interventions or
   model features that break the analytical form, the simulation-based
   likelihood takes over — same `loglikelihood` interface, but each
   call runs many simulations, so sampling is markedly slower.

The `loglikelihood` methods are the shared backend throughout: an optimiser
maximises them directly, and Turing models route through the distribution
wrappers, which call the same methods. Switching between MLE, MAP, and
posterior is a question of which entry point you call, not which package.

## Live diagnostics during long fits

`sample(...)` accepts a `callback=` kwarg (from AbstractMCMC) that
runs once per chain per step. Use it with
[TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl)
or any logger to stream per-iteration diagnostics while the fit runs.
