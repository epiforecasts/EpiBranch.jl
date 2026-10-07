# Inference

What can you learn about R and the dispersion k from outbreak data? This page
estimates them from three kinds of data:

| Data | What you observe | Typical source | Estimates |
|:-----|:-----------------|:---------------|:----------|
| Offspring counts ([`OffspringCounts`](@ref)) | the number of secondary cases each case caused | contact tracing that links infectors to infectees | R and k |
| Chain sizes ([`ChainSizes`](@ref)) | the final number of cases in each cluster, but not who infected whom | cluster investigations, imported cases and their clusters | R, and k given enough clusters |
| Chain lengths ([`ChainLengths`](@ref)) | the number of generations in each cluster | generation-linked tracing data | R |

`NegBin(R, k)` is the negative binomial offspring distribution with mean R and
dispersion k; smaller k means more superspreading.

How the likelihood is calculated depends on the model:

- No interventions: for Poisson and negative binomial offspring the
  likelihood has an exact formula. It is fast, and any fitting method works.
  Chain sizes assume each cluster has finished growing, which in practice
  means R < 1 or clusters observed long after they ended.
- Interventions in the model: if cases are isolated or contacts traced,
  observed chains are shorter than the offspring distribution alone implies,
  and the chain-size likelihood has to be estimated by simulating the whole
  model. This is slower, the estimate is noisy, and Bayesian fitting needs a
  sampler that copes with that (see [Inference under interventions](@ref)).

You can fit by maximum likelihood, which gives a point estimate with a
**confidence interval** from the profile likelihood (much as you would use
`optim()` and profiling in R), or by Bayesian inference, which gives a posterior
distribution with **credible intervals** (as with Stan or brms). Both use the
same likelihood, so the estimates can be compared directly.

!!! note
    The examples use [Turing.jl](https://turinglang.org) for Bayesian
    inference and maximum likelihood, and
    [ProfileLikelihood.jl](https://github.com/SciML/ProfileLikelihood.jl) for
    confidence intervals. Neither is installed with EpiBranch; add them with
    `using Pkg; Pkg.add(["Turing", "ProfileLikelihood", "OptimizationOptimJL"])`.

```@example inference
using EpiBranch
using Distributions
using Turing
using StableRNGs
```

## From offspring counts

The simplest case: you observe how many secondary cases each case caused. The
data below are 50 such counts drawn from a negative binomial with R = 0.8 and
k = 0.5. Fifty cases is few for estimating k, as the intervals below show.

```@example inference
rng = StableRNG(42)  # a fixed seed makes the results reproducible
true_R, true_k = 0.8, 0.5
data = rand(rng, NegBin(true_R, true_k), 50)
println("Observed offspring counts: mean=$(round(mean(data), digits=2)), var=$(round(var(data), digits=2))")
```

The variance is well above the mean, the sign of overdispersion.

### Writing the model in Turing

A Turing model is a Julia function marked with `@model`. Each line with `~`
reads "is distributed as". A line with a parameter on the left gives its
prior; a line with the data on the left gives the likelihood, like a sampling
statement in Stan:

```@example inference
@model function offspring_model(data)
    R ~ LogNormal(0.0, 1.0)  # prior: R log-normal with median 1
    k ~ Exponential(1.0)     # prior: k exponential with mean 1
    # likelihood: each count follows the offspring distribution NegBin(R, k)
    data ~ offspring_distribution(BranchingProcess(NegBin(R, k)))
end
```

[`offspring_distribution`](@ref) gives the distribution of secondary cases per
case under a model; [`chain_size_distribution`](@ref) and
[`chain_length_distribution`](@ref) do the same for chain sizes and lengths, and
are used in the same place below.

!!! warning "Offspring counts and interventions"
    `offspring_distribution` describes secondary cases without interventions.
    If you give it a model that includes interventions, they are ignored. If
    your offspring counts come from a period with isolation or quarantine in
    place, the R you estimate is the R under those interventions, not the R
    without them (see
    [issue #421](https://github.com/epiforecasts/epiBranch.jl/issues/421)).

### Maximum likelihood

`maximum_likelihood` ignores the priors and finds the values of R and k that
make the data most likely, as `optim()` would in R:

```@example inference
fit_ml = maximum_likelihood(StableRNG(5), offspring_model(data))
est = NamedTuple(fit_ml.params)  # the estimates, by parameter name
println("Maximum-likelihood estimate: R=$(round(est.R, digits=2)), k=$(round(est.k, digits=2))")
```

### Confidence intervals from the profile likelihood

[ProfileLikelihood.jl](https://github.com/SciML/ProfileLikelihood.jl) gives
profile-likelihood confidence intervals. For each parameter in turn it fixes
the parameter at a range of values, maximises the likelihood over the others,
and keeps the values where the log-likelihood is within 1.92 of its maximum
(1.92 is half the 95% quantile of a χ² distribution with one degree of
freedom). It needs the log-likelihood as a function of a parameter vector;
here that function calls EpiBranch's `loglikelihood` directly. Working with
log R and log k keeps both parameters positive and spreads the profile evenly
over several orders of magnitude of k.

```@example inference
using ProfileLikelihood, OptimizationOptimJL

# The log-likelihood, with θ = [log R, log k] (Julia counts from 1, as R does)
negbin_loglik(θ, data) = loglikelihood(data, NegBin(exp(θ[1]), exp(θ[2])))

function negbin_problem(data)
    return LikelihoodProblem(
        negbin_loglik, [0.0, 0.0];  # start from log R = log k = 0, i.e. R = k = 1
        data = data,
        syms = [:log_R, :log_k],    # names for the two parameters
        # the optimiser needs the slope of the log-likelihood; compute it automatically
        f_kwargs = (adtype = AutoForwardDiff(),),
        # search R between 0.01 and 10 and k between 0.01 and 100
        prob_kwargs = (lb = log.([0.01, 0.01]), ub = log.([10.0, 100.0]))
    )
end

function print_intervals(prob)
    sol = mle(prob, Optim.LBFGS())  # maximise, like optim(method = "L-BFGS-B")
    prof = profile(prob, sol; confidence_interval_method = :extrema)
    for (name, sym) in ((:R, :log_R), (:k, :log_k))
        ci = get_confidence_intervals(prof, sym)
        # back-transform from the log scale
        println("$name: $(round(exp(sol[sym]), digits = 2)) (95% confidence interval " *
                "$(round(exp(ci.lower), digits = 2))–$(round(exp(ci.upper), digits = 2)))")
    end
end

print_intervals(negbin_problem(OffspringCounts(data)))
```

Both intervals should contain the values used to simulate the data (R = 0.8,
k = 0.5). The same function works for chain sizes; only the data change:

```@example inference
# 200 chains from NegBin(R=0.6, k=0.2)
chain_sizes = rand(StableRNG(1), chain_size_distribution(NegBin(0.6, 0.2)), 200)
print_intervals(negbin_problem(ChainSizes(chain_sizes)))
```

!!! note "k often has no upper limit"
    With limited data, the confidence interval for k is often not bounded
    above: the data cannot rule out little or no superspreading. As k grows the
    negative binomial approaches a Poisson distribution and the likelihood
    levels off. The profile for k then stays within 1.92 of its maximum all the
    way up. The upper bound on k given to `LikelihoodProblem` (k = 100 above,
    where the offspring distribution is close to Poisson) then caps the
    interval. Read an interval whose upper end equals that bound as "k is at
    least the lower end". With `confidence_interval_method = :extrema` the
    profile reports the bound in this case; the default method can wrongly
    report the maximum-likelihood estimate as the upper end instead.

A parametric bootstrap gives intervals too: simulate many datasets of the same
size from the fitted model with `simulate`, refit each one, and take quantiles
of the estimates. It needs one fit per dataset, and with few data many of the
refitted estimates of k sit at the upper bound.

### Bayesian estimation

`sample` draws from the posterior distribution. `NUTS()` is the No-U-Turn
Sampler, the same algorithm Stan uses. The call below runs four chains of 1,000
draws each, one after another (`MCMCSerial()`):

```@example inference
posterior = sample(StableRNG(11), offspring_model(data), NUTS(), MCMCSerial(), 1000, 4;
    progress = false)

# Posterior median and 95% credible interval of one parameter
function summarise_posterior(posterior, name)
    draws = vec(posterior[name])
    lower, upper = quantile(draws, [0.025, 0.975])
    println("$name: $(round(median(draws), digits = 2)) (95% credible interval " *
            "$(round(lower, digits = 2))–$(round(upper, digits = 2)))")
end

summarise_posterior(posterior, :R)
summarise_posterior(posterior, :k)
```

Before using the estimates, check that the sampler has converged. The summary
table reports, for each parameter, `rhat` (R-hat, which should be close to 1,
say below 1.01, when the chains agree) and `ess_bulk` and `ess_tail` (effective
sample sizes, which should be in the hundreds at least):

```@example inference
using Turing.FlexiChains: summarystats
summarystats(posterior)
```

`maximum_a_posteriori` gives the posterior mode in the same way that
`maximum_likelihood` gives the maximum-likelihood estimate.

### Case-level covariates

The number of secondary cases a case causes often depends on its own
characteristics: the setting of exposure, age, or time of infection. The data
below have one binary covariate `x` (household or community exposure), and
each case's R depends on it through a log link,
log(R) = β0 + β1 × x. The dot in `exp.(...)`, `.+` and `.*` applies the
operation to every element, as R's vectorised arithmetic does.

```@example inference
rng_cov = StableRNG(7)
x = rand(rng_cov, 0:1, 100)  # household (1) or community (0) exposure
β0_true, β1_true, k_true = -0.3, 0.8, 0.6
R_true = exp.(β0_true .+ β1_true .* x)  # each case's R
y = [rand(rng_cov, NegBin(R, k_true)) for R in R_true]  # one count per case
```

Passing a vector of offspring distributions, one per case, evaluates each count
against its own distribution:

```@example inference
ll = loglikelihood(OffspringCounts(y), NegBin.(R_true, k_true))
println("Log-likelihood at the true values: $(round(ll, digits = 2))")
```

To fit β0, β1 and k, combine the per-case distributions with
`product_distribution` (from Distributions.jl) so that one `~` line covers all
the counts:

```@example inference
@model function offspring_covariate_model(x, y)
    β0 ~ Normal(0.0, 2.0)  # log R in community exposure
    β1 ~ Normal(0.0, 2.0)  # log ratio of R, household vs community
    k ~ Exponential(1.0)
    y ~ product_distribution(NegBin.(exp.(β0 .+ β1 .* x), k))
end

fit_cov = maximum_likelihood(StableRNG(8), offspring_covariate_model(x, y))
est_cov = NamedTuple(fit_cov.params)
println("Maximum-likelihood estimate: β0=$(round(est_cov.β0, digits=2)), " *
        "β1=$(round(est_cov.β1, digits=2)), k=$(round(est_cov.k, digits=2))")
println("R ratio, household vs community: $(round(exp(est_cov.β1), digits = 2)) " *
        "(true $(round(exp(β1_true), digits = 2)))")
```

`exp(β1)` is the ratio of the mean number of secondary cases after household
exposure to that after community exposure.

### Data that only include cases who infected someone

Some contact-tracing data list only cases with at least one secondary case.
Fitting these as if they were all cases would overestimate R. Truncate each
offspring distribution at 1 so that the likelihood allows for the missing
zeros:

```@example inference
spreaders = y .> 0  # cases with at least one secondary case
truncated_offspring = truncated.(NegBin.(R_true[spreaders], k_true), 1, Inf)
ll = loglikelihood(OffspringCounts(y[spreaders]), truncated_offspring)
println("Log-likelihood of the spreaders only: $(round(ll, digits = 2))")
```

## From chain sizes

When you observe the final size of each cluster but not who infected whom:

```@example inference
# Simulate 200 chains from a Poisson process with R = 0.7
rng = StableRNG(42)
true_R = 0.7
model = BranchingProcess(Poisson(true_R))
states = simulate(model, 200; rng = rng)
sizes = chain_statistics(states).size
println("Observed $(length(sizes)) chain sizes, mean=$(round(mean(sizes), digits=2))")
```

### Maximum likelihood

With one parameter, a grid is enough: compute the log-likelihood over a range
of R and take the highest. The same grid traces the likelihood profile: the
values of R within 1.92 of the maximum give a 95% confidence interval.

```@example inference
data = ChainSizes(sizes)
R_grid = 0.05:0.01:0.95  # R from 0.05 to 0.95 in steps of 0.01
ll_grid = [loglikelihood(data, Poisson(R)) for R in R_grid]
R_mle = R_grid[argmax(ll_grid)]
R_ci = extrema(R_grid[ll_grid .>= maximum(ll_grid) - 1.92])
println("Maximum-likelihood estimate: R=$(round(R_mle, digits=2)) " *
        "(95% confidence interval $(round(R_ci[1], digits=2))–$(round(R_ci[2], digits=2)))")
```

### Bayesian estimation

```@example inference
@model function chain_size_model(data)
    R ~ Beta(2, 2)  # prior on (0, 1): assumes R < 1
    data ~ chain_size_distribution(BranchingProcess(Poisson(R)))
end

posterior = sample(StableRNG(12), chain_size_model(sizes), NUTS(), 1000; progress=false)
println("True R = $true_R")
summarise_posterior(posterior, :R)
```

The `Beta(2, 2)` prior only allows R < 1. That suits clusters known to have
died out by themselves, but it forces the estimate below 1. If R might be above
1, use a prior that allows it, and account for clusters that may still be
growing with `prob_concluded` (the probability that each cluster has finished),
passed to `chain_size_distribution`.

## Comparing data types

Offspring counts and chain sizes from the same process should give similar
estimates of R:

```@example inference
rng = StableRNG(42)
true_R = 0.6

# Offspring counts
offspring_data = rand(rng, Poisson(true_R), 100)

# Chain sizes
model = BranchingProcess(Poisson(true_R))
states = simulate(model, 200; rng=StableRNG(99))
size_data = ChainSizes(chain_statistics(states).size)

R_offspring = mean(offspring_data)  # for Poisson counts the MLE is the sample mean
R_grid = 0.05:0.01:0.95
R_chains = R_grid[argmax([loglikelihood(size_data, Poisson(R)) for R in R_grid])]
println("From offspring counts: R=$(round(R_offspring, digits=2))")
println("From chain sizes:     R=$(round(R_chains, digits=2))")
println("True:                 R=$true_R")
```

The two estimates differ from each other and from the true value through
sampling variation alone; with more data both would move closer to it. Because the log-likelihoods of
independent datasets add up, you can also fit both together by adding the two
`loglikelihood` values, or by putting both `~` lines in one Turing model.

## Inference under interventions

If the outbreaks you observed were under isolation, fitting the offspring
distribution alone would underestimate R, because isolation ends chains early.
Put the isolation in the model with a [`ModelSpec`](@ref), and EpiBranch
estimates the chain-size likelihood by simulating that model.

The data below are 100 chains simulated under isolation with R = 2.0. Each
case is isolated after an exponentially distributed delay from symptom onset
(mean 2 days) for 7 days (`duration`), after which they can transmit again;
the incubation period is log-normal with a median of about 4.5 days, and the
generation time exponential with a mean of 5 days. Each simulated outbreak
stops at 50 cases.

```@example inference
rng = StableRNG(42)
true_R = 2.0
iso = Isolation(onset_to_isolation_delay=Exponential(2.0), duration = 7.0)
clinical = clinical_presentation(incubation_period=LogNormal(1.5, 0.5))
true_model = ModelSpec(BranchingProcess(Poisson(true_R), Exponential(5.0));
    interventions=[iso], attributes=clinical)

observed_states = simulate(true_model, 100; max_cases=50, rng=rng)
observed_sizes = chain_statistics(observed_states).size
println("Observed $(length(observed_sizes)) chain sizes under isolation")
println("Mean size: $(round(mean(observed_sizes), digits=1)), " *
        "reaching the 50-case cap: $(count(>=(50), observed_sizes))")
```

With R = 2, isolation stops some chains early, but many still reach the cap.

The model below estimates R from these sizes. Outbreaks still growing when they
reach the cap are counted as "at least 50 cases" rather than exactly 50.
This way the cap does not bias the estimate.

```@example inference
@model function intervention_model(data, iso, clinical)
    R ~ LogNormal(0.5, 0.5)
    model = ModelSpec(BranchingProcess(Poisson(R), Exponential(5.0));
        interventions = [iso], attributes = clinical)
    # n_sim: simulated chains per likelihood estimate. A fixed seed reuses the
    # same random numbers for every value of R tried, so differences in the
    # likelihood come from R and not from simulation noise.
    data ~ chain_size_distribution(model;
        max_cases = 50, n_sim = 100, rng = StableRNG(1))
end

posterior = sample(
    StableRNG(13), intervention_model(observed_sizes, iso, clinical),
    MH(), 1000; progress=false
)
println("True R = $true_R")
summarise_posterior(posterior, :R)
```

In this run the 95% credible interval contains the true R, despite the
isolation and the cap on outbreak size.

!!! note "Choosing a sampler for simulated likelihoods"
    A likelihood estimated by simulation is noisy. NUTS follows the slope of the
    likelihood and cannot be used. `MH()` (Metropolis-Hastings) needs
    only likelihood values. It is slower and needs more draws, so check
    convergence as above, and increase `n_sim` if the estimates vary between
    runs. Each likelihood evaluation runs `n_sim` simulations, and fitting takes
    much longer than with an exact formula.

## Clusters started by more than one case

Some clusters start from more than one introduced case: for example, two
travellers arriving together. Passing `seeds`, the number of index cases in
each cluster, to `chain_size_distribution` accounts for this (or to
[`ChainSizes`](@ref) when you call `loglikelihood` directly). All clusters are
assumed to have finished, and the exact formula still applies.

```@example inference
true_R, true_k = 0.6, 0.2

rng = StableRNG(7)
n = 50
seeds = rand(rng, [1, 1, 1, 2], n)  # a quarter of clusters start from two cases
size_per_index_case = chain_size_distribution(NegBin(true_R, true_k))
# each cluster's size is the sum of the chains started by each of its index cases
sizes = [sum(rand(rng, size_per_index_case) for _ in 1:s) for s in seeds]

println("Clusters: $(length(sizes)) (one / two index cases: " *
        "$(count(==(1), seeds)) / $(count(==(2), seeds)))")

@model function cluster_size_model(sizes, seeds)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    sizes ~ chain_size_distribution(BranchingProcess(NegBin(R, k)); seeds = seeds)
end

posterior = sample(StableRNG(14), cluster_size_model(sizes, seeds), NUTS(), 1000; progress = false)
println("True R=$true_R, k=$true_k")
summarise_posterior(posterior, :R)
summarise_posterior(posterior, :k)
```

With 50 clusters the credible interval for k is wide; chain sizes hold less
information about superspreading than offspring counts do.

## Watching long fits

For fits that take a long time, `sample` accepts a `callback` function that
runs after every step, which you can use to log progress, for example with
[TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl).
