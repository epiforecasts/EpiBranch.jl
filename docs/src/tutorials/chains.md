# Chain statistics, likelihood, and fitting

How large do transmission chains get, and what do observed chain sizes say
about R? A transmission chain (or cluster) is the set of cases that descend from
one index case. This page computes chain sizes and lengths from simulations,
evaluates the likelihood of observed chain sizes, offspring counts and chain
lengths, and fits R to them. Exact formulae are used where they exist;
otherwise EpiBranch simulates the model. The [inference tutorial](inference.md)
covers estimation in more depth.

## Chain statistics

The model below has Poisson offspring with R = 0.9 (each case causes on average
0.9 secondary cases) and an exponential generation time with a mean of 5 days.
The simulation starts from 20 index cases and stops if it reaches 10,000 cases
(`10_000` is Julia's way of writing 10000 readably).

```@example chains
using EpiBranch
using Distributions
using DataFrames
using StableRNGs

model = BranchingProcess(Poisson(0.9), Exponential(5.0))

rng = StableRNG(42)  # a fixed seed makes the results reproducible
state = simulate(model; n_initial = 20, max_cases = 10_000, rng = rng)

cs = chain_statistics(state)
first(cs, 10)
```

Each row is one chain. `size` is the number of cases in the chain, including
the index case. `length` is the number of generations of onward transmission.
An index case that infects nobody has size 1 and length 0 (the R package
epichains counts this as length 1).

```@example chains
println("Mean size: $(round(mean(cs.size), digits=2)), Max: $(maximum(cs.size))")
println("Mean length: $(round(mean(cs.length), digits=2))")
```

With R below 1 every chain dies out. Most stay small, but a few grow much
larger than the mean.

## Exact chain size distributions

When R < 1 the distribution of final chain sizes is known exactly for Poisson
offspring (the Borel distribution) and for negative binomial offspring (the
gamma-Borel distribution). `NegBin(R, k)` is a negative binomial with mean R and
dispersion k; smaller k means more superspreading.

```@example chains
# Poisson offspring with R = 0.8: Borel chain sizes
d_borel = chain_size_distribution(Poisson(0.8))
println("Poisson(R = 0.8), mean chain size: $(round(mean(d_borel), digits=2))")

# Negative binomial offspring with R = 0.8, k = 0.5
d_nb = chain_size_distribution(NegBin(0.8, 0.5))
println("NegBin(R = 0.8, k = 0.5), P(size = 1): $(round(pdf(d_nb, 1), digits=4))")
```

The mean chain size with R = 0.8 is 1/(1 − R) = 5. P(size = 1) is the
probability that an introduction causes no onward transmission at all; with
strong superspreading (k = 0.5) most introductions do not spread.

## Likelihood

To ask how well a value of R explains observed chain sizes, put the sizes in
[`ChainSizes`](@ref) and compute the log-likelihood with `loglikelihood`. The
higher the log-likelihood, the better that value of R explains the data. The
calculation assumes each chain started from one index case and has finished
growing; see the [inference tutorial](inference.md) for clusters that started
from several cases or may still be growing.

```@example chains
data = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])

# Compare values of R for Poisson offspring
for R in [0.3, 0.5, 0.7, 0.9]
    ll = loglikelihood(data, Poisson(R))
    println("Poisson(R = $R): log-likelihood = $(round(ll, digits=2))")
end
```

Of the values tried, the log-likelihood is highest for R between 0.3 and 0.7;
the Fitting section below finds the maximum.

With negative binomial offspring:

```@example chains
ll = loglikelihood(data, NegBin(0.9, 0.5))
println("NegBin(R = 0.9, k = 0.5): log-likelihood = $(round(ll, digits=2))")
```

### Under-reporting

If only some cases are detected, observed clusters are smaller than the true
ones, and a lone detected case may belong to a larger cluster. To allow for this, say that each case is detected independently with
probability `detection_prob`, using [`PerCaseObservation`](@ref) in a
[`ModelSpec`](@ref). Each observed cluster size is then the number of detected
cases out of its true size.

```@example chains
data = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])
full = ModelSpec(BranchingProcess(Poisson(0.9)))
partial = ModelSpec(BranchingProcess(Poisson(0.9));
    observation = PerCaseObservation(detection_prob = 0.7))
println("All cases detected: $(round(loglikelihood(data, full), digits=2))")
println("70% detected:       $(round(loglikelihood(data, partial), digits=2))")
```

The log-likelihood changes because the model of what was observed changes.
Fit R under the reporting you believe applies: assuming complete reporting
when some cases are missed makes clusters look smaller than they are, and
biases the estimate of R downwards.

!!! warning "Clusters with no detected case"
    A cluster in which no case was detected never appears in the data, and the
    likelihood should allow for that. At present it does not, so it penalises
    every observed cluster more the lower the detection probability, and a fit
    of R and `detection_prob` together will favour detection that is too high
    (see [issue #416](https://github.com/epiforecasts/epiBranch.jl/issues/416)).

### Offspring counts

If you know who infected whom, put the number of secondary cases each case
caused in [`OffspringCounts`](@ref):

```@example chains
offspring_data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 5, 0])
ll = loglikelihood(offspring_data, Poisson(1.4))
println("Offspring counts, Poisson(R = 1.4): log-likelihood = $(round(ll, digits=2))")
```

### Chain lengths

If you know how many generations each chain lasted rather than how many cases
it had, put these in [`ChainLengths`](@ref), with 0 for a chain where the index
case infected nobody:

```@example chains
length_data = ChainLengths([0, 1, 0, 2, 1, 0, 0, 3, 0, 1])
ll = loglikelihood(length_data, Poisson(0.5))
println("Chain lengths, Poisson(R = 0.5): log-likelihood = $(round(ll, digits=2))")
```

### Building from contact-tracing records

Contact-tracing data usually come as a table of infector-infectee pairs or a
table of cluster memberships, not as counts already tallied.
[`OffspringCounts`](@ref) and [`ChainSizes`](@ref) build directly from those
records. For pairs, give the infector IDs and the infectee IDs as two vectors
of the same length; `unlinked` is the number of further cases with no known
infector and no known secondary cases. For clusters, give each case's cluster
label:

```@example chains
# 1 infected 2; 2 infected 3 and 4; two further cases have no known links.
infector = [1, 2, 2]
infectee = [2, 3, 4]
offspring_from_pairs = OffspringCounts(infector, infectee; unlinked = 2)

# Chain 1 has 2 cases, chain 2 has 1, chain 3 has 3.
sizes_from_membership = ChainSizes(; membership = [1, 1, 2, 3, 3, 3])
println(sort(offspring_from_pairs.data))
println(sort(sizes_from_membership.data))
```

The first line lists the number of secondary cases for each of the six cases;
the second lists the three chain sizes.

## Likelihood under interventions

Interventions change chain sizes. If cases are isolated after symptom onset,
chains end sooner than the offspring distribution alone implies, and fitting R
without allowing for isolation would underestimate it. To account for this,
give `loglikelihood` the whole model, isolation included, and it estimates the
likelihood by simulating that model.

The isolation below starts after an exponentially distributed delay from
symptom onset (mean 2 days) and stops all onward transmission for 7 days
(`duration`), after which the case can transmit again. The incubation period
is log-normal with a median of about 4.5 days.

```@example chains
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)

model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))

ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2]), model;
    n_sim = 1000,
    max_cases = 100,
    rng = StableRNG(42))
println("Log-likelihood under isolation: $(round(ll, digits=2))")
```

`n_sim` is the number of simulated chains used to estimate the likelihood: more
simulations give a less noisy estimate and take longer. The result is a Monte
Carlo estimate. It changes slightly from run to run unless you fix `rng`.
`max_cases` stops a simulated chain at 100 cases; a chain that reaches the cap
counts as "at least 100 cases".

The same `model` can be passed to `simulate` and to `loglikelihood`: the
model you fit is the model you simulate.

## Fitting

All the fitting methods use the same `loglikelihood` calculation. With one
parameter the simplest maximum-likelihood estimate is a grid: compute the
log-likelihood over a range of R and take the highest. For more parameters,
confidence intervals and Bayesian estimation, see the
[inference tutorial](inference.md).

```@example chains
rng = StableRNG(42)
truth = BranchingProcess(Poisson(0.5), Exponential(5.0))
states = simulate(truth, 500; rng = rng)  # 500 chains, one index case each
data = ChainSizes(chain_statistics(states).size)

# Try R from 0.05 to 0.95 in steps of 0.01 and keep the best
R_grid = 0.05:0.01:0.95
R_hat = R_grid[argmax([loglikelihood(data, Poisson(R)) for R in R_grid])]
println("Maximum-likelihood estimate from chain sizes: R = $(round(R_hat, digits=2)) (true R = 0.5)")
```

### Bayesian inference with Turing.jl

[Turing.jl](https://turinglang.org) is a Julia package for Bayesian inference,
playing the role of Stan or brms in R. A Turing model lists each parameter's
prior and then says how the data are distributed given the parameters. In both
lines `~` reads "is distributed as". [`chain_size_distribution`](@ref) gives
the distribution of chain sizes under a model, ready to put after `~`
([`chain_length_distribution`](@ref) and [`offspring_distribution`](@ref) do the
same for chain lengths and offspring counts):

```julia
using Turing

@model function chain_model(data)
    R ~ LogNormal(-0.5, 1.0)  # prior for R
    data ~ chain_size_distribution(BranchingProcess(Poisson(R)))  # likelihood
end
```

With no interventions this uses the exact formula and fitting is fast. The
formula also covers the options `seeds` (the number of index cases in each
cluster) and `prob_concluded` (the probability that each cluster has finished
growing):

```julia
data ~ chain_size_distribution(BranchingProcess(Poisson(R));
    seeds = seeds, prob_concluded = prob_concluded)
```

If the model includes interventions, EpiBranch estimates the likelihood by
simulation instead, which is slower.

!!! warning
    `seeds` and `prob_concluded` only work for models without interventions.
    With interventions, `prob_concluded` stops with an error, and `seeds` is
    ignored: every cluster is treated as starting from a single index case.

The [inference tutorial](inference.md) runs these models in full.
