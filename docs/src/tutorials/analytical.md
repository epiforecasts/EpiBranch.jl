# Analytical functions

How likely is one introduced case to start a large outbreak, how much of
transmission comes from the most infectious cases, and how large do chains get
when R < 1? For a branching process these have exact answers in terms of R and
the dispersion k, with no simulation needed. Throughout, `NegBin(R, k)` is a
negative binomial offspring distribution with mean R and dispersion k: smaller k
means more superspreading, and k = 1 gives a geometric distribution.

## Extinction and epidemic probability

The extinction probability is the chance that the transmission chain started by
one introduced case dies out by itself, without interventions. It is the
smallest solution of q = G(q), where G is the probability generating function
of the offspring distribution. When R ≤ 1 it is 1; for geometric offspring
(k = 1) with R > 1 it is 1/R, which gives a check:

```@example analytical
using EpiBranch
using Distributions
using DataFrames

# R = 3, k = 1 (geometric offspring)
q = extinction_probability(3.0, 1.0)
println("Geometric(R=3): P(ext) = $(round(q, digits=4)) (exact: $(round(1/3, digits=4)))")
```

The epidemic probability is the complement: the chance that one introduced case
starts an outbreak that does not die out by itself. With R = 2.5 and k = 0.16,
estimates for SARS-CoV-2
([Endo et al. 2020](https://doi.org/10.12688/wellcomeopenres.15842.3)), it is low despite the high R:

```@example analytical
p = epidemic_probability(2.5, 0.16)
println("NegBin(R=2.5, k=0.16): P(epidemic) = $(round(p, digits=3))")
```

### Effect of dispersion

With the same R, more superspreading (lower k) makes extinction more likely,
because most cases infect nobody and the outbreak depends on the rare cases that
infect many:

```@example analytical
for k in [0.01, 0.1, 0.5, 1.0, 10.0]
    q = extinction_probability(2.5, k)
    println("k = $(lpad(k, 5)): P(ext) = $(round(q, digits=3))")
end
```

At k = 0.1 most introductions die out even though R = 2.5; at k = 10 the
offspring distribution is close to Poisson and most introductions take off.

### Using a distribution or a model instead of R and k

Every analytical function also accepts an offspring distribution from
Distributions.jl, or a [`BranchingProcess`](@ref) model, in place of R and k:

```@example analytical
println("Poisson(2.0):   P(ext) = $(round(extinction_probability(Poisson(2.0)), digits=4))")
println("NegBin(2, 0.5): P(ext) = $(round(extinction_probability(NegBin(2.0, 0.5)), digits=4))")

model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
println("Model:          P(ext) = $(round(extinction_probability(model), digits=4))")
```

The model's answer depends only on its offspring distribution and matches
`extinction_probability(2.5, 0.16)`; the generation time does not change it.

!!! warning "Interventions are not included"
    These functions also accept a [`ModelSpec`](@ref), but they read only its
    offspring distribution. Isolation, contact tracing or vaccination in the
    model are left out, so the answer is for the outbreak without any of them
    (see [issue #421](https://github.com/epiforecasts/epiBranch.jl/issues/421)).
    For the chance of containing an outbreak under interventions, simulate the
    model and use [`containment_probability`](@ref), as in
    [Interventions](interventions.md).

## Superspreading: proportion of transmission

What fraction of transmission comes from the most infectious 20% of cases? This
is the "80/20 rule"
([Lloyd-Smith et al. 2005](https://doi.org/10.1038/nature04153)). The calculation
assumes each case's individual reproduction number (its expected number of
secondary cases) follows a gamma distribution with mean R and shape k, the model
behind the negative binomial offspring distribution:

```@example analytical
# SARS-CoV-2: R = 2.5, k = 0.16
prop = proportion_transmission(2.5, 0.16; prop_cases = 0.2)
println("Top 20% of cases cause $(round(prop * 100, digits=1))% of transmission")

# The same, from a model
println("Model: top 20% cause $(round(proportion_transmission(model; prop_cases=0.2) * 100, digits=1))%")
```

Under this gamma model the proportion does not depend on R, only on k:

```@example analytical
for k in [0.01, 0.1, 0.16, 0.5, 1.0, 10.0, 1000.0]
    prop = proportion_transmission(2.5, k; prop_cases = 0.2)
    println("k = $(lpad(k, 6)): top 20% → $(round(prop * 100, digits=1))%")
end
```

As k grows (no superspreading), the top 20% of cases cause close to 20% of
transmission.

The reverse question, what proportion of cases cause a given share of
transmission (such as the commonly reported "proportion of cases responsible
for 80% of transmission"), has two answers that can differ substantially,
because they rank cases by different things:

- [`proportion_cases_individual`](@ref) ranks cases by their individual
  reproduction number, under the same gamma model as
  `proportion_transmission`.
- [`proportion_cases_offspring`](@ref) ranks cases by the number of secondary
  cases they actually caused, a whole number. This is the version usually
  reported alongside the "80/20 rule", and it works for any discrete offspring
  distribution with a finite mean, not only the negative binomial.

```@example analytical
R, k = 1.0, 0.4
individual = proportion_cases_individual(R, k; prop_transmission = 0.8)
offspring = proportion_cases_offspring(R, k; prop_transmission = 0.8)
println("Individual-R version:       $(round(individual * 100, digits=1))% of cases cause 80% of transmission")
println("Realised-offspring version: $(round(offspring * 100, digits=1))% of cases cause 80% of transmission")
```

Here the realised version is lower: chance adds to the differences between
cases, so transmission is concentrated in fewer cases than their individual
reproduction numbers imply. Report both, clearly labelled, since published
figures use either.

## Chain size distributions

When R < 1 every chain dies out, and the distribution of its final size is
known exactly for Poisson and negative binomial offspring. With Poisson
offspring it is the Borel distribution, whose parameter is R:

```@example analytical
# Chain sizes for Poisson offspring with R = 0.8
d = Borel(0.8)
println("Borel(0.8):")
println("  Mean chain size: $(round(mean(d), digits=2))")
for n in 1:5
    println("  P(size=$n) = $(round(pdf(d, n), digits=4))")
end
```

P(size = 1) is the chance that an introduction causes no onward transmission.

[`chain_size_distribution`](@ref) chooses the formula matching the offspring
distribution:

```@example analytical
d_pois = chain_size_distribution(Poisson(0.8))
d_nb = chain_size_distribution(NegBin(0.8, 0.5))
println("Poisson(R=0.8):        mean size $(round(mean(d_pois), digits=2)), P(size=1) = $(round(pdf(d_pois, 1), digits=3))")
println("NegBin(R=0.8, k=0.5):  mean size $(round(mean(d_nb), digits=2)), P(size=1) = $(round(pdf(d_nb, 1), digits=3))")
```

Both have the same mean size, 1/(1 − R) = 5, but with superspreading many more
introductions stop at a single case.

## Checking against simulation

The exact extinction probability should match the proportion of simulated
introductions that die out. The simulation below runs 1,000 outbreaks, each
from one introduced case, and stops any that reaches 10,000 cases or 200
generations; those count as not extinct. The caps make the simulated
proportion a slight underestimate. [`containment_probability`](@ref) gives the
proportion of simulated outbreaks that ended before reaching a cap.

```@example analytical
using StableRNGs

R, k = 1.5, 0.5
q_exact = extinction_probability(R, k)

model = BranchingProcess(NegBin(R, k), Exponential(5.0))
results = simulate(model, 1000;
    max_cases = 10_000, max_generations = 200,
    rng = StableRNG(42),
)
q_sim = containment_probability(results)

println("R=$R, k=$k:")
println("  Exact:      $(round(q_exact, digits=4))")
println("  Simulated:  $(round(q_sim, digits=4))")
```
