# Analytical functions

Closed-form solutions for offspring distribution properties — no simulation needed.

## Extinction and epidemic probability

For a branching process with Negative Binomial offspring, the extinction
probability is computed via fixed-point iteration on the probability
generating function:

```@example analytical
using EpiBranch
using Distributions
using DataFrames

# For Geometric offspring (k=1), extinction prob = 1/R
q = extinction_probability(3.0, 1.0)
println("Geometric(R=3): P(ext) = $(round(q, digits=4)) (analytical: $(round(1/3, digits=4)))")
```

The epidemic probability is the complement of the extinction probability:

```@example analytical
p = epidemic_probability(2.5, 0.16)
println("NegBin(R=2.5, k=0.16): P(epidemic) = $(round(p, digits=3))")
```

### Effect of dispersion

Higher overdispersion (lower k) increases extinction probability — even
with the same R:

```@example analytical
for k in [0.01, 0.1, 0.5, 1.0, 10.0]
    q = extinction_probability(2.5, k)
    println("k = $(lpad(k, 5)): P(ext) = $(round(q, digits=3))")
end
```

### Distribution dispatch

`Distribution` objects are also accepted directly:

```@example analytical
println("Poisson(2.0):   P(ext) = $(round(extinction_probability(Poisson(2.0)), digits=4))")
println("NegBin(2, 0.5): P(ext) = $(round(extinction_probability(NegBin(2.0, 0.5)), digits=4))")
```

## Superspreading: proportion of transmission

What fraction of transmission is caused by the most infectious fraction
of cases? This is the "80/20 rule" metric:

```@example analytical
# SARS-CoV-2: R = 2.5, k = 0.16
prop = proportion_transmission(2.5, 0.16; prop_cases = 0.2)
println("Top 20% of cases cause $(round(prop * 100, digits=1))% of transmission")
```

The result depends almost entirely on the dispersion parameter k:

```@example analytical
for k in [0.01, 0.1, 0.16, 0.5, 1.0, 10.0, 1000.0]
    prop = proportion_transmission(2.5, k; prop_cases = 0.2)
    println("k = $(lpad(k, 6)): top 20% → $(round(prop * 100, digits=1))%")
end
```

As k → ∞ (no overdispersion), the top 20% cause ≈ 20% of transmission.

## Chain size distributions

Analytical chain size distributions for specific offspring families:

```@example analytical
# Borel distribution (chain sizes from Poisson offspring)
d = Borel(0.8)
println("Borel(0.8):")
println("  Mean chain size: $(round(mean(d), digits=2))")
for n in 1:5
    println("  P(size=$n) = $(round(pdf(d, n), digits=4))")
end
```

[`chain_size_distribution`](@ref) dispatches on the offspring type and
returns the appropriate analytical distribution:

```@example analytical
d_pois = chain_size_distribution(Poisson(0.8))
d_nb = chain_size_distribution(NegBin(0.8, 0.5))
println("Poisson(0.8) chain sizes: $(typeof(d_pois))")
println("NegBin(0.8, 0.5) chain sizes: $(typeof(d_nb))")
```

## Validation against simulation

```@example analytical
using StableRNGs

R, k = 1.5, 0.5
q_exact = extinction_probability(R, k)

model = BranchingProcess(NegBin(R, k), Exponential(5.0))
results = simulate_batch(model, 5000;
    sim_opts = SimOpts(max_cases = 10_000, max_generations = 200),
    rng = StableRNG(42),
)
q_sim = containment_probability(results)

println("R=$R, k=$k:")
println("  Analytical: $(round(q_exact, digits=4))")
println("  Simulated:  $(round(q_sim, digits=4))")
```
