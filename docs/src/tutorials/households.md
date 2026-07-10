# Household models

`HouseholdProcess` spreads infection within households. The population is
partitioned into households; within a household every infectious member can
infect every susceptible household-mate, with the timing of infectious contact
drawn from a **contact-interval** kernel (Kenah 2011). It is a structure-driven
model like [`NetworkProcess`](@ref), and like it is simulated by the **Sellke
construction** in continuous time (the exact generative model of its pairwise
likelihood) rather than by the generation-based engine; a household is a small,
depleting clique rather than a fixed graph.

It lives in the companion `EpiHouseholds` package.

## Defining a household model

The contact-interval kernel is the one required input. `sizes` gives the size of
each household.

```@example households
using EpiBranch
using EpiHouseholds
using Distributions
using StableRNGs

# 300 households of four, a Weibull contact interval, a six-day infectious period
model = HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0); infectious_period = 6.0)
```

## Simulating

`simulate` returns a `SimulationState`, and [`linelist`](@ref) renders the
one-row-per-case table. Each household is seeded with one index, and the
outbreak spreads within it.

```@example households
state = simulate(model; rng = StableRNG(1))
df = linelist(state)
(cases = size(df, 1), indexes = count(df.index))
```

## A flexible natural history

The infectious timeline is a `progression` of [`Transition`](@ref)s, exactly as
for [`BranchingProcess`](@ref). `infectious_period` and `latent_period` are sugar
for the two common transitions; pass a full `progression` for the general case.
The progression's states become line-list columns, so symptom onset, testing and
recovery come straight out of the simulation.

```@example households
clinical = HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0);
    latent_period = LogNormal(1.2, 0.4),     # infection → infectiousness
    infectious_period = Gamma(6, 1))          # infectiousness → recovery
sort(propertynames(linelist(simulate(clinical; rng = StableRNG(2)))))
```

`date_infectious` and `date_recovered` appear because the progression writes
`:infectious_time` and `:recovered_time` onto each case.

## The pairwise likelihood

Infections are latent: the model generates them, and the progression maps each to
its observable outcomes. [`pairwise_surv_loglik`](@ref) is the contact-process
density of that **infection layer**, which [`household_infections`](@ref) reads out
of a simulation. Because the Sellke construction is the likelihood's generative
model, `simulate → loglikelihood` is an exact round trip, so the simulated outbreak
recovers the kernel.

```@example households
truth = HouseholdProcess(fill(4, 500), Exponential(4.0); infectious_period = 6.0)
data = household_infections(simulate(truth; rng = StableRNG(3)), truth)

ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 2.0:0.5:6.0
grid[argmax([ll(s) for s in grid])]   # ≈ the true scale, 4.0
```

## Fitting with Turing

When the infection layer is observed (here it comes directly from the simulation),
the likelihood slots into a Turing `@model`. Put a prior on the log contact rate
and add the pairwise log-density to the target. The household structure is fixed
across draws, so [`compile_household_pairs`](@ref) captures the pair layout once
and each evaluation reuses it — no per-sample rebuild:

```@example households
using Turing

layout = compile_household_pairs(data)   # the fixed pair structure, compiled once

@model function household_fit(data, layout)
    logβ ~ Normal(-1, 1)                # log within-household contact rate
    Turing.@addlogprob! pairwise_surv_loglik(Exponential(1 / exp(logβ)), data, layout)
end

chain = sample(StableRNG(4), household_fit(data, layout), NUTS(), 300; progress = false)
exp(-mean(chain[:logβ]))                # posterior mean contact-interval scale, ≈ 4.0
```

The plain `pairwise_surv_loglik(kernel, data; external_hazard)` form re-derives the
pair structure (a bucketed pass over households, one susceptible-grouped row list)
on every call. [`HouseholdPairsLayout`](@ref) hoists that structural work out of the
gradient loop: [`compile_household_pairs`](@ref) enumerates the ordered
(susceptible, infector) rows once — everything that doesn't depend on the sampled
parameters — and the three-argument `pairwise_surv_loglik(kernel, data, layout)`
then evaluates the density in two allocation-free passes, reading the (possibly
augmented) times on the fly. The two forms agree up to row order.

In real data the infection times are unobserved. A household `@model` then augments
them and conditions the observed onsets and tests through the progression's delays,
with `pairwise_surv_loglik` supplying the contact-process density of the augmented
configuration. The layout stays valid across draws as long as the household
structure and the set of ever-infected hosts are fixed — only the latent times
move — so it is compiled once, outside the model, and reused.
