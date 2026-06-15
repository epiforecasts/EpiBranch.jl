# Household models

`HouseholdProcess` spreads infection within households. The population is
partitioned into households; within a household every infectious member can
infect every susceptible household-mate, with the timing of infectious contact
drawn from a **contact-interval** kernel (Kenah 2011). It is a structure-driven
model like [`NetworkProcess`](@ref), but a household is a small, depleting
clique, so it is simulated by the **Sellke construction** in continuous time —
the exact generative model of its pairwise likelihood — rather than the
generation-based engine.

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

`date_infectious` and `date_recovered` appear because the progression stamps
`:infectious_time` and `:recovered_time` on each case.

## The pairwise likelihood

Infections are latent: the model generates them, and the progression maps each to
its observable outcomes. The contact-process likelihood scores that **infection
layer** — read out of a simulation with [`household_infections`](@ref) — through
[`pairwise_surv_loglik`](@ref). Because the Sellke construction is the likelihood's
generative model, `simulate → loglikelihood` is an exact round trip: the simulated
outbreak recovers the kernel.

```@example households
truth = HouseholdProcess(fill(4, 500), Exponential(4.0); infectious_period = 6.0)
data = household_infections(simulate(truth; rng = StableRNG(3)), truth)

ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 2.0:0.5:6.0
grid[argmax([ll(s) for s in grid])]   # ≈ the true scale, 4.0
```

In real inference the infection times are latent and augmented in a Turing
`@model`: the contact process scores the augmented infections, while the observed
onsets and tests are conditioned through the progression's delays. The
inference-friendly `pairwise_surv_loglik(kernel, data; external_hazard)` takes the
fitted kernel and the augmented infection layer separately, so neither the model
nor the household structure is rebuilt per evaluation.
