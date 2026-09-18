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

The contact-interval kernel is the one required input to `HouseholdProcess`.
`sizes` gives the size of each household. The disease's natural history is a
`progression` attached with a [`ModelSpec`](@ref).

```@example households
using EpiBranch
using EpiHouseholds
using Distributions
using StableRNGs

# 300 households of four, a Weibull contact interval, a six-day infectious period
model = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
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

The infectious timeline is a `progression` of [`Transition`](@ref)s on the
[`ModelSpec`](@ref), exactly as for [`BranchingProcess`](@ref). A latent period is
a `Transition(:infectious; from = :infection, …)`; an infectious period is a
terminal removal transition timed from the state before it. The progression's
states become line-list columns, so symptom onset, testing and recovery come
straight out of the simulation. The kernel times each infectious contact from the
infectious window's start; with a latent period present, that `from` state is
derived as `:infectious`, otherwise `:infection`.

```@example households
clinical = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [
        Transition(:infectious; from = :infection, delay = LogNormal(1.2, 0.4)),  # infection → infectiousness
        Transition(:recovered; from = :infectious, delay = Gamma(6, 1),           # infectiousness → recovery
            terminal = true)])
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
truth = ModelSpec(HouseholdProcess(fill(4, 500), Exponential(4.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
data = household_infections(simulate(truth; rng = StableRNG(3)), truth)

ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 2.0:0.5:6.0
grid[argmax([ll(s) for s in grid])]   # ≈ the true scale, 4.0
```

The kernel can also be a callable `(infector, susceptible) -> Distribution` that
takes host ids, which allows covariate models such as adults transmitting faster
than children. The simulator and every `pairwise_surv_loglik` form that takes
household data call it with the ids in that order. One callable therefore works for
both simulation and fitting, and fitting the simulated outbreak recovers each of its
parameters.

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
augmented) times on the fly. The two forms agree up to row order. The layout is
EpiBranch's [`ContactPairsLayout`](@ref), built by
[`compile_contact_pairs`](@ref) from the household partition. Both work for any
contact structure, and a contact network is fitted the same way (see
[Fitting on a network](@ref "Fitting on a network")).

In real data the infection times are unobserved. A household `@model` then augments
them and conditions the observed onsets and tests through the progression's delays,
with `pairwise_surv_loglik` supplying the contact-process density of the augmented
configuration. The layout stays valid across draws as long as the household
structure and the set of ever-infected hosts are fixed, because only the latent
times move. Compile it once, outside the model, and reuse it.

Data collected up to a date describe an outbreak that may still be going. Give
`HouseholdInfections` that date as `followup_end` and the density ignores
infections and exposure after it; a case still infectious at the end of
follow-up keeps a removal time of `Inf`. An impossible configuration, such as a
case infected when none of its household-mates is infectious and no community
hazard can reach it, has zero density, and `pairwise_surv_loglik` returns `-Inf`
for it. Without a community hazard the density conditions on index cases, and
they need no possible infector. The `-Inf` comes with a zero gradient, since
whether a configuration is possible at all depends on the times alone.

### Fitting a community hazard

A positive `external_hazard` and no community hazard are different conditionings,
and the density jumps between them at `α = 0`. With a constant rate `α > 0` an
index case infected at time `t` contributes `log(α) - α t`, which falls to
`-Inf` as `α → 0`: a model that admits community introductions has to
explain the ones it saw, and vanishingly rare introductions explain them
vanishingly badly. At exactly `external_hazard = 0` index cases are conditioned on
instead and contribute nothing, and the value stays finite. Each is correct for
what it conditions on.

For fitting, this means a likelihood ratio between "some community transmission"
and "none" cannot be read off by letting `α` approach zero. Score the two models
separately.

The discontinuity is only at that one point, and the density behaves regularly
as `α` approaches it. Drop the terms free of `α` and the log-density near zero is
`k log α - α T`, where `k` counts the cases the community alone can explain and
`T` is the total time the population is exposed to it. In `log α` that is a
straight line of slope `k`. On 400 households of four with `k = 401`,
`d ll / d log α` is 401.0 at `α = 1e-6` and 393.5 at `1e-3`, falling to zero at
the mode near `α = 0.052`.

ForwardDiff cannot differentiate a `Gamma`, whether it is the community hazard or
the contact-interval kernel. Its cumulative hazard calls
`SpecialFunctions._gamma_inc`, which has no `ForwardDiff.Dual` method, and the
resulting `MethodError` comes from there rather than from this package. Fit a
`Gamma` with a reverse-mode backend, `NUTS(; adtype = AutoMooncake())`.
`Weibull` and `Exponential`, the kernels used above, differentiate under either
mode.
