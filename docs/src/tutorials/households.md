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

## The reproduction number between households

A household model describes what happens inside a household. Transmission
*between* them is a branching process of its own, whose unit is a whole
household: an infected household infects other households through the community
contacts of its members, and the epidemic grows only if one infected household
infects more than one other on average. That threshold is R*.

[`household_offspring`](@ref) gives the law those households follow. It needs
one number the household process does not have, the rate at which an infectious
individual makes contact outside its own household. Early in an epidemic each
such contact reaches a susceptible person in a fresh household, so a household
infects a Poisson number of others with mean that rate times the total infectious
person-time of its own outbreak. That person-time is random, because the
household outbreak is.

```@example households
offspring = household_offspring(model; global_rate = 0.1, rng = StableRNG(5))
reproduction_number(offspring)
```

### The size-one limit

Shrinking every household to a single member switches the within-household
epidemic off: there is no household-mate left to infect, so the household-level
offspring law reduces to a single case's own community contacts. This is the
ordinary branching process, the household construction's degenerate case, and
it is a check the construction has to pass: everything
[`chain_size_distribution`](@ref) and [`extinction_probability`](@ref) already
gave for a plain [`BranchingProcess`](@ref) has to fall out unchanged.

With a fixed six-day infectious window, community contacts arrive at a constant
Poisson rate for exactly that long, so the offspring law is a plain Poisson and
[`chain_size_distribution`](@ref) recovers `Borel` exactly, with no simulation
on either side of the comparison:

```@example households
lone = ModelSpec(HouseholdProcess(fill(1, 1), Exponential(1.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
R_lone = reproduction_number(household_offspring(lone; global_rate = 0.1,
    rng = StableRNG(9)))
chains = chain_size_distribution(BranchingProcess(Poisson(R_lone)))
(law = typeof(chains), mean = mean(chains), formula = 1 / (1 - R_lone))
```

An exponential window instead makes the community-infectious period itself
random. A Poisson count compounded over an exponential mean is a geometric, a
Negative Binomial with one degree of freedom, so the chain size is
`GammaBorel`, again exactly:

```@example households
lone_exp = ModelSpec(HouseholdProcess(fill(1, 1), Exponential(1.0));
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(6.0), terminal = true)])
R_lone_exp = reproduction_number(household_offspring(lone_exp; global_rate = 0.1,
    rng = StableRNG(9)))
typeof(chain_size_distribution(BranchingProcess(NegBin(R_lone_exp, 1.0))))
```

A real household, of more than one member, breaks this. The compounding then
runs over the household's own final-size distribution rather than one case's
window, and the result no longer falls into a named family.
`reproduction_number` and `extinction_probability` answer the same questions
regardless, since they never assumed a family, but `chain_size_distribution`
has nothing to dispatch on beyond `Poisson` and `NegativeBinomial`, so the number of
households ultimately infected has no closed form once households hold more
than one person.

The law itself is a `Distributions.jl` distribution, so it can be plotted, sampled,
or handed to a [`BranchingProcess`](@ref) to simulate chains of infected households:

```@example households
law = household_offspring_law(offspring)
(pdf(law, 0), pdf(law, 1), pdf(law, 2))
```

One household differs from another by its size, so size is the type of this
branching process. A community contact reaches a *person*, and with them
their household: larger households are therefore reached more often than their
share of households alone would suggest. They then make more onward infections,
because more of their members are infected. Both effects are in the answer:

```@example households
mixed = ModelSpec(HouseholdProcess([fill(2, 400); fill(5, 200)], Weibull(1.5, 12.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
sized = household_offspring(mixed; global_rate = 0.1, rng = StableRNG(6))
(sizes = sized.sizes, reached = sized.mixing, offspring = sized.means)
```

[`extinction_probability`](@ref) answers the question a household model is usually
asked: an infected household appears, how likely is that the end of it? It gives
one probability per household type, because the first household's type is the one
thing that is not drawn from the mixing weights.

```@example households
extinction_probability(sized)
```

### The same construction as a multi-type `BranchingProcess`

A household's type bears only on how many other households it infects; who
those households are is drawn from the size-biased mixing weights whatever the
parent's type was. That is exactly the offspring matrix of a multi-type
[`BranchingProcess`](@ref): `M[i, j] = mixing[i] * means[j]`, a rank-one matrix
whose column `j` gives type-`j`'s contribution and whose rows share the same
proportions. Built this way, the general multi-type
[`reproduction_number`](@ref) and [`extinction_probability`](@ref), which work
from any offspring matrix and not just a household's, apply directly:

```@example households
M = sized.mixing * sized.means'
household_bp = BranchingProcess(M, R -> Poisson(R), Exponential(5.0))
(exact = reproduction_number(sized), multitype = reproduction_number(household_bp))
```

R\* agrees exactly, because it depends on the offspring matrix alone. Choosing
`Poisson` to turn each type's mean into a distribution is an assumption `sized`
never makes: it uses the household's own, generally non-Poisson law instead.
`extinction_probability` depends on more than the mean, so the two part ways
there:

```@example households
(exact = extinction_probability(sized),
    poisson_approximation = extinction_probability(household_bp))
```

R\* is a between-household summary and survives any distributional guess at
the family; the extinction threshold is a property of the whole offspring law,
and only [`household_offspring`](@ref)'s own, kernel-derived law gets it right.

With a covariate kernel, households of one size need not be alike: who the
members are decides how fast the household outbreak runs. The law is then built
from the model's own households, each starting from a member picked uniformly at
random, and the type is the household itself. Households whose kernels agree pair
for pair share a type, so here, where one in three households transmits faster,
there are two types of each size:

```@example households
fast_household = [h % 3 == 0 for h in 1:600]
household_of = reduce(vcat, [fill(h, n) for (h, n) in
    enumerate([fill(2, 400); fill(5, 200)])])
covariate = ModelSpec(
    HouseholdProcess([fill(2, 400); fill(5, 200)],
        (infector, susceptible) -> Weibull(1.5,
            fast_household[household_of[infector]] ? 4.0 : 12.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
typed = household_offspring(covariate; global_rate = 0.1, rng = StableRNG(8))
(sizes = typed.sizes, reached = typed.mixing, offspring = typed.means)
```

The model's own layers apply throughout. An isolation intervention shortens each
case's infectious window, which cuts both the household members it infects and the
community contacts it makes, so R* follows from the censoring:

```@example households
isolated = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [Transition(:onset; from = :infection, delay = 1.0),
        Transition(:recovered; from = :infection, delay = 6.0, terminal = true)],
    interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0),
        eligibility = AllCases())])
reproduction_number(household_offspring(isolated; global_rate = 0.1,
    rng = StableRNG(7)))
```

### What this formulation buys, and what it costs

Casting the classical households model as a branching process over households
gets its threshold and extinction behaviour for nothing: the global level is
exactly as analytic as the plain branching process it specialises, R\* and
`extinction_probability` come from the same fixed-point machinery, and an
intervention on the household layer is read straight through into the
global offspring law without a separate global-level parameter to
recalibrate. The cost falls entirely on the within-household level: the
household kernel is resolved exactly when it is Markovian and otherwise by
simulating households, which is exact per household but leaves a
Monte Carlo error in the fed-forward mean once an intervention with
individual-level timing enters the household's own infectious window.

The construction is a branching approximation, not an epidemic model: every
community contact is assumed to reach a household untouched by the outbreak
so far, which holds only while infected households are a small fraction of
all households. There is no population to deplete, so there is no epidemic
peak and no final size for the whole population to compute, only the early,
branching phase this chapter describes: R\*, and how likely a single
introduction is to die out. A model that keeps a finite, depleting pool of
households for the phase this one cannot reach is a different piece of work.

Between-household contact tracing is not separately implemented: it is the
same `ContactTracing` shown in [Interventions](interventions.md), attached to
[`HouseholdProcess`](@ref), where a case's household-mates are already its
contacts. Tracing therefore finds a flagged case's household-mates one at a
time, through the same competing-risk resolution as any other contact, rather
than as a single action against the household as a whole, so it gets no
benefit from the fact that a real household's members share one exposure and
could in principle all be flagged together.

The within-household epidemic behind all of these is available on its own.
[`household_final_size`](@ref) gives the exact distribution of how many of a
household's members are ultimately infected, for any contact-interval kernel and
infectious window:

```@example households
d = household_final_size(4, Weibull(1.5, 12.0), 6.0)
(mean = mean(d), all_four = pdf(d, 4))
```

Where the household epidemic has a closed form, which an exponential contact
interval racing an exponential infectious window gives when no intervention
applies, the offspring law is solved exactly and no simulation runs. Otherwise
households of each size are simulated (`n_samples`, 10,000 by default; with a
covariate kernel, the whole model is simulated until that many households have
run, and at least once), and the Monte Carlo error then sits only in the
within-household epidemic. The Poisson compounding on top of it is analytical,
and so, with a shared kernel, is the mean whenever the infectious window is a
single delay of the progression (except in a large, weakly transmitting
household with a random window, where the final-size recursion loses accuracy
and the mean comes from the simulated households).

The construction is the classical two-level-mixing model of Ball, Mollison and
Scalia-Tomba (1997), and R* is their R*. The within-household final size comes
from Ball's (1986) recursion.

## The pairwise likelihood

Infections are latent: the model generates them, and the progression maps each to
its observable outcomes. [`pairwise_surv_loglik`](@ref) is the contact-process
density of that **infection layer**, which [`household_infections`](@ref) reads out
of a simulation. Because the Sellke construction is the likelihood's generative
model, `simulate → loglikelihood` is an exact round trip, so the simulated outbreak
recovers the kernel.

!!! note "Per-contact risks thin the hazard"
    A model that also carries per-contact competing risks — a per-individual
    susceptibility or infectiousness, a leaky isolation, a vaccine's efficacy —
    blocks some of the contacts the race proposes, and the pair goes on meeting
    afterwards. Blocking a fraction `p` of contacts thins each pair's hazard to
    `(1 - p)` of it, which for an exponential contact interval is the same
    process at a rate scaled by `1 - p`. The round trip then holds against the
    scaled kernel rather than the one the model was given, and only for a risk
    that is in place throughout: an isolation, or a dose a trace gives, arrives
    partway through a window and has no term in the pairwise likelihood at all.

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
