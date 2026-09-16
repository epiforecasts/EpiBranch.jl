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

A household model describes what happens inside a household. What happens
*between* them is a branching process of its own, whose unit is a whole
household: an infected household infects other households through the community
contacts of its members, and the epidemic grows only if one infected household
infects more than one other on average. That threshold is R*.

[`household_offspring`](@ref) builds the law those households follow. It needs
one number the household process does not carry, the rate at which an infectious
individual makes contact outside its own household. Early in an epidemic each
such contact reaches a susceptible person in a fresh household, so a household
infects a Poisson number of others with mean that rate times the total infectious
person-time of its own outbreak — which is random, because the household outbreak
is.

```@example households
offspring = household_offspring(model; global_rate = 0.1, rng = StableRNG(5))
reproduction_number(offspring)
```

The law itself is a `Distributions.jl` distribution, so it can be plotted, sampled,
or handed to a [`BranchingProcess`](@ref) to simulate chains of infected households:

```@example households
law = household_offspring_law(offspring)
(pdf(law, 0), pdf(law, 1), pdf(law, 2))
```

Household size is what makes one household differ from another, so it is the type
of this branching process. A community contact reaches a *person*, and with them
their household, so larger households are reached more often than their share of
households alone would suggest — and they then make more onward infections,
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
one probability per household size, because the first household's size is the one
thing that is not drawn from the mixing weights.

```@example households
extinction_probability(sized)
```

The model's own layers are in all of this. An isolation intervention shortens each
case's infectious window, which cuts both the household members it infects and the
community contacts it makes, so R* falls out of the censoring rather than being
adjusted by hand:

```@example households
isolated = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [Transition(:onset; from = :infection, delay = 1.0),
        Transition(:recovered; from = :infection, delay = 6.0, terminal = true)],
    interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0),
        eligibility = AllCases())])
reproduction_number(household_offspring(isolated; global_rate = 0.1,
    rng = StableRNG(7)))
```

The within-household epidemic behind all of these is available on its own.
[`household_final_size`](@ref) gives the exact distribution of how many of a
household's members are ultimately infected, for any contact-interval kernel and
infectious window:

```@example households
d = household_final_size(4, Weibull(1.5, 12.0), 6.0)
(mean = mean(d), all_four = pdf(d, 4))
```

Where the household epidemic has a closed form — an exponential contact interval
racing an exponential infectious window, with no interventions — the offspring law
is solved exactly and no simulation runs. Otherwise households of each size are
simulated (`n_samples`, 10,000 by default) and only the within-household epidemic
carries Monte Carlo error; the Poisson compounding on top of it is analytical, and
so is the mean whenever the infectious window is a single delay of the progression.

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
augmented) times on the fly. The two forms agree up to row order.

In real data the infection times are unobserved. A household `@model` then augments
them and conditions the observed onsets and tests through the progression's delays,
with `pairwise_surv_loglik` supplying the contact-process density of the augmented
configuration. The layout stays valid across draws as long as the household
structure and the set of ever-infected hosts are fixed — only the latent times
move — so it is compiled once, outside the model, and reused.
