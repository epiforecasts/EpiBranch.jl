# Simulating daily counts without individuals

A branching process in which nothing about a person feeds back on transmission does not need each person simulated to give the right daily counts. [`simulate`](@ref) runs such a model on a count-level path: it draws the number of people infected on each day from the cohorts of earlier days, at a cost that grows with the number of days, whatever the number of cases (an offspring law other than `Poisson` or `NegBin` is the exception, see below). The result is a [`CountSimulationState`](@ref), which builds the people, and so the line list, only when they are asked for.

## When a model runs on counts

The model must have a generation time that is a discrete distribution on whole days, starting at day 1 and ending at a largest value. Nothing about individuals may act on transmission: the only interventions allowed are the vaccination campaigns described below. The population is unlimited or finite: a finite one is depleted day by day, type by type, as [`simulate_by_generation`](@ref) does for a generation. A model with several types qualifies when it is built from an offspring matrix with an offspring law for each type of one of the kinds below and a population with a size for each type (`Population(sizes)`); a single `Population(size)` shared by several types runs as individuals. The offspring law is a `Poisson`, a negative binomial (`NegBin`), a [`PoissonLognormal`](@ref) or [`PoissonWeibull`](@ref), or a [`RenewalOffspring`](@ref) with a vector generation time. A negative binomial cohort needs one gamma draw for its total infectiousness. A Poisson lognormal or Poisson Weibull cohort needs one draw for each member, and the state keeps these draws to share out the infections of the cohort when the people are built. For these two laws the time and memory grow with the number of cases as well as the number of days. A model with one type and an unlimited population also qualifies with any other discrete offspring law on the non-negative integers, such as `Binomial` or `Geometric`: each member keeps its own offspring count, so the time and memory grow with the number of cases too. A [`ModelSpec`](@ref) adding `attributes`, `progression` and `observation` also qualifies, because these are drawn for each case once it exists and do not change who infects whom.

The run must be asked for in a way counts can answer: a finite `max_time` or a `max_cases`, with `max_generations` set to `nothing` or to a value other than its default, no `stopping_rules`, `condition` or `initial_cases`, no `index_types` other than a vector of types (the index cases are then listed by type), and no [`CommonRandomNumbers`](@ref). A run with `max_cases` ends at the end of the first day on which the total reaches it, and can hold more cases than that. A `max_generations` is applied exactly by tracking each day's cohort by generation, at a cost that grows with its value. A run with the default `max_generations`, or any other run not described here, is an individual simulation. [`containment_probability`](@ref)`(model, n; max_cases)` runs on counts for a model that qualifies, with the default `max_generations` applied.

## A finite population and vaccination

A finite population runs on counts as described above, and `state.susceptibles` holds the number left in each type. The contacts of the day fall at random on the people of each type, as under [`PerContactDepletion`](@ref), and the infections are shared among the cohorts in proportion to the contacts each made. The cost does not depend on the number of cases.

A [`ModelSpec`](@ref) with `GroupVaccination` (with `group_key = :type`) under [`CapacityConstrained`](@ref), and optionally `Scheduled` with a `start_time` and `end_time`, qualifies when the population is finite. The doses of the day interval `[t - 1, t)` reach the susceptibles before the contacts of day `t` are drawn, in the way [`simulate_by_generation`](@ref) gives them between generations: a dose reaches a susceptible of a type with probability equal to their share of the population of the type and protects with probability `efficacy`. This approximates the individual simulation, which offers doses to contacts as they are made. The two agree only when the budget reaches everyone. A specification this cannot apply runs as individuals.

```@example count_simulation
using EpiBranch, Distributions, StableRNGs, DataFrames, Statistics

generation_time = DiscreteNonParametric(1:4, [0.1, 0.4, 0.3, 0.2])
sized = BranchingProcess(NegBin(2.0, 0.5), generation_time; population = Population(5000))
campaign = CapacityConstrained(
    GroupVaccination(efficacy = 0.9, group_key = :type);
    budget_per_period = 100.0, period = 1.0
)
vaccinated = simulate(
    ModelSpec(sized; interventions = [campaign]); n_initial = 10, max_time = 60,
    max_cases = nothing, max_generations = nothing, rng = StableRNG(3)
)
(cases = vaccinated.cumulative_cases, susceptibles = only(vaccinated.susceptibles))
```

## Discretising a continuous generation time

A generation time measured in continuous time becomes a discrete one by double interval censoring: the infector's infection time is uniform within its day, and the infectee's is recorded as a whole day. With CensoredDistributions.jl loaded, `double_interval_censored` gives this distribution. Drop the mass on day 0 and renormalise, because a count-level step moves forward at least one day, then put the probabilities on the lags with `DiscreteNonParametric`:

```julia
using EpiBranch, Distributions, CensoredDistributions

continuous = Gamma(4.0, 1.2)
censored = double_interval_censored(continuous; interval = 1, upper = 15)
weights = [pdf(censored, lag) for lag in 1:15]
generation_time = DiscreteNonParametric(1:15, weights ./ sum(weights))
```

The individual simulation of a model with this generation time runs the same discrete model as the count-level path, which allows the two to be compared.

```@example count_simulation
model = BranchingProcess(NegBin(1.3, 0.5), generation_time)

state = simulate(
    model; n_initial = 10, max_time = 30, max_cases = nothing,
    max_generations = nothing, rng = StableRNG(1)
)
```

The state holds the count for each day, and the people are built when a table asks for them. The same seed gives the same line list however many times it is asked for.

```@example count_simulation
state.counts[1:10]
```

```@example count_simulation
first(linelist(state), 5)
```

## Continuing a run

`simulate(model, state; max_time)` continues the run. Each cohort keeps the infectiousness it was drawn with, so continuing gives the same result as running to the later time in one go.

```@example count_simulation
simulate(model, state; max_time = 40)
state
```

## A time-varying reproduction number

A [`RenewalOffspring`](@ref) with a vector generation time also qualifies, with `R(t)` read at the day of each infection:

```@example count_simulation
R(t) = t < 15 ? 1.4 : 0.8
renewal = BranchingProcess(Infectiousness(RenewalOffspring(R, [0.2, 0.5, 0.3]; k = 1.0)))
simulate(
    renewal; n_initial = 10, max_time = 40, max_cases = nothing,
    max_generations = nothing, rng = StableRNG(2)
)
```

The continuous form of `RenewalOffspring`, which thins contacts against `R_max`, does not qualify, because its generation time is not discrete on the day.
