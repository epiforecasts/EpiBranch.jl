# Simulating daily counts without individuals

A branching process in which nothing about a person feeds back on transmission does not need each person simulated to give the right daily counts. [`simulate`](@ref) runs such a model on a count-level path: it draws the number of people infected on each day from the cohorts of earlier days, at a cost that grows with the number of days, whatever the number of cases (a Poisson lognormal or Poisson Weibull offspring law is the exception, see below). The result is a [`CountSimulationState`](@ref), which builds the people, and so the line list, only when they are asked for.

## When a model runs on counts

The model must have a generation time that is a discrete distribution on whole days, starting at day 1 and ending at a largest value. Nothing about individuals may act on transmission: there are no interventions and the population is unlimited. The offspring law is a `Poisson`, a negative binomial (`NegBin`), a [`PoissonLognormal`](@ref) or [`PoissonWeibull`](@ref), or a [`RenewalOffspring`](@ref) with a vector generation time. A negative binomial cohort needs one gamma draw for its total infectiousness. A Poisson lognormal or Poisson Weibull cohort needs one draw for each member, and the state keeps these draws to share out the infections of the cohort when the people are built. For these two laws the time and memory grow with the number of cases as well as the number of days. A [`ModelSpec`](@ref) adding `attributes`, `progression` and `observation` also qualifies, because these are drawn for each case once it exists and do not change who infects whom.

The run must be asked for in a way counts can answer: a `max_time`, with `max_cases` and `max_generations` set to `nothing`, no `stopping_rules`, `condition`, `index_types` or `initial_cases`, and no [`CommonRandomNumbers`](@ref). Every other run is an individual simulation, as before.

`simulate(model, n)` takes the same keywords and returns a vector of `n` count-level runs. `containment_probability(model, n)` and the vector methods of `containment_probability` read the `extinct` flag of each run. A run is contained when transmission has died out by `max_time`.

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
using EpiBranch, Distributions, StableRNGs, DataFrames, Statistics

generation_time = DiscreteNonParametric(1:4, [0.1, 0.4, 0.3, 0.2])
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
