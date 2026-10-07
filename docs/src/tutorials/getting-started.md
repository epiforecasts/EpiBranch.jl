# Getting started

This page runs one complete analysis: how often do isolation and contact
tracing contain an outbreak of a pathogen with R = 2.5 and strong
superspreading? Along the way it defines a model, simulates outbreaks, and
turns the results into line lists and summary tables. Times throughout are in
days. If the Julia syntax is unfamiliar, see
[Julia for R users](../julia-for-r-users.md).

## Defining a model

A branching process needs at least the distribution of the number of secondary
cases each case infects (the offspring distribution). Interventions act in
time, and to use them you also need a generation time distribution: the time
from the infection of an infector to the infection of the person it infects.

```@example gettingstarted
using EpiBranch
using Distributions
using StableRNGs

model = BranchingProcess(
    NegBin(2.5, 0.16),         # R = 2.5, dispersion k = 0.16
    LogNormal(1.6, 0.5)        # generation time, in days
)
```

[`NegBin`](@ref) is the negative binomial written the epidemiological way, with
mean R and dispersion k. Smaller k means more variation between cases: with
k = 0.16 most cases infect nobody and a few infect many. The generation time is
measured between infections; the serial interval, between symptom onsets, is a
different quantity.

`LogNormal(μ, σ)`, from the Distributions package, takes the mean and standard
deviation of the *log* of the delay: `LogNormal(1.6, 0.5)` has a median of
about 5 days and a mean of about 5.6 days. `mean` and `std` give the values on
the natural scale:

```@example gettingstarted
gt = LogNormal(1.6, 0.5)
(mean = mean(gt), sd = std(gt))
```

To start from a mean `m` and standard deviation `s` in days, use
`σ = sqrt(log(1 + s^2 / m^2))` and `μ = log(m) - σ^2 / 2`.

This is already a complete model. You can add three more parts to it with a
[`ModelSpec`](@ref), as the sections below do: characteristics of the people
(such as their incubation period), interventions (such as isolation), and how
cases are observed or reported. Only the transmission part is required.

## Simulating one outbreak

```@example gettingstarted
rng = StableRNG(3)  # a fixed random seed, like set.seed() in R
outbreak = simulate(model;
    max_cases = 500,
    rng = rng,
)
println("Cases: $(outbreak.cumulative_cases), Extinct: $(outbreak.extinct)")
```

The simulation starts from one index case and stops when transmission dies out
(`Extinct: true`) or the outbreak reaches `max_cases`. An outbreak stopped at
the cap is recorded as not extinct, and because the last generation is
completed before stopping, its case count can go past 500. Other limits are a number of generations (`max_generations`, 100 by
default) and a time (`max_time`, in days).

The result holds one record per person (every case, and every potential
secondary case that was not infected), plus summary information such as the
total number of cases and whether the outbreak died out.

## Adding symptom onset

Isolation on symptom onset needs to know when people develop symptoms. Give the
cases an incubation period (infection to onset) with
[`clinical_presentation`](@ref), passed to the model as `attributes`:

```@example gettingstarted
spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
)

rng = StableRNG(42)
outbreak = simulate(spec; max_cases = 500, rng = rng)

# The index case: infected at day 0, with an onset some days later
ind = outbreak.individuals[1]
println("Infection: day $(round(ind.infection_time, digits=1)), Onset: day $(round(onset_time(ind), digits=1))")
```

Times are in days from the infection of the first index case.

## Adding interventions

Interventions are added to the model in the same way, and `simulate` then
applies them:

```@example gettingstarted
# Symptomatic cases isolate on average 2 days after onset, for 7 days
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)
# Each contact of an isolated case is traced with probability 0.5, on average
# 1.5 days after the case isolates, and quarantined for 7 days
ct = ContactTracing(
    probability = 0.5, isolation_to_trace_delay = Exponential(1.5),
    action = Quarantine(duration = 7.0)
)

spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    interventions = [iso, ct],
)

rng = StableRNG(18)
outbreak = simulate(spec; max_cases = 500, rng = rng)
println("Cases: $(outbreak.cumulative_cases), Extinct: $(outbreak.extinct)")
println("Isolated or quarantined: $(count(is_isolated, outbreak.individuals))")
println("Traced: $(count(is_traced, outbreak.individuals))")
```

`Exponential(θ)` has mean θ, here in days. The `action` says what happens to a
traced contact: here they are quarantined, which stops them transmitting from
the time they are traced, even before they have symptoms. `duration` sets how
long isolation and quarantine last. When it ends the person is released, and
one who is still infectious can transmit again. Use `duration = Inf` for
isolation or quarantine that is never lifted.
`count(is_isolated, outbreak.individuals)` counts the people who were isolated
or quarantined at some point, including quarantined contacts who were never
infected. `Extinct` shows whether the measures ended transmission in this
outbreak before it reached the cap.

## Estimating the containment probability

One outbreak says little; the question is how often the measures work. Simulate
500 outbreaks:

```@example gettingstarted
rng = StableRNG(42)
results = simulate(spec, 500; max_cases = 5000, rng = rng)
println("Containment probability: $(round(containment_probability(results), digits=3))")
```

The containment probability is the proportion of simulated outbreaks in which
transmission stopped before reaching 5000 cases (or 100 generations). With a
pathogen this overdispersed, many outbreaks die out by chance even without
control. Compare it with the extinction probability without interventions,
calculated under [Exact results](@ref) below.

## Potential and actual secondary cases

The number drawn from the offspring distribution is the number of *potential*
secondary cases: the people a case would infect if nothing intervened. Without
interventions all of them are infected, and the mean of `NegBin(2.5, 0.16)` is
R. Isolation and quarantine prevent some of these infections. Fewer people are
then infected than were drawn, and the reproduction number realised under
control is below 2.5. The people whose infection was prevented stay in
the results as contacts who were not infected:

```@example gettingstarted
n_total = length(outbreak.individuals)
n_infected = count(is_infected, outbreak.individuals)
println("People in the simulation: $n_total")
println("Infected (cases, including the index case): $n_infected")
println("Potential secondary cases not infected: $(n_total - n_infected)")
```

The last line is the number of infections the interventions prevented in this
outbreak. These people are not infected and the simulation draws no contacts
for them. See [Isolation and contact tracing](interventions.md) for more.

## Tables of results

The simulation can be turned into data frames, much as in R.

```@example gettingstarted
using DataFrames, Dates

# Line list: one row per case
ll = linelist(outbreak; reference_date = Date(2024, 1, 1))
println("Line list: $(nrow(ll)) rows, $(ncol(ll)) columns")
first(ll, 3)  # like head(ll, 3) in R
```

Simulated times are in days from the start of the outbreak; `reference_date`
sets the calendar date of day 0 and gives the line list dates, as in
surveillance data. When isolation or quarantine has a finite duration, as
here, the line list also gives the release date in `date_isolation_release`.

```@example gettingstarted
# Contacts table: one row per infector and potential secondary case,
# with whether that person was infected
ct_df = contacts(outbreak; reference_date = Date(2024, 1, 1))
println("Contacts: $(nrow(ct_df)) ($(count(ct_df.infected)) infected)")
```

A transmission chain is all the cases that descend from one index case. Its
size is the number of cases in it, including the index case, and its length is
the number of generations after the index case:

```@example gettingstarted
cs = chain_statistics(outbreak)
cs
```

[`generation_R`](@ref) divides the number of cases in each generation by the
number in the generation before, a rough within-simulation measure of how
transmission falls as the measures take effect. It is not the time-varying
reproduction number Rt estimated from incidence data.

```@example gettingstarted
r_df = generation_R(outbreak)
first(r_df, 5)
```

With few cases per generation and strong superspreading the ratio jumps about,
but in a contained outbreak it settles below 1 in the later generations.

## Exact results

Some quantities can be calculated exactly from the offspring distribution,
without simulation:

```@example gettingstarted
println("P(extinction): $(round(extinction_probability(spec), digits=3))")
println("P(epidemic):   $(round(epidemic_probability(spec), digits=3))")
println("Top 20% cause $(round(proportion_transmission(spec; prop_cases=0.2) * 100, digits=1))% of transmission")
```

!!! warning "These ignore interventions"
    `extinction_probability`, `epidemic_probability` and
    `proportion_transmission` use only the offspring distribution of the model.
    The isolation and tracing in `spec` are not included: the extinction
    probability printed here is the chance that an outbreak dies out without
    control. Compare it with the containment probability simulated above to see
    what the interventions add. For a closed-form containment probability under
    simple control, see [`probability_contain`](@ref) in
    [Analytical functions](analytical.md).

The first line is the probability that a single introduction dies out by
chance. The last line is the share of all transmission caused by the 20% of
cases that transmit most; with k = 0.16 it is a large majority.

## Outbreaks of a given size

To study outbreaks of a particular size, for example to compare with an
observed cluster, keep only simulated outbreaks whose final size is in range:

```@example gettingstarted
rng = StableRNG(42)
outbreak = simulate(spec;
    condition = 50:100,  # every whole number from 50 to 100
    max_cases = 200,
    rng = rng,
)
println("Outbreak size: $(outbreak.cumulative_cases) (target: 50-100)")
```

`simulate` reruns the outbreak until one ends with between 50 and 100 cases,
discarding the rest, and returns that one.

!!! note "Rare sizes take many attempts"
    If outbreaks of the requested size are rare, this needs many simulations.
    It gives up with an error after `max_attempts` (10,000 by default). Keep
    `max_cases` above the top of the range: an outbreak stopped at the cap does
    not have a final size.

## Next steps

- Interventions:
  - [Isolation and contact tracing](interventions.md): isolating cases and
    tracing and quarantining their contacts
  - [Vaccination](vaccination.md): ring, mass and group vaccination, and
    protecting contacts already exposed
- [Clinical transitions](transitions.md): clinical progression from symptom
  onset to reporting, admission, and recovery or death
- Transmission models:
  - [Multi-type models](multi-type.md): age-structured and heterogeneous transmission
  - [Network models](networks.md): transmission over a fixed contact network
  - [Household models](households.md): transmission within and between households
  - [Homogeneous models](homogeneous.md): a closed, well-mixed population
  - [Covariates and time-varying transmission](covariate-transmission.md):
    transmission that depends on who is in contact, the date, or what has
    happened so far in the outbreak (for example an intervention starting)
- [Line lists and contacts](linelist.md): generating epidemiological data
- Analysis:
  - [Chain statistics](chains.md): chain size and length, and their likelihoods
  - [Analytical functions](analytical.md): exact results without simulation
  - [Inference](inference.md): estimating parameters from outbreak data
- Extending EpiBranch:
  - [Extending EpiBranch](extending.md): which tool fits what you want to
    model, and changes that need no new code structure
  - [Writing an intervention](writing-interventions.md): control measures and
    clinical events of your own
  - [New transmission structures](new-structures.md): contact structures,
    routes and transmission models of your own
  - [Extension reference](extending-reference.md): the details an extension
    has to respect
