# Line lists and contacts

What would the surveillance data from a simulated outbreak look like?
[`linelist`](@ref) turns a simulation into a line list: a table (a
DataFrame, the Julia counterpart of an R data.frame) with one row per case,
in the format of the simulist R package. [`contacts`](@ref) gives the
matching table of contacts, infected or not.

Every line list has the columns `id`, `parent_id` (the infector's `id`),
`generation`, `chain_id` and `date_infection`. Anything else the simulation
records about a case, such as symptom onset, age or outcome, becomes a column
as well. A recorded time whose name ends in `_time` becomes a date column, so
`onset_time` appears as `date_onset`. A model with isolation or quarantine
adds `isolated` and `date_isolation`, and when they last a finite time (a
finite `duration`) also `date_isolation_release`, the date the person was
released. To get a new column, record that value on
each case during the simulation, for example as a population characteristic
(see the [clinical transitions tutorial](transitions.md)).

## Line list

The model below has a negative binomial offspring distribution with R = 1.5 and
dispersion k = 0.5 (`NegBin(R, k)`; smaller k means more superspreading) and a
log-normal generation time. All delays are in days. `LogNormal(μ, σ)` takes the
mean and standard deviation of the log: `LogNormal(1.5, 0.5)` is an
incubation period with a median of about 4.5 days. `Exponential(θ)` has mean θ.

After symptom onset each case is reported (mean 3 days after onset), admitted
to hospital with probability 0.2, and is due to die with probability 0.05.
Every case also gets a recovery time. Death and recovery are competing
outcomes: whichever comes first ends the case's course.

```@example linelist
using EpiBranch
using Distributions
using DataFrames
using Dates
using StableRNGs

attrs = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

progression = [
    Reporting(delay = Exponential(3.0)),
    Hospitalisation(delay = Exponential(5.0), probability = 0.2),
    Death(delay = Exponential(14.0), probability = 0.05),
    Recovery(delay = Exponential(14.0)),
]

# Secondary cases: NegBin(R = 1.5, k = 0.5); generation time: LogNormal (days)
model = ModelSpec(BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5));
    progression = progression, attributes = attrs)

rng = StableRNG(42)  # a fixed seed makes the results reproducible
state = simulate(model; condition = 50:200, max_cases = 200, rng = rng)

ll = linelist(state; reference_date = Date(2024, 1, 1))
first(ll, 5)
```

`condition = 50:200` repeats the simulation until it produces an outbreak of
between 50 and 200 cases (both ends included), and `max_cases` stops an outbreak
that keeps growing. Conditioning on size mimics an observed outbreak of known
size, but the outbreaks you get are a selected subset: their R and timing are
not representative of all outbreaks the model produces. `reference_date` is the
calendar date of time 0, from which all dates are counted.

A column appears only if the simulation recorded that information. Drop the
`Hospitalisation` step and `date_admission` disappears. Drop
`clinical_presentation` and there is no symptom onset. Then `date_onset`,
`date_reporting`, `date_admission`, `date_outcome` and `outcome` all disappear:
the reporting, admission and outcome delays are measured from onset.

## Demographics

[`demographics`](@ref) assigns each case an age (in years) and a sex when it is
created. They appear in the line list as `age` and `sex` columns:

```@example linelist
attrs_demo = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15), prob_female = 0.55),
]

model = ModelSpec(BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5));
    progression = progression, attributes = attrs_demo)

rng = StableRNG(42)
state = simulate(model; condition = 50:200, max_cases = 200, rng = rng)

ll = linelist(state; reference_date = Date(2024, 1, 1))
println("Age range: $(minimum(ll.age)) - $(maximum(ll.age))")
println("Female: $(round(count(==("female"), ll.sex) / nrow(ll) * 100, digits=1))%")
```

The proportion female should be close to the 55% asked for.

## Age-specific case fatality risk

To make the case fatality risk depend on age, give `Death` a function that
returns each case's probability of death. The function below uses risks of
0.1% under 15, 1% from 15 to 64 and 15% at 65 and over:

```@example linelist
attrs_demo = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
]

function cfr_by_age(rng, ind)
    age = ind.state[:age]
    if age < 15
        return 0.001
    elseif age < 65
        return 0.01
    else
        return 0.15
    end
end

age_stratified = [
    Death(delay = Exponential(14.0), probability = cfr_by_age),
    Recovery(delay = Exponential(14.0)),
]

model = ModelSpec(BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5));
    progression = age_stratified, attributes = attrs_demo)

rng = StableRNG(42)
state = simulate(model; condition = 100:500, max_cases = 500, rng = rng)

ll = linelist(state; reference_date = Date(2024, 1, 1))

# Cases and deaths in each age band
for (lo, hi) in [(0, 14), (15, 64), (65, 90)]
    group = filter(row -> lo <= row.age <= hi, ll)
    n_died = count(==("died"), group.outcome)
    pct = nrow(group) > 0 ? round(n_died / nrow(group) * 100, digits=1) : 0.0
    println("Age $lo-$hi: $(nrow(group)) cases, $n_died deaths ($pct%)")
end
```

EpiBranch calls the function with two arguments: the random number generator
`rng` and the case `ind`. `cfr_by_age` does not use `rng`, but it must accept
it. `ind.state[:age]` is the age that `demographics` gave the case.

The proportion dying rises with age, but in the oldest band it falls well
short of 15%.

!!! warning "The probability of death is not the case fatality risk"
    `probability` on `Death` is the chance that a case is due to die. Recovery
    still competes with it: a case due to die whose recovery time comes first
    recovers. Here death and recovery have the same delay distribution. About
    half the cases due to die therefore recover, and the proportion who die is
    about half of `probability`. For a fixed case fatality risk, use
    [`exclusive_probabilities`](@ref) to make death and recovery mutually
    exclusive, as shown in the [clinical transitions tutorial](transitions.md).

The same approach works for any characteristic you assign to cases, such as a
risk group or a comorbidity. See the
[clinical transitions tutorial](transitions.md) for the other steps a case can
go through.

## The whole population

`linelist` lists cases only by default. Pass `infected_only = false` to list
everyone in the population, as needed for a test-negative design, an attack
rate by exposure, or an exposed/unexposed comparison. This fits models that
simulate a fixed population from the start: random mixing
([`HomogeneousProcess`](@ref)), networks (`NetworkProcess`) and households
(`HouseholdProcess`).

The model below has 200 people mixing at random. `transmission_rate` is the
rate per day at which each infectious person makes infectious contacts, spread
evenly over the population, and each infected person recovers after an
exponentially distributed time with a mean of 5 days, which ends their
infectiousness:

```@example linelist
pool = ModelSpec(HomogeneousProcess(; transmission_rate = 0.6, population_size = 200);
    progression = [Transition(:recovered; from = :infection, delay = Exponential(5.0),
        terminal = true)],
    attributes = attrs)

pool_state = simulate(pool; n_initial = 2, rng = StableRNG(1))

pop = linelist(pool_state; reference_date = Date(2024, 1, 1), infected_only = false)
println("Population: $(nrow(pop)), infected: $(count(pop.infected))")
first(pop, 5)
```

In a [`BranchingProcess`](@ref) model the rows are the cases plus the contacts
they exposed who were not infected.

An uninfected row has `missing` for `date_infection` and for every date that
follows from an infection: `date_onset`, the reporting, admission and outcome
dates, and any date from your own `_time` values. Dates of events that can
happen to a person whether or not they are infected are kept:

- `date_trace`, when the contact was traced;
- `date_vaccination` and `date_immunity`;
- `date_isolation`, when the contact was quarantined on being traced, and
  `date_isolation_release`, when that quarantine ended. If a traced contact
  was later due to be isolated at what would have been their symptom onset,
  that isolation is not shown (they were never infected, so never had an
  onset): both columns give the earlier quarantine instead.

Columns that are not dates, such as `asymptomatic`, `traced` or `vaccinated`,
are shown unchanged.

## Contacts table

[`contacts`](@ref) returns one row per contact: the case (`from`), the
contact (`to`), whether the contact was infected (`infected`), and the
contact's generation and infection date:

```@example linelist
ct = contacts(state; reference_date = Date(2024, 1, 1))
println("Total: $(nrow(ct)), Infected: $(count(ct.infected)), Not infected: $(nrow(ct) - count(ct.infected))")
first(ct, 5)
```

Contacts who were not infected appear when something stopped transmission,
such as isolation, quarantine or vaccination. This model has no interventions:
every contact was infected.

!!! note "Household and network models"
    In the household and network models (`HouseholdProcess`, `NetworkProcess`,
    `RoutedNetwork`) the contacts table lists each pair of people who are in
    contact, and a vaccinated or otherwise protected pair appears exactly as an
    unprotected one would. It does not count how many separate contact events
    a pair had, or when they happened. If you need that count, add a
    [`ContactRecorder`](@ref) to the model; see
    [Recording every contact event](@ref "Recording every contact event") in
    the extending guide.
