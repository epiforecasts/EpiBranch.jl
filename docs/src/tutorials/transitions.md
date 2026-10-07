# Clinical transitions

What happens to each case after infection: when do symptoms start, when is the
case reported, is it admitted to hospital, and does it recover or die? Each
case follows a disease timeline of events, each happening a delay after an
earlier one. You describe the timeline as a list of events called
`progression` and give it to a [`ModelSpec`](@ref) alongside the transmission
process. Control measures (isolation, contact tracing, vaccination) go in the
model's `interventions`; the course of disease goes in its `progression`.

Each event takes its timing as either a `delay` or a `rate`. A `delay` is a
fixed number of days or a distribution of days. A `rate` r gives an
exponentially distributed waiting time with mean 1/r days, as in a
compartmental model; only the general [`Transition`](@ref) event below takes
a `rate`.

## Built-in events

The package has four ready-made events, each timed from symptom onset by
default:

- [`Reporting`](@ref): a symptomatic case is reported after a delay, with a
  probability that can be below 1.
- [`Hospitalisation`](@ref): a proportion of cases are admitted after a delay.
- [`Death`](@ref): a case dies after a delay, with a given probability.
- [`Recovery`](@ref): every symptomatic case gets a recovery time.

Death and recovery are final outcomes: each case ends with one of them (see
[Competing outcomes](@ref) below). Each event has its own delay distribution
and probability:

```@example transitions
using EpiBranch
using Distributions
using StableRNGs

# incubation period, infection to onset: mean about 5 days
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

progression = [
    Reporting(delay = LogNormal(1.0, 0.3)),                         # onset to report
    Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.2), # onset to admission
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),          # onset to death
    Recovery(delay = LogNormal(2.0, 0.4)),                           # onset to recovery
]

model = ModelSpec(
    BranchingProcess(Poisson(2.0), Exponential(5.0));  # R = 2, generation time mean 5 days
    progression = progression, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 200, rng = rng)

cases = linelist(state)
first(cases[:, [:id, :date_infection, :date_onset, :date_reporting,
    :date_admission, :outcome, :date_outcome]], 5)
```

`LogNormal(μ, σ)` takes the mean and standard deviation of the logarithm of
the delay in days, not of the delay itself: `LogNormal(1.0, 0.3)` has a mean
of about 2.8 days. `Exponential(θ)` has mean θ days. Each row of the line list
is a case, with the date of each event it reached and its final outcome.

!!! note "Events are drawn separately"
    Each event is drawn on its own. An admission date can therefore fall after
    the date of recovery. To tie one event to another, make its probability
    depend on the earlier event (see
    [Events that depend on an earlier event](@ref)) or time it from that
    event (see [Measuring delays from an earlier event](@ref)).

## Asymptomatic cases

Only cases with a symptom onset go through events timed from onset. Cases
without an onset, because they are asymptomatic or never observed, are
skipped. [`clinical_presentation`](@ref) makes a proportion of cases
asymptomatic with `prob_asymptomatic`, for example
`clinical_presentation(incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3)`.
The default of 0 gives every case an onset.

## Probabilities and delays that depend on the person

`probability` and `delay` also accept a function of the random number
generator and the person, `(rng, ind) -> value`.
`ind.state[:age]` is the person's age, set by [`demographics`](@ref). This
covers age-specific case fatality, delays that differ by risk group, and
reporting that differs between groups. `c ? a : b` is Julia for "if `c` then
`a`, otherwise `b`", like `ifelse` in R.

```@example transitions
attrs = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
]

# Probability of death among symptomatic cases: 30% for 80+, 2% otherwise.
death_age = Death(
    delay = LogNormal(2.5, 0.4),
    probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02,
)

# Onset to admission: 1 day for under-30s, 5 days otherwise.
hosp_age = Hospitalisation(
    delay = (rng, ind) -> ind.state[:age] < 30 ? 1.0 : 5.0,
    probability = 0.2,
)

progression = [hosp_age, death_age, Recovery(delay = LogNormal(2.0, 0.4))]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = attrs)

rng = StableRNG(42)
state = simulate(model; max_cases = 300, rng = rng)

n_died = count(ind -> ind.state[:outcome] == :died, state.individuals)
n_died_80plus = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:age] >= 80
end
println("Deaths: $n_died, of which 80+: $n_died_80plus")
```

People aged 80 and over are about one in nine of the population here, yet
they account for most of the few deaths. Age is one example: any
characteristic you give people through `attributes` (risk group,
comorbidity, vaccination status) can be used the same way.

## Events that depend on an earlier event

Some events happen only after another: only reported cases are admitted, only
tested cases are treated. Make the probability of the later event 0 for
everyone who has not had the earlier one. List the events in the order they
happen, because an event can only depend on events listed before it.

```@example transitions
# Only reported cases can be admitted; 20% of them are.
reported_hosp = Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0,
)

progression = [
    Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5),
    reported_hosp,
]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 200, rng = rng)

n_admitted = count(ind -> ind.state[:admitted], state.individuals)
n_admitted_unreported = count(ind -> ind.state[:admitted] && !ind.state[:reported],
    state.individuals)
println("Admitted: $n_admitted, of which not reported: $n_admitted_unreported")
```

No unreported case was admitted. `get(ind.state, :reported, false)` reads
whether the case was reported, treating it as not reported if the value is
missing. Conditions can be combined, for example admitting only cases that
were reported and are not vaccinated.

## Measuring delays from an earlier event

By default each built-in event is timed from symptom onset. The `from`
argument times it from another event instead. Suppose cases are reported
after a positive test, with the reporting delay measured from the test
and not from onset. The general [`Transition`](@ref) event describes the
test: `Transition(:tested; from = :onset, ...)` means that a case is tested
a delay after onset, here with probability 0.9 (the proportion of
symptomatic cases who get a positive test). It records whether the case was
tested as `:tested`, and when as `:tested_time`. Reporting is then timed from
the test with `from = :tested_time`:

```@example transitions
testing = Transition(:tested; from = :onset, delay = LogNormal(0.5, 0.3),
    probability = 0.9)
reporting_post_test = Reporting(delay = LogNormal(0.0, 0.2), from = :tested_time)

progression = [testing, reporting_post_test]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 100, rng = rng)

first(linelist(state)[:, [:id, :date_onset, :tested, :date_tested, :date_reporting]], 5)
```

Cases without a test have no report. In each row, the test date follows
onset and the report date follows the test.

!!! warning "`from` takes an event for `Transition` and a time for the others"
    `Transition` takes the name of the earlier event (`from = :onset`,
    `from = :tested`). `Reporting`, `Hospitalisation`, `Death` and `Recovery`
    take the name of its recorded time (`from = :onset_time`,
    `from = :tested_time`). A mix-up gives no error:
    `Transition(...; from = :tested_time)` never happens, and
    `Reporting(...; from = :tested)` reads the yes/no record as day 0 or
    day 1, so every case is reported as if timed from the start of the
    outbreak.

The general `Transition` covers any event in the timeline: it takes the name
of the event, `from` (the earlier event it is timed from, by default
infection), a `delay` or `rate`, a `probability`, and `terminal = true` for a
final outcome.

### Delays from infection

If you do not model symptom onset, time events from infection instead. With
`from = :infection` (the default for `Transition`), no
`clinical_presentation` is needed:

```julia
Transition(:reported; from = :infection, delay = LogNormal(2.0, 0.3))
```

The built-in events take `from` too, as a function of the person:
`Reporting(delay = LogNormal(2.0, 0.3), from = ind -> ind.infection_time)`.
Each event has its own `from`: a timeline can admit cases a delay after onset
and time their outcome from admission.

## Combining with multi-type models and population characteristics

The disease timeline is separate from the transmission model. A
[multi-type model](multi-type.md) with age or risk groups can have outcomes
that differ by type. A rule can look up the case's type, `ind.state[:type]`
(1 for the first type, 2 for the second, in the order of the next-generation
matrix), as well as its age:

```@example transitions
attrs_age = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
]

# Probability of death depends on both type and age.
death_type_age = Death(
    delay = LogNormal(1.5, 0.4),  # onset to death: mean about 5 days
    probability = (rng, ind) -> begin
        base = ind.state[:age] >= 65 ? 0.15 : 0.01
        ind.state[:type] == 1 ? 0.5 * base : base  # children: half the risk
    end,
)

# Onset to admission depends on type.
hosp_type = Hospitalisation(
    delay = (rng, ind) -> ind.state[:type] == 1 ? 1.0 : 3.0,
    probability = 0.2,
)

progression = [
    hosp_type,
    death_type_age,
    Recovery(delay = LogNormal(2.0, 0.4)),
]

# Two types, children and adults. Columns are infectors, rows infectees:
# a child causes 2.0 child and 0.8 adult cases, an adult 0.5 child and
# 1.5 adult cases. Secondary cases are negative binomial with dispersion k = 0.5.
multitype = ModelSpec(
    BranchingProcess(
        [2.0 0.5; 0.8 1.5],
        R -> NegBin(R, 0.5),
        LogNormal(1.6, 0.5),
        type_labels = ["children", "adults"]);
    progression = progression, attributes = attrs_age)

rng = StableRNG(3)
state = simulate(multitype; max_cases = 500, rng = rng)

n_died_kids = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:type] == 1
end
n_died_adults = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:type] == 2
end
println("Deaths. Children: $n_died_kids. Adults: $n_died_adults.")
```

Children and adults make up similar numbers of cases in this outbreak. The
halved risk among children shows as fewer deaths, although the counts are
small and noisy. Death here comes sooner after onset than in the first
example, which leaves fewer of those who would die to recover first (see
[Competing outcomes](@ref)).
Relative susceptibility and infectiousness can also vary between people,
through [`transmission_traits`](@ref), and outcome rules can use any of these
characteristics.

## Competing outcomes

Death and recovery compete: each case ends with whichever final outcome
happens first, the situation survival analysis calls competing risks. The
outcome is recorded as `:outcome`, and its time as `:outcome_time`.

This affects what the `probability` of `Death` means. It is the probability
that a case would die if it did not recover first. When recovery tends to
come sooner than death, as in the first example on this page, some of the
cases that would have died recover first. The proportion of cases dying is
then below the `probability` given.

Further final outcomes join the competition through the general `Transition`
with `terminal = true`. Here 10% of cases are lost to follow-up, a mean of
about 5 days after onset, unless they die or recover first:

```@example transitions
progression = [
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
    Recovery(delay = LogNormal(2.0, 0.4)),
    Transition(:lost; from = :onset, delay = LogNormal(1.5, 0.5),
        probability = 0.1, terminal = true),
]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 200, rng = rng)

outcomes = [ind.state[:outcome] for ind in state.individuals if haskey(ind.state, :outcome)]
println("Outcome counts: ",
    (died = count(==(:died), outcomes),
     recovered = count(==(:recovered), outcomes),
     lost = count(==(:lost), outcomes)))
```

Most cases recover. The deaths and losses to follow-up are fewer than 5% and
10% of cases, because recovery often comes first. The same approach gives
admission to intensive care after admission to hospital, or outcomes that
depend on treatment.

### An exact case fatality ratio

To fix the proportion of symptomatic cases who die, the outcomes must be
mutually exclusive. If death and recovery each happened independently, with
probabilities 0.05 and 0.95, some cases would get both and some neither. The
proportion dying would then differ from 5%. [`exclusive_probabilities`](@ref)
makes the outcomes mutually exclusive: each case has exactly one of them, with
the probabilities given. `Recovery` always happens and has no probability to set.
The recovery outcome is therefore a general `Transition` with its own
probability:

```@example transitions
death_p, recovered_p = exclusive_probabilities([0.05, 0.95])
exact_cfr = [
    Death(delay = LogNormal(2.5, 0.4), probability = death_p),
    Transition(:recovered, from = :onset, delay = LogNormal(2.0, 0.4),
        probability = recovered_p, terminal = true),
]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = exact_cfr, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 300, rng = rng)
symptomatic = [ind for ind in state.individuals if !isnan(onset_time(ind))]
n_missing = count(ind -> !haskey(ind.state, :outcome), symptomatic)
n_died = count(ind -> get(ind.state, :outcome, nothing) == :died, symptomatic)
println("Symptomatic cases: ", length(symptomatic), ", missing an outcome: ", n_missing)
println("Died: ", n_died, " of ", length(symptomatic))
```

Every symptomatic case has an outcome. The proportion who died is close to 5%,
with the difference due to chance.

`ModelSpec` warns if no final outcome is certain to happen, because some cases
would then end without an outcome.

## Likelihood of the clinical timeline

To fit reporting or outcome delays to observed cases, you need the likelihood
of each case's timeline. [`progression_loglik`](@ref) gives the log-likelihood
of the simulated cases' timelines under the `progression`: for each event, the
probability that it did or did not happen and, when it happened, the
probability density of its delay. It takes the simulated outbreak that
`simulate` returned:

```@example transitions
progression = [
    Reporting(delay = LogNormal(1.0, 0.3), probability = 0.7),
    Recovery(delay = LogNormal(2.0, 0.4)),
]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = clinical)

rng = StableRNG(42)
state = simulate(model; max_cases = 200, rng = rng)
progression_loglik(model, state)
```

The result is a log-likelihood: comparing it across parameter values shows
which fit the timelines better, with higher values fitting better.
[`pairwise_surv_loglik`](@ref) gives the likelihood of who infected whom and
when. The two added together are the log-likelihood of the whole outbreak:
infection times, the transmission tree and clinical timelines, including the
parts that are not directly observed. Cases that never reached the event an
event is timed from (an asymptomatic case for an event timed from onset)
contribute nothing.

!!! warning "Delays that depend on the person cannot be evaluated"
    A delay given as a function `(rng, ind) -> ...` can be simulated but has
    no density, and `progression_loglik` throws an error on it. To use the
    likelihood, give each delay as a distribution or a fixed number of days.

To write your own kind of event, with its own likelihood, see
[Extending EpiBranch](extending.md).
