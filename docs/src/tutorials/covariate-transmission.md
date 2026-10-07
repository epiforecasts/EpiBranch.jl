# Covariates and time-varying transmission

How does transmission change with who is in contact, with the time of year, or
with a policy introduced during the outbreak? In [network](@ref "Network models")
and [household](@ref "Household models") models, transmission between each pair
of people is described by the contact interval. This page shows how to make it
depend on the two people, on the calendar date, and on events during the
outbreak, both in simulation and in the pairwise likelihood used for fitting.

The contact interval is the waiting time from the start of the infector's
infectious period until the infector would infect a given contact, if that
contact is still susceptible and the infector is still infectious. A shorter
contact interval means more transmission. It differs from the generation time
in two ways. The generation time runs from the infector's infection and
includes the latent period. It is also observed only for contacts that were
actually infected: a contact interval that ends after the infector has
recovered, or after the contact has already been infected by someone else,
never becomes a generation time. This is the pairwise survival approach to
transmission: each pair has a hazard of infectious contact over time.

When the contact interval depends only on fixed characteristics of the two
people, give a function that takes the infector and the susceptible and returns
a distribution, `(infector, susceptible) -> Distribution`, as the [households
tutorial](@ref "The pairwise likelihood") shows. For anything more,
[`PairKernel`](@ref) describes the rule for each pair (the transmission kernel,
called the rule below): a contact interval that depends on when the infector
was infected, a contact rate that changes on the calendar, and characteristics
or events recorded on each person during the outbreak.

## Covariates and the infector's infection date

Here each person has a fixed covariate, such as a relative risk score. People
are numbered from 1, and `covariates[3]` is person 3's value. The contact
interval is exponential, with a mean that depends on the susceptible's
covariate and on the day the infector was infected:

```math
\text{mean contact interval} = \exp(0.1 \times \text{infector's infection day} + 0.2 \times \text{susceptible's covariate})
```

Positive coefficients lengthen the contact interval, which reduces
transmission: here, people with higher covariate values are less likely to be
infected, and infectors infected later in the outbreak transmit more slowly.

`PairKernel` takes a function of `context`, which holds information about the
pair: `context.infector` and `context.susceptible` are the two people's
numbers, and `context.infector_infection_time` is the day the infector was
infected. The network here has three people, all connected:
`adjacency[1] = [2, 3]` says person 1 is in contact with persons 2 and 3.

```@example contextual
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

covariates = [0.5, 1.0, 1.5]
kernel = PairKernel(context -> Exponential(exp(
    0.1 * context.infector_infection_time +
    0.2 * covariates[context.susceptible])))

# infectious 0.75 days after infection, recovered 5 days later
progression = [Transition(:infectious; delay = 0.75),
    Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
network_model = ModelSpec(NetworkProcess(adjacency, kernel); progression)
network_state = simulate(network_model; rng = Xoshiro(233))
network_data = network_infections(network_state, network_model)
loglikelihood(network_data, network_model)
```

`ModelSpec(...; progression)` is Julia shorthand for
`progression = progression`. [`network_infections`](@ref) extracts who was
infected and when, the data the likelihood needs. The printed number is the
log-likelihood of that outbreak under the model; fitting compares it across
parameter values.

The same rule works for a household model and its likelihood:

```@example contextual
household_model = ModelSpec(HouseholdProcess([3], kernel); progression)
household_state = simulate(household_model; rng = Xoshiro(233))
household_data = household_infections(household_state, household_model)
loglikelihood(household_data, household_model)
```

`context.infector_infection_time` is the day of infection, even when a latent
period means the infectious period starts later. The contact interval is
still measured from the start of the infectious period.

## Fitting

To estimate parameters such as the coefficients above, evaluate the likelihood
at candidate values and maximise it, or sample from the posterior.
[`compile_contact_pairs`](@ref) prepares the contact structure once, so that
repeated evaluations are fast. Each evaluation reads the infection times from
the data it is given. Unobserved infection times can therefore be estimated
alongside the other parameters:

```@example contextual
layout = compile_contact_pairs(network_data)
pairwise_surv_loglik(kernel, network_data, layout)
```

The result matches the network log-likelihood printed in the first example. Gradients of the likelihood with
respect to the parameters and the infection times are available for
gradient-based fitting, such as Hamiltonian Monte Carlo in Turing, provided the
contact interval distribution supports them.

## A policy starting on a calendar day

To model a policy that changes transmission from a fixed date, such as a
lockdown, give the rule a `calendar`. Transmission is multiplied by a factor
that depends on the calendar day. [`Steps`](@ref) gives a factor that changes
at given days: `Steps([3.0], [0.4, 0.1])` is 0.4 before day 3 and 0.1 from
day 3, one more value than there are change days.

Suppose the rate of infectious contact is 0.4 per day before day 3 and 0.1
per day afterwards, a 75% reduction. The rule's own distribution,
`Exponential(1.0)`, has a constant rate of 1 per day, and the calendar factor
sets the actual rate:

```@example calendar
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

policy_day = 3.0
before_rate = 0.4
after_rate = 0.1
kernel = PairKernel(context -> Exponential(1.0);
    calendar = Steps([policy_day], [before_rate, after_rate]))
```

A policy can change transmission partway through someone's infectious period.
For a person infectious from day 2, the cumulative hazard of infecting a given
contact by day 4 is 0.4 × 1 day (days 2 to 3) + 0.1 × 1 day (days 3 to 4) =
0.5. Simulation and the likelihood both compute this exactly. Here is a
network simulation with a latent period drawn uniformly between 0.4 and 0.8
days, and its likelihood:

```@example calendar
progression = [Transition(:infectious; delay = Uniform(0.4, 0.8)),
    Transition(:recovered; from = :infectious, delay = 4.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
model = ModelSpec(NetworkProcess(adjacency, kernel); progression)
state = simulate(model; rng = Xoshiro(234))
data = network_infections(state, model)
loglikelihood(data, model)
```

For a household model, replace `NetworkProcess(adjacency, kernel)` with
`HouseholdProcess([3], kernel)` and use `household_infections`. Both
likelihoods take the start of each infectious period as given in the data.
Unobserved latent periods must be estimated as parameters; the likelihood reads
their current values at each evaluation.

If each contact has its own distribution, set in a list per network edge, a
calendar cannot be added to that list. Look up the pair's distribution inside
a `PairKernel` instead, and add the calendar there. `findfirst(==(j), v)`
finds the position of `j` in `v`, like `match(j, v)` in R:

```julia
edges = [[Exponential(1.0), Exponential(2.0)] for _ in adjacency]
PairKernel(context -> edges[context.infector][findfirst(==(context.susceptible),
        adjacency[context.infector])];
    calendar = Steps([3.0], [1.0, 0.25]))   # rate falls to a quarter on day 3
```

To give each pair its own policy date, for example the lockdown date in the
susceptible's region, return both the distribution and the calendar from the
function, as `(profile = ..., calendar = ...)`. This replaces the rule's
shared `calendar` for that pair. Here the policy day is read from the
susceptible's covariate. The contact interval is a Weibull distribution
(`Weibull(shape, scale)`) whose scale grows with the infector's day of
infection:

```@example calendar
covariates = [1.0, 2.0, 4.0]
covariate_kernel = PairKernel(context ->
    (profile = Weibull(2.0, exp(1.0 + 0.05 * context.infector_infection_time)),
     calendar = Steps([covariates[context.susceptible]], [1.0, 0.25])))
```

The rates and change days of a schedule, and the start of each infectious
period, can be estimated: the likelihood has gradients with respect to them,
provided the contact interval distribution supports them.

## A seasonal contact rate

A schedule need not be a step function. [`Seasonal`](@ref) is the smooth
schedule the package provides, rising and falling once a year around a peak
day. A seasonal network or household model needs only this one extra keyword:

```@example calendar
seasonal_kernel = PairKernel(context -> Exponential(4.0);
    calendar = Seasonal(amplitude = 0.6, peak_day = 30.0))
```

The cumulative hazard over two days from the start of an infectious
period on day 100 is the integral
of the multiplier times the profile's constant rate of 0.25 per day:

```@example calendar
seasonal_interval = EpiBranch.pair_kernel(seasonal_kernel, 1, 2, 0.0, 100.0)
exact = 0.25 * (2 + 0.6 * 365.2425 / 2π *
    (sin(2π * (102 - 30) / 365.2425) - sin(2π * (100 - 30) / 365.2425)))
(EpiBranch.cumhazard(seasonal_interval, 2.0), exact)
```

Simulation draws contact intervals by inverting that same integrated hazard,
which keeps the network simulation consistent with its likelihood:

```@example calendar
seasonal_model = ModelSpec(NetworkProcess(adjacency, seasonal_kernel); progression)
seasonal_state = simulate(seasonal_model; rng = Xoshiro(235))
loglikelihood(network_infections(seasonal_state, seasonal_model), seasonal_model)
```

A plain callable `t -> multiplier` is read as a smooth schedule too, for
seasonal forcing of any other shape:

```@example calendar
seasonal(t) = 1 + 0.6 * cos(2π * (t - 30) / 365.2425)
plain_kernel = PairKernel(context -> Exponential(4.0); calendar = seasonal)
EpiBranch.cumhazard(EpiBranch.pair_kernel(plain_kernel, 1, 2, 0.0, 100.0), 2.0) ==
    EpiBranch.cumhazard(seasonal_interval, 2.0)
```

[`Seasonal`](@ref) and a plain callable cover most seasonal forcing. Beyond
either, any type with its own
[`calendar_multiplier`](@ref EpiBranch.calendar_multiplier) method can still be
a `calendar`, and one that declares itself smooth through
[`calendar_shape`](@ref EpiBranch.calendar_shape) is integrated by quadrature
instead of segment by segment, exactly as `Seasonal` itself is. Here is a
schedule with two peaks a year instead of one, written in three steps, each a
few lines of Julia:

```math
\text{multiplier on day } t = 1 + \text{amplitude} \cos\left(4\pi \frac{t - \text{first\_peak}}{365}\right)
```

1. `struct TwoPeakSeasonal ... end` defines a new kind of object that holds
   the schedule's parameters, the amplitude and the first peak's day, like a
   named list in R with fixed elements. `{T <: Real}` lets `amplitude` hold
   any kind of number, which gradient-based fitting needs.
2. `EpiBranch.calendar_multiplier(s::TwoPeakSeasonal, t) = ...` tells
   EpiBranch how to compute the multiplier on day `t` for a `TwoPeakSeasonal`
   schedule.
3. `EpiBranch.calendar_shape(::TwoPeakSeasonal) = EpiBranch.SmoothCalendar()`
   says that the multiplier changes continuously, without steps. The package
   then integrates it numerically.

```julia
struct TwoPeakSeasonal{T <: Real}
    amplitude::T
    first_peak::Float64
end
function EpiBranch.calendar_multiplier(s::TwoPeakSeasonal, t)
    return 1 + s.amplitude * cos(4π * (t - s.first_peak) / 365)
end
EpiBranch.calendar_shape(::TwoPeakSeasonal) = EpiBranch.SmoothCalendar()
```

With a unit-rate profile, `Exponential(1.0)`, the multiplier is the pair's
hazard on the calendar. Any calendar-time hazard can therefore be written as a
schedule. `Seasonal`'s fields are typed so that the likelihood can be
differentiated through them, as with `Steps`.

## Characteristics recorded during simulation

Transmission can depend on characteristics each person is given during
simulation. Here each person has a random contact level, drawn uniformly
between 0.5 and 1.5, and the mean contact interval between two people is the
sum of their two levels.

The `state` argument is a function that picks out what the rule needs from each
person. The rule then takes three arguments: the pair's `context`, the
infector's record (here called `source`) and the susceptible's record (`target`).
`watches` lists the person-level variables the `state` function reads; the
next section explains why it matters. `::Float64` asserts that the value is a
`Float64` (a decimal number); an integer or other number type there gives an
error.

```@example stateful
using EpiBranch, EpiNetwork, Distributions, Random

# each person gets a contact level between 0.5 and 1.5
attributes = (rng, ind) -> (ind.state[:contact_scale] = rand(rng, Uniform(0.5, 1.5)))
# the record the rule reads for each person
project(ind) = (scale = ind.state[:contact_scale]::Float64,)
# source = infector, target = susceptible
contact_law(context, source, target) = Exponential(source.scale + target.scale)
kernel = PairKernel(contact_law; state = project, watches = (:contact_scale,))
adjacency = [[2, 3], [1, 3], [1, 2]]
progression = [Transition(:recovered; delay = 5.0, terminal = true)]
model = ModelSpec(NetworkProcess(adjacency, kernel); attributes, progression)
state = simulate(model; initial_cases = [1], rng = Xoshiro(235))
```

### Transmission that changes during the infectious period

A case's transmission can change during its infectious period: a post-exposure
vaccine dose takes effect in a contact, or symptoms begin and the case is
isolated. The simulation then has to update the timing of that case's future
contacts. `watches` lists the person-level variables, by their names in
`ind.state`, whose changes should trigger that update. In the example above,
that is `:contact_scale`, the one variable the `state` function reads.

!!! warning "List every variable the `state` function reads in `watches`"
    If a variable the `state` function reads is missing from `watches`, a
    change to it during the outbreak does not update the contacts already
    timed. The simulation then silently uses the old transmission rate. There
    is no error or warning. List every variable the `state` function reads,
    even one that nothing in your current model changes. When in doubt,
    include it. A `state` function that reads none, such as one that looks
    a value up by the person's number, takes `watches = ()`.

The next rule shows `watches` at work. Its `state` function reads the
vaccination date, which [`RingVaccination`](@ref) records (`Inf` means never
vaccinated), and the onset date, which [`clinical_presentation`](@ref) records.
Both are listed in `watches`. The rule gives a vaccinated susceptible a mean
contact interval of 4 days against 1.5, but it checks only whether a
vaccination date exists, so it is not a model of vaccine protection to fit:
the [checklist](@ref "Checklist for time-varying transmission") below explains
why. [`EpiBranch.watched_records`](@ref) shows what a
rule watches:

```@example stateful
dosed_kernel = PairKernel(
    # vaccinated susceptible: longer time to infectious contact
    (context, source, target) -> Exponential(isfinite(target.dosed) ? 4.0 : 1.5);
    state = ind -> (
        dosed = get(ind.state, :vaccination_time, Inf)::Float64,
        onset = get(ind.state, :onset_time, NaN),
    ),
    watches = (:vaccination_time, :onset_time)
)
EpiBranch.watched_records(dosed_kernel)
```

This rule sets each infector's mean contact interval to a fixed value looked
up by their number, `ind.id` (a larger value means slower transmission), and
reads nothing that can change:

```@example stateful
scale_by_id = [1.0, 2.0, 0.5]
by_id_kernel = PairKernel(
    (context, source, target) -> Exponential(source.scale);
    state = ind -> (scale = scale_by_id[ind.id],), watches = ()
)
EpiBranch.watched_records(by_id_kernel)
```

### The likelihood with recorded characteristics

After simulation, [`record_kernel`](@ref) extracts each person's record, and
the same rule then gives the likelihood:

```@example stateful
data = network_infections(state, model)
recorded = record_kernel(kernel, state)
layout = compile_contact_pairs(data)
pairwise_surv_loglik(recorded, data, layout)
```

For observed data, build `PairKernel(contact_law; state = records)` directly
from a list of measured characteristics, one record per person. When the
characteristics are unobserved or depend on parameters being fitted, rebuild
the rule from the current records at each likelihood evaluation; the
prepared contact structure (`layout`) can be reused. The likelihood throws an
error for a rule with a `state` function whose records have not been
extracted, unless the infection data hold the times the function reads, as in
the next section.

## Infectiousness timed from symptom onset

If infectiousness starts at symptom onset, give the process `from = :onset`,
for example `NetworkProcess(adjacency, kernel; from = :onset)`, and give cases
an onset with [`clinical_presentation`](@ref). Simulation and the likelihood
then both measure the contact interval from each case's onset. An ordinary
distribution is enough.

A rule can also read the onset itself, for transmission that depends on it in
other ways. Here no one is infected before the infector's symptom onset: the
contact interval, measured from infection, is the incubation period plus an
exponential waiting time with mean 1 day. The incubation period is
`Gamma(2.0, 1.0)` (shape 2, scale 1, so a mean of 2 days). Adding a number to a
distribution shifts it by that number.

```@example stateful
onset_state(ind) = (onset = get(ind.state, :onset_time, NaN),)
after_onset(context, source, target) =
    (source.onset - context.infector_infection_time) + Exponential(1.0)
onset_kernel = PairKernel(after_onset; state = onset_state, watches = (:onset_time,))
onset_model = ModelSpec(NetworkProcess(adjacency, onset_kernel);
    attributes = clinical_presentation(incubation_period = Gamma(2.0, 1.0)),
    progression)
onset_run = simulate(onset_model; initial_cases = [1], rng = Xoshiro(237))
```

Simulation sets each case's onset before timing that case's contacts. For the
likelihood, the onset dates go into the infection data alongside the infection
times, with `host_times`. The likelihood then gives the rule the same record
for each person:

```@example stateful
onset_data = network_infections(onset_run, onset_model; host_times = (:onset_time,))
loglikelihood(onset_data, onset_model.process)
```

When onsets are not observed, they are estimated alongside the infection
times. Rebuild the infection data with the current onsets at each evaluation;
the rule stays the same.

## A policy triggered during an outbreak

Policies are often triggered by the outbreak itself, for example once
cumulative cases reach a threshold. Here, the policy starts half a day after
the second case is infected, and cuts the rate of infectious contact from 0.4
to 0.1 per day. Transmission before the policy keeps the old rate.

This needs a small custom intervention (see
[Extending EpiBranch](extending.md) for writing your own):

1. `struct TwoCasePolicy <: AbstractIntervention end` defines a new kind of
   intervention with no parameters.
2. `EpiBranch.resolve_individual!(::TwoCasePolicy, ind, state)` is run for each case once its infection time is known (the `!` marks a
   function that changes its arguments). `state.cumulative_cases == 2 ||
   return nothing` reads "stop here unless this is the second case". For the
   second case, it records the policy start, its infection time plus half a
   day, on every person as `:policy_time`.
3. The rule reads `:policy_time` from the susceptible's record. Before any
   policy date is set it returns a rate of 0.4 per day; afterwards it returns a
   calendar that switches from 0.4 to 0.1 on the policy date.

```@example stateful
struct TwoCasePolicy <: AbstractIntervention end
function EpiBranch.resolve_individual!(::TwoCasePolicy, ind, state)
    state.cumulative_cases == 2 || return nothing
    for person in state.individuals
        person.state[:policy_time] = ind.infection_time + 0.5
    end
    return nothing
end

policy_state(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
function policy_contact(context, source, target)
    isfinite(target.date) || return Exponential(1 / 0.4)
    return (profile = Exponential(1.0), calendar = Steps([target.date], [0.4, 0.1]))
end
policy_kernel = PairKernel(policy_contact; state = policy_state, watches = (:policy_time,))
policy_model = ModelSpec(NetworkProcess(adjacency, policy_kernel);
    progression, interventions = [TwoCasePolicy()])
policy_run = simulate(policy_model; initial_cases = [1], rng = Xoshiro(236))
policy_data = network_infections(policy_run, policy_model)
policy_records = record_kernel(policy_kernel, policy_run)
pairwise_surv_loglik(policy_records, policy_data)
```

Transmission to each susceptible follows the rate in force at each time, so a
policy that starts partway through someone's infectious period changes only the
rest of that period. A run in which no policy is ever triggered follows the
same distribution as the rule without the policy.

In a household model, a policy can depend on cases in other households, and a
vaccine dose limit can be shared between households, as a single stock or as
a daily or weekly allowance. All households then run on one shared calendar,
so doses are counted in the order they are given.

## Checklist for time-varying transmission

Simulation and the likelihood both rely on a few rules. A model that breaks
them still runs, but fits a different model from the one you meant.

- Transmission at time t may depend only on what has happened up to time t.
  Use the date of an event, not a flag for whether it ever happened. For
  example, record a vaccination date and reduce transmission from that date
  on. A flag saying "vaccinated" by the end of the outbreak would also reduce
  transmission before the dose was given.
- The `state` function and the rule must only read values, never change
  them. Changes to people, such as a vaccination or a policy date, belong in
  interventions or in the natural history.
- Keep every date your rule needs. [`record_kernel`](@ref) extracts only what
  the `state` function returns at the end of the simulation, and cannot
  recover a value that was later overwritten. For a rate that changes more
  than once, keep each date and value, and return a distribution or calendar
  that follows them.
- Do not count an effect twice. If an intervention already reduces
  transmission, for example a leaky [`Isolation`](@ref), do not reduce it in
  the rule as well.
- The likelihood here covers transmission only, given the recorded histories.
  If characteristics or intervention assignments are random, their own
  probability models must be added for a joint model.
- Unobserved quantities, such as infection or onset dates, must be estimated
  alongside the other parameters or integrated out. Anything that depends on
  them, such as the trigger date of an outbreak-driven policy, must be
  recalculated at each likelihood evaluation. Reusing the final simulated
  dates fits a different model.
- The rule cannot use future outcomes, such as when the susceptible will
  eventually be infected, or read them from an outside table.
- Infections from outside the network or households, such as community
  introductions, use `external_hazard` (see
  [Community introductions](@ref)). The rules on this page describe
  transmission within the network or household.
