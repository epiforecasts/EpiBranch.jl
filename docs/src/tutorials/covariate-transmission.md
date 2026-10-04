# Covariates and time-varying transmission

In [network](@ref "Network models") and [household](@ref "Household models")
models, transmission between two people is described by the contact interval:
the time from the infector becoming infectious to an infectious contact with
the other person. Often that interval depends on who the two people are, on
when contact happens, or on events during the outbreak such as vaccination or a
change in policy. This page shows how to model each of these, in simulation and
in the pairwise likelihood.

The simplest case needs no extra machinery. A callable
`(infector, susceptible) -> Distribution` gives each ordered pair its own
contact interval from fixed covariates indexed by population ID, as the
[households tutorial](@ref "The pairwise likelihood") shows. The rest of this
page uses [`PairKernel`](@ref) for the cases a callable cannot express: a
contact interval that depends on the infector's infection date, a contact rate
that changes on the calendar, and characteristics or events recorded on each
person during the outbreak.

## Covariates and the infector's infection date

Here the mean contact interval depends on a fixed covariate of the susceptible
and on the date the infector was infected. The covariate vector is indexed by
population ID and stays fixed throughout simulation and likelihood evaluation.

```@example contextual
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

covariates = [0.5, 1.0, 1.5]
kernel = PairKernel(context -> Exponential(exp(
    0.1 * context.infector_infection_time +
    0.2 * covariates[context.susceptible])))

progression = [Transition(:infectious; delay = 0.75),
    Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
network_model = ModelSpec(NetworkProcess(adjacency, kernel); progression)
network_state = simulate(network_model; rng = Xoshiro(233))
network_data = network_infections(network_state, network_model)
loglikelihood(network_data, network_model)
```

The same kernel works for a household model and its likelihood:

```@example contextual
household_model = ModelSpec(HouseholdProcess([3], kernel); progression)
household_state = simulate(household_model; rng = Xoshiro(233))
household_data = household_infections(household_state, household_model)
loglikelihood(household_data, household_model)
```

`context.infector_infection_time` is the infection date, even when a latent period
starts the infectious period later. The returned distribution still measures time
from the start of the infectious period. `context.infector` and
`context.susceptible` are population IDs. Without a `state` argument, the callback
takes only this context. With one, it also receives a record for each of the two
people, as later sections show.

## Inference

A compiled layout stores the contact structure. Each evaluation reads the
infector's infection time from the supplied data, including when latent infection
times change during inference:

```@example contextual
layout = compile_contact_pairs(network_data)
pairwise_surv_loglik(kernel, network_data, layout)
```

Forward- and reverse-mode automatic differentiation can include both kernel
parameters and infection times, subject to the chosen distribution's own
differentiation support.

The lower-level `PairwiseSurvivalData` representation contains counting-process
rows but lacks infector IDs and their infection dates. Its callable kernels still
receive a row index. Use an `InfectionLayer` with a `PairKernel`, or supply the
needed information through a row-indexed callback yourself.

## A policy starting on a calendar day

The `calendar` argument multiplies the contact rate by a schedule on the
calendar. With a [`Steps`](@ref) schedule the multiplier is a step function of
the date. A policy can then change transmission during someone's infectious
period, including when their latent period was sampled during simulation.

Suppose the contact rate is 0.4 per day before day 3 and 0.1 afterwards. A flat,
unit-hazard profile multiplied by a single step at day 3 is exactly this policy:

```@example calendar
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

policy_day = 3.0
before_rate = 0.4
after_rate = 0.1
kernel = PairKernel(context -> Exponential(1.0);
    calendar = Steps([policy_day], [before_rate, after_rate]))
```

The cumulative hazard splits into a segment before the policy day and a segment
after it, and both simulation and the likelihood compute it exactly for any
profile.

Consider a person who becomes infectious on day 2. By day 4, the cumulative
hazard is `0.4 × 1 + 0.1 × 1 = 0.5`, and the pair's contact interval has exactly
that cumulative hazard after two days:

```@example calendar
interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 2.0)
EpiBranch.cumhazard(interval, 2.0)
```

The last two arguments are the infector's infection date and the start of its
infectious period.
The process supplies these automatically. Here is a network simulation with a
sampled latent period and its infection likelihood:

```@example calendar
progression = [Transition(:infectious; delay = Uniform(0.4, 0.8)),
    Transition(:recovered; from = :infectious, delay = 4.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
model = ModelSpec(NetworkProcess(adjacency, kernel); progression)
state = simulate(model; rng = Xoshiro(234))
data = network_infections(state, model)
loglikelihood(data, model)
```

A per-edge vector of distributions does not take a `calendar` schedule. To apply
one, look the pair's distribution up inside a `PairKernel` callback and pass the
schedule there:

```julia
edges = [[Exponential(1.0), Exponential(2.0)] for _ in adjacency]
PairKernel(context -> edges[context.infector][findfirst(==(context.susceptible),
        adjacency[context.infector])];
    calendar = Steps([3.0], [1.0, 0.25]))   # rate falls to a quarter on day 3
```

Replace `NetworkProcess(adjacency, kernel)` with `HouseholdProcess([3], kernel)`
and extract `household_infections` to use the same policy in a household model.
Both likelihoods condition on the observed start of each infectious period. A
compiled layout reads those times again at every evaluation, allowing them to
change during inference.

A pair whose schedule differs from the shared one instead returns it from the
callback, as `(profile = ..., calendar = ...)`, overriding the kernel's own
`calendar` for that pair. For example, this calendar hazard depends on a fixed
recipient covariate and the source's infection date, with the policy day itself
read from the recipient's covariate:

```@example calendar
covariates = [1.0, 2.0, 4.0]
covariate_kernel = PairKernel(context ->
    (profile = Weibull(2.0, exp(1.0 + 0.05 * context.infector_infection_time)),
     calendar = Steps([covariates[context.susceptible]], [1.0, 0.25])))
```

Automatic differentiation through a schedule's rates and breakpoints, and
through the start of each infectious period, uses the profile's own differentiation support.
The tests check forward and reverse derivatives for a step-scaled Weibull
profile against its analytical likelihood.

## A seasonal contact rate

A schedule need not be a step function. Any type with a
[`calendar_multiplier`](@ref EpiBranch.calendar_multiplier) method can be a
`calendar`, and one that declares itself smooth through
[`calendar_shape`](@ref EpiBranch.calendar_shape) is integrated by quadrature
instead of segment by segment. Here contact rates rise and fall over a year:

```@example calendar
struct Seasonal{T <: Real}
    amplitude::T
    peak_day::Float64
end
function EpiBranch.calendar_multiplier(s::Seasonal, t)
    return 1 + s.amplitude * cos(2π * (t - s.peak_day) / 365)
end
EpiBranch.calendar_shape(::Seasonal) = EpiBranch.SmoothCalendar()

seasonal_kernel = PairKernel(context -> Exponential(4.0);
    calendar = Seasonal(0.6, 30.0))
```

The cumulative hazard over two days from the start of an infectious
period on day 100 is the integral
of the multiplier times the profile's constant rate of 0.25 per day:

```@example calendar
seasonal_interval = EpiBranch.pair_kernel(seasonal_kernel, 1, 2, 0.0, 100.0)
exact = 0.25 * (2 + 0.6 * 365 / 2π *
    (sin(2π * (102 - 30) / 365) - sin(2π * (100 - 30) / 365)))
(EpiBranch.cumhazard(seasonal_interval, 2.0), exact)
```

Simulation draws contact intervals by inverting that same integrated hazard,
which keeps the network simulation consistent with its likelihood:

```@example calendar
seasonal_model = ModelSpec(NetworkProcess(adjacency, seasonal_kernel); progression)
seasonal_state = simulate(seasonal_model; rng = Xoshiro(235))
loglikelihood(network_infections(seasonal_state, seasonal_model), seasonal_model)
```

With a unit-rate profile, `Exponential(1.0)`, the multiplier is the pair's
hazard on the calendar. Any calendar-time hazard can therefore be written as a
schedule. The schedule's fields are typed so that the likelihood can be
differentiated through them, as with `Steps`.

## Characteristics recorded during simulation

The `state` argument selects what the kernel reads from each person. Its
callback then also receives a record for each of the two people. Here, each
person receives a sampled contact-scale attribute:

```@example stateful
using EpiBranch, EpiNetwork, Distributions, Random

attributes = (rng, ind) -> (ind.state[:contact_scale] = rand(rng, Uniform(0.5, 1.5)))
project(ind) = (scale = ind.state[:contact_scale]::Float64,)
contact_law(context, source, target) = Exponential(source.scale + target.scale)
kernel = PairKernel(contact_law; state = project, watches = (:contact_scale,))
adjacency = [[2, 3], [1, 3], [1, 2]]
progression = [Transition(:recovered; delay = 5.0, terminal = true)]
model = ModelSpec(NetworkProcess(adjacency, kernel); attributes, progression)
state = simulate(model; initial_cases = [1], rng = Xoshiro(235))
```

The rate at which a case infects its contacts can change during its infectious
period: a post-exposure dose takes effect, or symptoms begin and the case is
isolated. When the kernel reads one of those dates from the record, simulation
has already drawn the times of that case's contacts at the rate in force then,
and has to draw them again at the new rate. `watches` names the
records to watch for: the `individual.state` keys the `state` function reads. Above,
that is the one key the attributes builder sets.

Name every key the `state` function reads, even one that nothing in this model
writes:

```@example stateful
# Two keys read, both declared: the dose date a `RingVaccination` writes and
# the onset `clinical_presentation` sets.
dosed_kernel = PairKernel(
    (context, source, target) -> Exponential(isfinite(target.dosed) ? 4.0 : 1.5);
    state = ind -> (
        dosed = get(ind.state, :vaccination_time, Inf)::Float64,
        onset = get(ind.state, :onset_time, NaN),
    ),
    watches = (:vaccination_time, :onset_time)
)
EpiBranch.watched_records(dosed_kernel)
```

A `state` function that reads no record at all declares `()`. This one scales each
person's contact rate by a fixed covariate, indexed by their own id:

```@example stateful
scale_by_id = [1.0, 2.0, 0.5]
by_id_kernel = PairKernel(
    (context, source, target) -> Exponential(source.scale);
    state = ind -> (scale = scale_by_id[ind.id],), watches = ()
)
EpiBranch.watched_records(by_id_kernel)
```

Leave a key out and the contacts keep the rate they were drawn at after the
record changes, with nothing to report it. Name a key that never changes and
simulation pays one comparison per case. When in doubt, name it.

After simulation, extract the records and use the same callback in the
likelihood. `record_kernel` copies each person's record into a vector indexed
by population ID:

```@example stateful
data = network_infections(state, model)
recorded = record_kernel(kernel, state)
layout = compile_contact_pairs(data)
pairwise_surv_loglik(recorded, data, layout)
```

For observed data, build `PairKernel(contact_law; state = records)` directly
from measured covariates. When attributes are latent or contain fitted
parameters, build that kernel from the current records on each likelihood
evaluation. The compiled layout can be reused. Records retain their numeric
types, including AD values. A kernel whose records have not been extracted
raises an error in the likelihood, unless the infection layer holds the times
it reads, as in the next section.

## Infectiousness timed from symptom onset

If infectiousness starts at symptom onset, set `from = :onset` on the process
and add an onset transition to the progression. Simulation and the likelihood
then both measure the contact interval from each case's onset, and an ordinary
distribution or callable is enough.

A kernel can also read the onset itself, for a contact rate that depends on it
in other ways. Here the record holds the onset, and the callback shifts the
contact interval by the time from infection to onset:

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

Simulation sets each case's onset before it draws that case's contacts, and the
record reads it directly. In the likelihood, the onsets belong in the infection
layer. `host_times` records them alongside the infection times. The likelihood
then builds the same record for each person as a [`LayerHost`](@ref):

```@example stateful
onset_data = network_infections(onset_run, onset_model; host_times = (:onset_time,))
loglikelihood(onset_data, onset_model.process)
```

In inference the onsets are augmented with the infection times. Build the layer
with the current onsets as `host_times` on each evaluation, and the kernel stays
unchanged.

## A policy triggered during an outbreak

Suppose the second case triggers a policy half a day later. The intervention
records that date on each person. The kernel's per-pair calendar reads it back
and switches the rate from that date, leaving the hazard of an earlier exposure
unchanged:

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

A contact drawn before a policy took effect still follows the hazard in force
once it has. Simulation keeps pending contacts consistent with the records as
they change. A run whose records never change follows the same distribution as
an ordinary kernel.

A policy can depend on cases in other households. The existing restriction on
periodic shared capacity budgets still applies.

## Requirements for simulation and inference

The kernel must define a predictable hazard. An event recorded at time `t` may
change the hazard at or after `t`; it must preserve the earlier hazard. Store
dates or event histories instead of using a final vaccinated or quarantined flag
to change the whole infectious window. The `state` function and the callback must be
free of side effects. State updates occur in the existing case-resolution and
intervention hooks; scheduled future effects must be encoded in the returned
hazard law.

`record_kernel` extracts the history your `state` function retains. It cannot
recover past values that an intervention overwrote. For several changes, retain
all relevant dates and values and construct a distribution whose hazard follows
them. An intervention that also contributes a built-in risk must not have that
same effect counted again in the kernel.

The likelihood evaluates the transmission contribution along the supplied
histories. A joint model of sampled attributes or stochastic intervention
assignment also needs their probability models. Unobserved histories need to be
augmented or integrated out. When changing infection times changes an endogenous
policy's trigger date, reconstruct that history at each likelihood evaluation;
reusing the final simulated dates would fit a different model.

The susceptible's eventual infection time is unknown when simulation chooses its
contact distribution. Neither the `state` function nor the callback may read
future outcomes from an external table. Community introductions from outside the
network or households are modelled by `external_hazard`. The kernels on this page describe transmission within the
network or household.
