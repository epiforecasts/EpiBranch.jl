# Contextual and calendar-time pair kernels

[`PairKernel`](@ref) gives a pair-kernel callback information that both a
simulator and an infection-layer likelihood can supply: the two population IDs,
the infector's infection time and, optionally, each host's own record. The
callback returns the contact-interval profile, measured from the infector's
infectious opening, and optionally a step schedule that multiplies the rate on
the calendar.

Use an ordinary `(infector, susceptible)` callable when IDs are sufficient. A
`PairKernel` selects the richer forms without changing existing callbacks.
Shared distributions and per-edge distribution vectors keep their existing use.

## Fixed covariates and infection time

Here the mean contact interval depends on a fixed recipient covariate and the
infector's infection date. The covariate vector is indexed by population ID and
stays fixed throughout simulation and likelihood evaluation.

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
opens the infectious window later. The returned distribution still measures time
from that window's opening. `context.infector` and `context.susceptible` are IDs;
`PairContext` contains no live individual or state dictionary. With no `state`
given to `PairKernel`, the callback takes only this context.

## Inference

A compiled layout stores the contact structure. Each evaluation reads the
infector's infection time from the supplied data, including when latent infection
times change during inference:

```@example contextual
layout = compile_contact_pairs(network_data)
pairwise_surv_loglik(kernel, network_data, layout)
```

The context preserves the infection time's number type. Forward- and reverse-mode
automatic differentiation can include both kernel parameters and infection times,
subject to the chosen distribution's existing differentiation support. Context
construction uses a small immutable value and leaves the compiled layout unchanged.

The lower-level `PairwiseSurvivalData` representation contains counting-process
rows but lacks infector IDs and their infection dates. Its callable kernels still
receive a row index. Use an `InfectionLayer` with a `PairKernel`, or supply the
needed information through a row-indexed callback yourself.

## A policy starting on a calendar day

`calendar`, a [`Steps`](@ref) schedule, multiplies a `PairKernel`'s returned
profile by a step function of the calendar date — the infector's infectious
opening plus time elapsed — rather than time since opening. This lets a policy
change transmission partway through someone's infectious period, including when
their latent period was sampled during simulation.

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
after it, each a scaled difference of the profile's own cumulative hazard, so it
stays exact whatever the profile.

Consider a person who becomes infectious on day 2. By day 4, the cumulative
hazard is `0.4 × 1 + 0.1 × 1 = 0.5`. The contact interval returned by the adapter
has exactly that cumulative hazard after two elapsed days:

```@example calendar
interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 2.0)
EpiBranch.cumhazard(interval, 2.0)
```

The last two arguments are the infector's infection date and infectious opening.
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

Replace `NetworkProcess(adjacency, kernel)` with `HouseholdProcess([3], kernel)`
and extract `household_infections` to use the same policy in a household model.
Both likelihoods condition on the observed infectious openings. A compiled layout
reads those openings again at every evaluation, allowing them to change during
inference.

A pair whose schedule differs from the shared one returns it instead from the
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
through infectious openings, uses the profile's own differentiation support.
The tests check forward and reverse derivatives for a step-scaled Weibull
profile against its analytical likelihood.

The multiplier only rescales the hazard over calendar time, so a calendar law
whose shape changes continuously over time cannot be written this way; a smooth
change needs a step approximation, breaking it into enough `Steps` breakpoints.

## Attributes sampled during simulation

`state` lets a `PairKernel` choose which parts of an individual it reads. Given
`state`, the callback also takes the pair's two host records, built from the
usual `PairContext`. Here, each person receives a sampled contact-scale
attribute:

```@example stateful
using EpiBranch, EpiNetwork, Distributions, Random

attributes = (rng, ind) -> (ind.state[:contact_scale] = rand(rng, Uniform(0.5, 1.5)))
project(ind) = (scale = ind.state[:contact_scale]::Float64,)
contact_law(context, source, target) = Exponential(source.scale + target.scale)
kernel = PairKernel(contact_law; state = project)
adjacency = [[2, 3], [1, 3], [1, 2]]
progression = [Transition(:recovered; delay = 5.0, terminal = true)]
model = ModelSpec(NetworkProcess(adjacency, kernel); attributes, progression)
state = simulate(model; initial_cases = [1], rng = Xoshiro(235))
```

After simulation, extract the selected records and use the same callback in the
likelihood. `record_kernel` copies the projection results into a vector indexed
by population ID. The likelihood reads those typed records without constructing
individuals or reading their dictionaries:

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
types, including AD values. An unrecorded projection raises an error on the
likelihood path unless the infection layer holds the times it reads, as in the
next section.

## Infectiousness timed from symptom onset

An infector often becomes infectious around its symptom onset, so its contact
interval depends on its own incubation period. The projection reads the onset
from the host, and the callback shifts the contact law by the time from
infection to onset:

```@example stateful
onset_state(ind) = (onset = get(ind.state, :onset_time, NaN),)
after_onset(context, source, target) =
    (source.onset - context.infector_infection_time) + Exponential(1.0)
onset_kernel = PairKernel(after_onset; state = onset_state)
onset_model = ModelSpec(NetworkProcess(adjacency, onset_kernel);
    attributes = clinical_presentation(incubation_period = Gamma(2.0, 1.0)),
    progression)
onset_run = simulate(onset_model; initial_cases = [1], rng = Xoshiro(237))
```

Simulation sets each case's onset before it draws that case's contacts, so the
projection reads the onset directly. In the likelihood, the onsets belong in the
infection layer: `host_times` records them alongside the infection times, and
the likelihood applies the same projection to each host as a
[`LayerHost`](@ref):

```@example stateful
onset_data = network_infections(onset_run, onset_model; host_times = (:onset_time,))
loglikelihood(onset_data, onset_model.process)
```

In inference the onsets are augmented with the infection times. Build the layer
with the current onsets as `host_times` on each evaluation, and the kernel stays
unchanged.

## A policy triggered during an outbreak

Suppose the second case triggers a policy half a day later. The intervention
records that date on each host, and the kernel's per-pair calendar reads it back
to switch the rate at that date, so an earlier exposure keeps its original
hazard:

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
policy_kernel = PairKernel(policy_contact; state = policy_state)
policy_model = ModelSpec(NetworkProcess(adjacency, policy_kernel);
    progression, interventions = [TwoCasePolicy()])
policy_run = simulate(policy_model; initial_cases = [1], rng = Xoshiro(236))
policy_data = network_infections(policy_run, policy_model)
policy_records = record_kernel(policy_kernel, policy_run)
pairwise_surv_loglik(policy_records, policy_data)
```

A contact scheduled before a policy took effect still follows the hazard in
force once it has: simulation keeps pending contacts consistent with the records
as they change, and a run whose records never change follows the same
distribution as an ordinary kernel.

A policy can depend on cases in other households. The existing restriction on
periodic shared capacity budgets still applies.

## History and inference contracts

The kernel must define a predictable hazard. An event recorded at time `t` may
change the hazard at or after `t`; it must preserve the earlier hazard. Store
dates or event histories instead of using a final vaccinated or quarantined flag
to change the whole infectious window. Projections and kernel callbacks must be
free of side effects. State updates occur in the existing case-resolution and
intervention hooks; scheduled future effects must be encoded in the returned
hazard law.

`record_kernel` extracts the history retained by your projection. It cannot
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
contact distribution. Neither projections nor callbacks may read future outcomes
from an external table. Independent community introductions remain the role of
`external_hazard`; these kernel adapters change within-network or within-household
transmission.
