# Contextual pair kernels

[`ContextualKernel`](@ref) gives a pair-kernel callback information that both a
simulator and an infection-layer likelihood can supply: the two population IDs
and the infector's infection time. The callback returns the distribution of time
from infectious opening to an infectious contact.

Use an ordinary `(infector, susceptible)` callable when IDs are sufficient. An
explicit wrapper selects the context form without changing existing callbacks.
Shared distributions and per-edge distribution vectors keep their existing use.

## Fixed covariates and infection time

Here the mean contact interval depends on a fixed recipient covariate and the
infector's infection date. The covariate vector is indexed by population ID and
stays fixed throughout simulation and likelihood evaluation.

```@example contextual
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

covariates = [0.5, 1.0, 1.5]
kernel = ContextualKernel(context -> Exponential(exp(
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
`PairContext` contains no live individual or state dictionary.

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
receive a row index. Use an `InfectionLayer` with `ContextualKernel`, or supply the
needed information through a row-indexed callback yourself.

## A policy starting on a calendar day

`CalendarKernel` interprets a distribution's hazard on the calendar-time axis.
It conditions that distribution on each infector's infectious opening, then
converts calendar dates to elapsed contact intervals. This allows a policy to
change transmission partway through someone's infectious period, including when
their latent period was sampled during simulation.

Suppose the contact rate is 0.4 per day before day 3 and 0.1 afterwards. Standard
distributions can express this in two parts: a contact before day 3, or survival
to day 3 followed by an exponential waiting time at the lower rate.

```@example calendar
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

policy_day = 3.0
before_rate = 0.4
after_rate = 0.1
survive_to_policy = exp(-before_rate * policy_day)
calendar_law = MixtureModel(
    [truncated(Exponential(1 / before_rate); upper = policy_day),
     policy_day + Exponential(1 / after_rate)],
    [1 - survive_to_policy, survive_to_policy])
kernel = CalendarKernel(calendar_law)
```

The mixture weights are the probabilities of making the first contact before or
after the policy date. They ensure the rate is 0.4 before day 3 and 0.1 after it.
The policy date and rates are fixed inputs, shared by simulation and inference.

Consider a person who becomes infectious on day 2. By day 4, the cumulative
hazard is `0.4 × 1 + 0.1 × 1 = 0.5`. The contact interval returned by the adapter
has exactly that cumulative hazard after two elapsed days:

```@example calendar
interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 2.0)
-logccdf(interval, 2.0)
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

`CalendarKernel` also wraps ID callbacks, `ContextualKernel` and network per-edge
distribution vectors. For example, this calendar hazard depends on a fixed
recipient covariate and the source's infection date:

```@example calendar
covariates = [0.5, 1.0, 1.5]
covariate_kernel = CalendarKernel(ContextualKernel(context ->
    Weibull(2.0, exp(1.0 + 0.1 * covariates[context.susceptible] +
                     0.05 * context.infector_infection_time))))
```

Automatic differentiation through calendar-law parameters and infectious openings
uses the chosen distribution's differentiation support. The tests check forward
and reverse derivatives for a Weibull calendar law against its analytical
likelihood. Derivatives at a sharp policy boundary need particular care because
the hazard itself jumps there. At day 3 the example mixture includes both
component endpoint densities; likelihoods evaluated exactly at a policy date
need a distribution with the endpoint convention required by the model.

Conditioning requires positive survival at the infectious opening. Choose a
calendar law whose tail probabilities can be represented numerically over the
simulation period; truncating after its survival has underflowed to zero cannot
produce a valid conditional distribution.

## Scope

A callback returns a distribution that remains fixed throughout the infector's
infectious window. That distribution can have a changing hazard. `CalendarKernel`
aligns that hazard with calendar dates; an ordinary kernel measures elapsed time
from infectious opening.

Known policy dates and fixed covariate tables can be shared by simulation and
inference. Policies triggered by evolving case counts, attributes sampled during
the run and vaccination or tracing histories generated by interventions still
need an explicit history representation. This adapter does not read mutable
intervention state or reconstruct those histories from final flags.

The susceptible's eventual infection time is deliberately absent: it is unknown
when the simulator chooses a contact distribution. A kernel must not read future
outcomes from an external table. Independent community introductions remain the
role of `external_hazard`; changing that hazard does not change within-network or
within-household transmission.
