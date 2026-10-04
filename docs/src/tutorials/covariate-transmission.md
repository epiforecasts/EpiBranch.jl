# Covariates and time-varying transmission

In [network](@ref "Network models") and [household](@ref "Household models")
models, transmission between two people is described by the contact interval:
the time from the infector becoming infectious to an infectious contact with
the other person. Often that interval depends on who the two people are, or on
when contact happens. This page shows how to model both, in simulation and in
the pairwise likelihood.

## Covariates of either person

A callable `(infector, susceptible) -> Distribution` gives each ordered pair its
own contact interval. Here the contact rate rises with a fixed covariate of the
susceptible, held in a vector indexed by population ID:

```@example covariates
using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random

covariates = [0.5, 1.0, 1.5]
kernel(infector, susceptible) = Exponential(exp(-0.2 * covariates[susceptible]))

progression = [Transition(:infectious; delay = 0.75),
    Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
network_model = ModelSpec(NetworkProcess(adjacency, kernel); progression)
network_state = simulate(network_model; rng = Xoshiro(233))
network_data = network_infections(network_state, network_model)
loglikelihood(network_data, network_model)
```

The same callable works for a household model and its likelihood:

```@example covariates
household_model = ModelSpec(HouseholdProcess([3], kernel); progression)
household_state = simulate(household_model; rng = Xoshiro(233))
household_data = household_infections(household_state, household_model)
loglikelihood(household_data, household_model)
```

The simulator and the likelihood call it with the same IDs, and one function
serves both. Per-person differences known in advance, such as age or a
sampled contact rate, can be generated before the run and captured in the same
way.

## A policy starting on a calendar day

By default the contact interval is measured from when the infector becomes
infectious. With `calendar_time = true` the distribution instead describes the
contact hazard on the calendar-time axis. This allows a policy to change
transmission partway through someone's infectious period, including when their
latent period was sampled during simulation.

Suppose the contact rate is 0.4 per day before day 3 and 0.1 afterwards.
Standard distributions can express this in two parts: a contact before day 3,
or survival to day 3 followed by an exponential waiting time at the lower rate.

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
```

The mixture weights are the probabilities of making the first contact before or
after the policy date. With these weights the rate is 0.4 before day 3 and 0.1
after it.
Between days 2 and 4 the cumulative hazard is `0.4 × 1 + 0.1 × 1 = 0.5`:

```@example calendar
logccdf(calendar_law, 2.0) - logccdf(calendar_law, 4.0)
```

Here is a network simulation with a sampled latent period, and its likelihood:

```@example calendar
progression = [Transition(:infectious; delay = Uniform(0.4, 0.8)),
    Transition(:recovered; from = :infectious, delay = 4.0, terminal = true)]
adjacency = [[2, 3], [1, 3], [1, 2]]
model = ModelSpec(NetworkProcess(adjacency, calendar_law; calendar_time = true);
    progression)
state = simulate(model; rng = Xoshiro(234))
data = network_infections(state, model)
loglikelihood(data, model)
```

`HouseholdProcess([3], calendar_law; calendar_time = true)` with
`household_infections` uses the same policy in a household model. Both
likelihoods condition on the observed start of each infectious period. They read
those times again at every evaluation, allowing them to change during inference.

Covariates and calendar time combine: a callable can return a calendar-time
distribution for each pair.

```@example calendar
covariates = [0.5, 1.0, 1.5]
covariate_law(infector, susceptible) =
    Weibull(2.0, exp(1.0 + 0.1 * covariates[susceptible]))
NetworkProcess(adjacency, covariate_law; calendar_time = true)
```

The calendar-time distribution must have positive survival at the start of each
infectious period. Choose one whose tail probabilities can be represented
numerically over the simulation period. The hazard jumps at a policy date. A
likelihood evaluated exactly there depends on the distribution's convention at
that endpoint.

## Other modelling choices

To time infectiousness from symptom onset, set `from = :onset` on the process
and add an onset transition to the progression. Simulation and the likelihood
then both measure the contact interval from each case's onset.

Policies triggered by the outbreak itself, such as distancing that starts once a
number of cases have been reported, are [interventions](interventions.md).
Simulation supports them on network and household models. The pairwise
likelihood scores the infectious periods that interventions shorten, such as by
isolation. It does not yet score interventions that change the contact rate
during an infectious period.

Community introductions from outside the network or households are modelled by
`external_hazard`.
