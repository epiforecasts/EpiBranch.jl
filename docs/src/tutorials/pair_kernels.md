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

## Scope

The callback must return the same distribution for a pair throughout the
infector's infectious window. Infection date can select that distribution, but
this is not a hazard that is re-evaluated as calendar time advances. Mutable
intervention histories and attributes sampled during the run are outside this
interface. Simulate them only through an interface that represents their timing
and supplies the corresponding information to inference.

The susceptible's eventual infection time is deliberately absent: it is unknown
when the simulator chooses a contact distribution. A kernel must not read future
outcomes from an external table. Independent community introductions remain the
role of `external_hazard`; changing that hazard does not change within-network or
within-household transmission.
