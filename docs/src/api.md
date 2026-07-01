# API reference

## Transmission models

```@docs
BranchingProcess
Infectiousness
NetworkProcess
HouseholdProcess
single_type_offspring
EpiBranch.draw_offspring
```

## Household likelihood

```@docs
household_infections
HouseholdInfections
pairwise_surv_loglik
PairwiseSurvivalData
```

## Types

```@docs
Individual
SimulationState
SimOpts
AbstractStoppingRule
Extinction
MaxCases
MaxGenerations
MaxTime
should_stop
```

## Simulation

```@docs
simulate
EpiBranch.generate_offspring
EpiBranch.contacts_of
EpiBranch.collect_exposures
EpiBranch.gather_by_target
EpiBranch.model_generation_time
EpiBranch.transmission_risks
make_contact!
susceptible_fraction
```

### Building a model's starting population

A transmission model defines [`initialise_state`](@ref EpiBranch.initialise_state)
to set up its starting population. These helpers carry the shared work, so a
model never touches the `SimulationState` constructor or the engine's
bookkeeping directly.

```@docs
EpiBranch.initialise_state
EpiBranch.new_state
EpiBranch.add_individuals!
EpiBranch.seed!
```

## Interventions

```@docs
AbstractIntervention
Isolation
IsolationEligibility
SymptomaticOnly
AllCases
EpiBranch.is_eligible_for_isolation
ContactTracing
TraceEligibility
OnSymptomOnset
OnLabConfirmation
OnIsolation
TraceEveryone
TraceNobody
AlwaysEligible
SymptomaticParent
NoTracing
AnyOf
AllOf
NoneOf
TraceRate
ConstantRate
TraceDelay
ConstantDelay
TraceAction
Quarantine
FlagOnly
EpiBranch.is_eligible
EpiBranch.trigger_time
EpiBranch.traces
EpiBranch.draw_trace_delay
EpiBranch.apply_trace!
AbstractVaccination
RingVaccination
MassVaccination
AbstractEffectMode
LeakyMode
AllOrNothingMode
Scheduled
Risk
EpiBranch.HostSusceptibility
EpiBranch.InfectorInfectiousness
EpiBranch.InfectiousSource
EpiBranch.initialise_individual!
EpiBranch.resolve_individual!
EpiBranch.apply_post_transmission!
EpiBranch.keep_active
EpiBranch.competing_risk
EpiBranch.intervention_time
EpiBranch.reset!
is_active
```

## Natural history (progression)

```@docs
AbstractClinicalTransition
Transition
Reporting
Hospitalisation
Death
Recovery
is_terminal
terminal_event
EpiBranch.resolve_transitions!
```

## State accessors

```@docs
onset_time
incubation_period
is_isolated
isolation_time
is_traced
is_quarantined
is_vaccinated
is_asymptomatic
is_test_positive
is_infected
individual_type
set_isolated!
```

## Output

```@docs
linelist
contacts
chain_statistics
compute_trace_level!
realised_generation_interval
realised_generation_intervals
containment_probability
is_extinct
generation_R
weekly_incidence
```

## Analytical

### Helpers

```@docs
extinction_probability
epidemic_probability
probability_contain
proportion_transmission
proportion_cluster_size
heterogeneous_contact_R
```

### Chain-size distributions

`chain_size_distribution(spec)` is the dispatch entry point: given an
offspring specification it returns the closed-form cluster-size PMF as a
`Distribution`. `Borel` (chain sizes from Poisson offspring) is the one
size law you also construct directly — it is a standard named
distribution. The observation-side counterpart (`ThinnedChainSize`)
lives with the observation models.

```@docs
chain_size_distribution
Borel
```

#### Dispatch outputs (not exported)

Returned by `chain_size_distribution` rather than constructed by name, so
they are not exported. `GammaBorel` is the size law for `NegativeBinomial`
offspring (individual-level Gamma-Poisson mixing); `PoissonGammaChainSize`
is the size law for Poisson offspring with a chain-level Gamma rate.

```@docs
EpiBranch.GammaBorel
EpiBranch.PoissonGammaChainSize
```

## Inference

### Data types

```@docs
OffspringCounts
ChainSizes
ChainLengths
```

### Distribution entry points

These return a `Distribution` (the analytical form where one exists,
otherwise a wrapper around `loglikelihood`) you can put on the
right-hand side of Turing's `~`. The [`chain_size_distribution`](@ref)
entry above lists its process-side (analytical) methods.

```@docs
chain_length_distribution
offspring_distribution
```

### Observation models

An observation model is part of the process. Pass `observation = …` to a
process constructor. It is added the same way an intervention is, by
implementing two methods dispatched on the observation type:
[`observe`](@ref) for the analytical likelihood and `apply_observation!`
for simulation.

```@docs
ObservationModel
NoObservation
PerCaseObservation
observe
EpiBranch.apply_observation!
ThinnedChainSize
```

### Cluster-level heterogeneity

```@docs
ClusterMixed
ChainSizeMixture
```

### Likelihood and fitting

EpiBranch extends `Distributions.loglikelihood` with methods for each
data wrapper:

```julia
loglikelihood(OffspringCounts(data), Poisson(0.5))
loglikelihood(ChainSizes(data), NegBin(0.8, 0.5))
loglikelihood(ChainLengths(data), Poisson(0.5))
loglikelihood(ChainSizes(data), model)   # interventions/observation read from model
```

For maximum-likelihood estimation, pair the `loglikelihood` interface
with Optim.jl, or use Turing's `maximum_likelihood` — the same model that
feeds `data ~ chain_size_distribution(model)` works for both.

See the [chains tutorial](@ref "Chain statistics, likelihood, and fitting")
for examples.

## Init functions

```@docs
clinical_presentation
demographics
transmission_traits
compose
```

## Convenience constructors

```@docs
NegBin
scale_distribution
incubation_linked_generation_time
```

## Internals

These functions are not part of the public API but are documented for
developers extending the package.

```@docs
EpiBranch.get_generation_time
EpiBranch._advance_generation!
EpiBranch._prepare_parents!
EpiBranch._intervene!
EpiBranch._resolve!
EpiBranch._decide_infected
EpiBranch.population_size
EpiBranch._create_individual
EpiBranch.logsumexp
EpiBranch.required_fields
EpiBranch._validate_required_fields
EpiBranch._column_order
EpiBranch._chain_length_ll_negbin
EpiBranch._borel_logpdf
EpiBranch._gammaborel_logpdf
EpiBranch._empirical_ll
```
