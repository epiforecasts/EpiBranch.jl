# API reference

This page documents every function and type in EpiBranch, grouped by what it
is used for. Most analyses need only a few of them:

- describe transmission with [`BranchingProcess`](@ref) and combine it with the
  natural history, interventions and population characteristics in a
  [`ModelSpec`](@ref);
- add control measures such as [`Isolation`](@ref), [`ContactTracing`](@ref)
  and [`RingVaccination`](@ref);
- run outbreaks with [`simulate`](@ref) and summarise them with
  [`containment_probability`](@ref), [`linelist`](@ref) or
  [`chain_statistics`](@ref);
- calculate exact results such as [`extinction_probability`](@ref) and
  [`chain_size_distribution`](@ref), and fit models to data with
  `loglikelihood`.

Delays and times are in whatever unit the delay distributions use; the
documentation uses days throughout. Names written as `EpiBranch.name` are not
made available by `using EpiBranch` and have to be written in full. A function
whose name ends in `!` changes its first argument in place. The last section,
[For extension authors](@ref), lists what you need only to write a new model,
intervention or observation model.

## Transmission models

A transmission model says who infects whom and when. [`BranchingProcess`](@ref)
is the offspring-distribution model used by epichains and ringbp, optionally
with several types of case. [`HomogeneousProcess`](@ref) is a stochastic
SIR/SEIR epidemic in a closed population. [`NetworkProcess`](@ref),
[`RoutedNetwork`](@ref) and [`HouseholdProcess`](@ref) spread infection over a
fixed set of contacts. [`ModelSpec`](@ref) adds the rest of the scenario.

```@docs
BranchingProcess
Infectiousness
EpiBranch.MultiTypeOffspring
HomogeneousProcess
NetworkProcess
RoutedNetwork
HouseholdProcess
ModelSpec
single_type_offspring
```

## Population characteristics

Functions that give each case its characteristics when it is created: symptom
onset and asymptomatic status, age and sex, susceptibility and infectiousness,
group membership, and willingness to be vaccinated. Pass them to `ModelSpec` as
`attributes`.

```@docs
clinical_presentation
demographics
transmission_traits
groups
group_attribute
vaccine_acceptance
```

## Clinical progression (natural history)

The steps of a case's natural history after infection, each with a probability
and a delay in days: becoming infectious after a latent period, symptom onset,
reporting, hospitalisation, recovery or death. Pass them to `ModelSpec` as
`progression`.

```@docs
AbstractClinicalTransition
Transition
Reporting
Hospitalisation
Death
Recovery
exclusive_probabilities
is_terminal
terminal_event
terminal_target
progression_loglik
```

## Interventions

Control measures applied during an outbreak, passed to `ModelSpec` as
`interventions`. Isolation applies to cases; quarantine applies to traced
contacts who are not (yet) known cases.

```@docs
AbstractIntervention
```

### Isolation

```@docs
Isolation
IsolationEligibility
SymptomaticOnly
AllCases
```

### Contact tracing

Whose contacts are traced, what share of contacts is found, how long tracing
takes and what happens to a traced contact.

```@docs
ContactTracing
TraceEligibility
OnSymptomOnset
OnLabConfirmation
OnIsolation
TraceEveryone
TraceNobody
PreviouslyTraced
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
```

### Vaccination

Ring vaccination of traced contacts, mass vaccination of the population and
vaccination of a case's group, with leaky or all-or-nothing protection.

```@docs
AbstractVaccination
VaccineEffect
RingVaccination
MassVaccination
GroupVaccination
AbstractEffectMode
LeakyMode
AllOrNothingMode
```

### Timing and limited resources

Start or stop an intervention at a given time or case count, or cap how many
people it can reach in each period.

```@docs
Scheduled
is_active
CapacityConstrained
capacity_usage
default_capacity_priority
```

## Transmission routes

A route of transmission, such as community, household or funeral contact, with
its own start, end and timing during a case's course of infection.

```@docs
RouteWindow
window_open
window_close
EpiBranch.INTERVENTION_REMOVAL
```

## Running simulations

[`simulate`](@ref) runs one outbreak or many. Stopping rules end a run early,
for example after a maximum number of cases or days. A run returns a
[`SimulationState`](@ref) holding one [`Individual`](@ref) per case or exposed
contact.

```@docs
simulate
SimOpts
AbstractStoppingRule
Extinction
MaxCases
MaxGenerations
MaxTime
SimulationState
Individual
InfectionEpisode
```

## Recording contacts

Network and household models stop simulating contacts with a person an
intervention has protected for good, such as a vaccinated contact. A contact
recorder keeps those contacts in the simulation. Exposures that did not
infect are listed by [`contacts`](@ref) without one.

```@docs
ContactRecorder
NoContactRecorder
```

## Information about each case

Read what happened to a simulated case: when symptoms began, whether and when
it was isolated, traced, quarantined or vaccinated, and how its infection
ended.

```@docs
onset_time
incubation_period
is_isolated
isolation_time
isolation_release_time
outcome_time
is_traced
is_quarantined
is_vaccinated
immunity_time
severity_efficacy
is_asymptomatic
is_test_positive
is_infected
susceptible_again_time
individual_type
```

## Output

Tables and summaries of simulated outbreaks: line lists, contact lists,
transmission chain statistics, containment probability, weekly incidence and
realised generation intervals.

```@docs
linelist
event_time_metadata
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

## Closed-form results

Results calculated exactly from the offspring distribution, without
simulation: the reproduction number, extinction and epidemic probabilities,
superspreading summaries (as in the superspreading R package) and the
distribution of chain sizes (as in epichains).

### Reproduction number, extinction and superspreading

```@docs
reproduction_number
extinction_probability
epidemic_probability
probability_contain
end_of_outbreak_probability
proportion_transmission
proportion_cases_individual
proportion_cases_offspring
proportion_cluster_size
heterogeneous_contact_R
```

### Chain-size distributions

[`chain_size_distribution`](@ref) gives the probability of each final chain
size (the total number of cases descending from one introduction), calculated
exactly from the offspring distribution. [`Borel`](@ref) is the chain-size
distribution when each case infects a Poisson number of others.

```@docs
chain_size_distribution
Borel
```

### Other chain-size distributions

These are returned by `chain_size_distribution` and are rarely built by hand.
`GammaBorel` is the chain-size distribution for negative binomial offspring.
`PoissonGammaChainSize` is the one for Poisson offspring whose mean varies
between chains following a gamma distribution.

```@docs
EpiBranch.GammaBorel
EpiBranch.PoissonGammaChainSize
```

### A different offspring distribution for the index case

A chain whose index case has its own offspring distribution, for example a
chain started by an imported case, uses [`IndexChainSize`](@ref).

```@docs
IndexChainSize
```

### Household reproduction number

The household reproduction number R\* is the expected number of other
households infected by one infected household. These functions give R\*, the
distribution of the number of households one household infects, and the final
size of the epidemic within a single household.

```@docs
household_offspring
HouseholdOffspring
household_offspring_law
household_final_size
```

## Inference

Fit models to outbreak data. Offspring counts, chain sizes and chain lengths
are fitted by likelihood; individual infection times from households or
contact networks are fitted by a pairwise survival likelihood.

### Data types

```@docs
OffspringCounts
ChainSizes
ChainLengths
```

### Likelihood and fitting

`loglikelihood` works on each kind of outbreak data, so parameters can be
estimated by maximum likelihood or in a Bayesian model:

```julia
loglikelihood(OffspringCounts(data), Poisson(0.5))
loglikelihood(OffspringCounts(data), NegBin.(μ, 0.5))  # one distribution per case
loglikelihood(ChainSizes(data), NegBin(0.8, 0.5))
loglikelihood(ChainLengths(data), Poisson(0.5))
loglikelihood(ChainSizes(data), model)   # interventions/observation read from model
```

The dot in `NegBin.(μ, 0.5)` applies `NegBin` to each element of `μ`, like
vectorised R code. With one distribution per case, each count is evaluated
against its own distribution, which allows case-level covariates such as
`NegBin.(exp.(X * β), k)`. For data that list only cases with at least one
secondary case, pass the zero-truncated distribution
`truncated.(offspring, 1, Inf)`; `Distributions.truncated` works with a single
distribution or with one per case.

For maximum-likelihood estimation, maximise `loglikelihood` with an optimiser
such as Optim.jl, or use Turing's `maximum_likelihood`; the same Turing model
containing `data ~ chain_size_distribution(model)` works for both. See the
[chains tutorial](@ref "Chain statistics, likelihood, and fitting") for
examples.

### Distributions for Bayesian fitting

These return the distribution of the data under a model, which can be written
on the right-hand side of `~` in a Turing.jl model (`data ~ ...`). Where no
exact formula exists, the distribution is estimated by simulation. The
[`chain_size_distribution`](@ref) entry above lists its exact methods.

```@docs
chain_length_distribution
offspring_distribution
```

### Observation models

An observation model describes how outbreaks are detected or reported: for
example each case is reported with some probability, or only chains above a
minimum size are seen. Add one to a model with `observation = ...`.

```@docs
ObservationModel
NoObservation
PerCaseObservation
MinimumSize
observe
ThinnedChainSize
TruncatedChainSize
```

### Reproduction number varying between chains

```@docs
ClusterMixed
ChainSizeMixture
```

### Pairwise survival likelihood

The likelihood of data on who was exposed to whom and when, used to estimate
the contact interval or transmission rate from household or contact-network
studies with individual infection times. The contact interval is the time from
the start of the infector's infectious period to a contact that would infect
if nothing intervened.

```@docs
pairwise_surv_loglik
PairwiseSurvivalData
InfectionLayer
compile_contact_pairs
ContactPairsLayout
PairKernel
EpiBranch.watched_records
Steps
record_kernel
PairContext
LayerHost
```

#### Household data

```@docs
household_infections
HouseholdInfections
HouseholdPairsLayout
compile_household_pairs
ConditionOn
RecruitedIndex
EarliestInfected
```

#### Network data

```@docs
network_infections
NetworkInfections
```

## Distribution helpers

```@docs
NegBin
scale_distribution
incubation_linked_generation_time
```

## For extension authors

The entries below are for writing a new transmission model, intervention,
transition or observation model, and are not needed to run the models above.
[Extending EpiBranch](@ref "Extending EpiBranch") explains how they fit
together.

### Writing a transmission model

A transmission model defines [`initialise_state`](@ref EpiBranch.initialise_state)
to set up its starting population. The helpers listed after it create the
`SimulationState` and add the index cases.

```@docs
EpiBranch.draw_offspring
EpiBranch.generate_offspring
EpiBranch.contacts_of
EpiBranch.collect_exposures
EpiBranch.gather_by_target
EpiBranch.model_generation_time
EpiBranch.transmission_risks
EpiBranch.race_groups
make_contact!
susceptible_fraction
EpiBranch.initialise_state
EpiBranch.new_state
EpiBranch.add_individuals!
EpiBranch.seed!
close_episode!
```

### Stopping rules

```@docs
should_stop
EpiBranch.time_bound
EpiBranch.honoured_without_should_stop
```

### Writing an intervention

```@docs
EpiBranch.initialise_individual!
EpiBranch.resolve_individual!
EpiBranch.apply_post_transmission!
EpiBranch.on_infection_settled!
EpiBranch.trace_contacts!
EpiBranch.traces_contacts
EpiBranch.supplies_contacts
EpiBranch.keep_active
EpiBranch.competing_risk
Risk
EpiBranch.HostSusceptibility
EpiBranch.InfectorInfectiousness
EpiBranch.InfectiousSource
EpiBranch.AbortedInfection
EpiBranch.HostImmunity
EpiBranch.infectious_removal_time
EpiBranch.risk_applies
EpiBranch.risk_depends_on_infector
EpiBranch.standing_block
EpiBranch.intervention_time
EpiBranch.reset!
EpiBranch.abort_infection!
EpiBranch.infection_aborted_time
set_isolated!
clear_isolated!
```

#### Isolation and contact tracing rules

```@docs
EpiBranch.is_eligible_for_isolation
EpiBranch.records_isolation
EpiBranch.is_eligible
EpiBranch.trigger_time
EpiBranch.traces
EpiBranch.draw_trace_delay
EpiBranch.apply_trace!
```

#### Vaccination

```@docs
EpiBranch.vaccine_effect
EpiBranch.realised_efficacy
EpiBranch.realise_prior_dose!
EpiBranch.supports_waning
```

#### Interventions with limited resources

```@docs
EpiBranch.InterventionAction
EpiBranch.intervention_actions
EpiBranch.action_draw!
EpiBranch._action_cache
EpiBranch.apply_actions!
EpiBranch.continuous_actions
EpiBranch.may_revise
EpiBranch.is_settled
EpiBranch.persistent_competing_risks
EpiBranch.capacity_key
EpiBranch.capacity_time_key
```

### Writing a transition

```@docs
EpiBranch.resolve_transitions!
EpiBranch.transition_time
EpiBranch.transition_loglik
EpiBranch.transition_term
```

### Population characteristics and recorded contacts

```@docs
EpiBranch.GroupAttribute
EpiBranch.records_contacts
```

### Writing an observation model

An observation model defines [`observe`](@ref) for the closed-form likelihood
and `apply_observation!` for simulation.

```@docs
EpiBranch.apply_observation!
```

### Pairwise likelihood building blocks

```@docs
EpiBranch.calendar_multiplier
EpiBranch.next_calendar_break
EpiBranch.calendar_shape
EpiBranch.PiecewiseConstantCalendar
EpiBranch.SmoothCalendar
EpiBranch.pair_kernel
pairwise_surv_loglik_by_component
EpiBranch.infection_likelihood_compatible
EpiBranch.susceptibility_components
EpiBranch.susceptibility_host_times
EpiBranch.binding_release
EpiBranch.removal_gap_host_times
EpiBranch.record_removal!
EpiBranch.removal_stretches
EpiBranch.HazardScaling
EpiBranch.contact_structure
EpiBranch.followup_end
EpiBranch.host_times
EpiBranch.PairwiseReduction
EpiBranch.ngroups
EpiBranch.group
EpiBranch.pairwise_reduce
EpiHouseholds.condition_mask
```

### Internal functions

These are not part of the public interface and are documented for people
working on the package itself.

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
