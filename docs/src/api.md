# API reference

## Transmission models

```@docs
BranchingProcess
```

## Types

```@docs
Individual
SimulationState
SimOpts
DelayOpts
OutcomeOpts
DemographicOpts
EpiBranch.GenerationTimeSpec
EpiBranch.OffspringSpec
```

## Simulation

```@docs
simulate
simulate_batch
simulate_conditioned
EpiBranch.step!
```

## Interventions

```@docs
AbstractIntervention
Isolation
ContactTracing
EpiBranch.initialise_individual!
EpiBranch.resolve_individual!
EpiBranch.apply_post_transmission!
```

## State accessors

```@docs
onset_time
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
containment_probability
is_extinct
effective_R
weekly_incidence
```

## Analytical

```@docs
extinction_probability
epidemic_probability
proportion_transmission
chain_size_distribution
chain_size_ll
chain_length_ll
Borel
GammaBorel
```

## Convenience constructors

```@docs
NegBin
scale_distribution
ringbp_generation_time
```

## Internals

These functions are not part of the public API but are documented for
developers extending the package.

```@docs
EpiBranch.get_generation_time
EpiBranch._draw_offspring
EpiBranch._create_contacts!
EpiBranch._resolve_infection
EpiBranch._susceptible_fraction
EpiBranch._residual_fraction
EpiBranch._create_individual
EpiBranch.logsumexp
```
