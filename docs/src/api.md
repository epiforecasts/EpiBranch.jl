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
```

## Simulation

```@docs
simulate
simulate_batch
EpiBranch.step!
```

## Interventions

```@docs
AbstractIntervention
Isolation
ContactTracing
RingVaccination
Scheduled
EpiBranch.initialise_individual!
EpiBranch.resolve_individual!
EpiBranch.apply_post_transmission!
EpiBranch.intervention_time
EpiBranch.reset!
EpiBranch.start_time
is_active
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
generation_R
weekly_incidence
```

## Analytical

```@docs
extinction_probability
epidemic_probability
probability_contain
proportion_transmission
proportion_cluster_size
network_R
chain_size_distribution
Borel
GammaBorel
PoissonGammaChainSize
```

## Inference

### Data types

```@docs
OffspringCounts
ChainSizes
ChainLengths
```

### Observation models

```@docs
PartiallyObserved
Reported
ThinnedChainSize
```

### Cluster-level heterogeneity

```@docs
ClusterMixed
ChainSizeMixture
```

### Real-time inference

```@docs
RealTimeChainSizes
end_of_outbreak_probability
```

### Likelihood and fitting

EpiBranch extends `Distributions.loglikelihood` with methods for each
data wrapper:

```julia
loglikelihood(OffspringCounts(data), Poisson(0.5))
loglikelihood(ChainSizes(data), NegBin(0.8, 0.5))
loglikelihood(ChainLengths(data), Poisson(0.5))
loglikelihood(ChainSizes(data), model; interventions=[iso])
```

It also extends `Distributions.fit` with MLE methods for chain-size and
chain-length data, whose likelihoods are not provided by Distributions.jl:

```julia
fit(Poisson, ChainSizes(data))
fit(NegativeBinomial, ChainSizes(data))
fit(Poisson, ChainLengths(data))
fit(NegativeBinomial, ChainLengths(data))
```

For raw offspring counts, use `Distributions.fit(Poisson, x)` directly,
or plug `loglikelihood(OffspringCounts(x), NegBin(R, k))` into Optim.jl
or Turing's `maximum_likelihood`.

See the [chains tutorial](@ref "Chain statistics, likelihood, and fitting")
for examples.

## Init functions

```@docs
Disease
clinical_presentation
demographics
compose
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
EpiBranch.post_isolation_transmission
EpiBranch._post_isolation_transmission
EpiBranch.population_size
EpiBranch._create_individual
EpiBranch.logsumexp
EpiBranch.required_fields
EpiBranch._validate_required_fields
EpiBranch._column_order
EpiBranch._with_start_time
EpiBranch._enforce_start_time!
EpiBranch._chain_length_ll_negbin
EpiBranch._borel_logpdf
EpiBranch._gammaborel_logpdf
EpiBranch._single_type_offspring
EpiBranch._empirical_ll
EpiBranch._golden_section_min
```
