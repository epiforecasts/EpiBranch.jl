# API reference

## Transmission models

```@docs
BranchingProcess
```

## Types

```@docs
Individual
SimOpts
DelayOpts
OutcomeOpts
DemographicOpts
```

## Simulation

```@docs
simulate
simulate_batch
simulate_conditioned
```

## Interventions

```@docs
AbstractIntervention
Isolation
ContactTracing
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
