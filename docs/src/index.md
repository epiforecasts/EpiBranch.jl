# EpiBranch.jl

**EpiBranch.jl** is a composable [Julia](https://julialang.org/) engine for
branching-process models in epidemiology. It does both stochastic simulation and
analytical inference, and uses closed-form results where they exist. A flexible
extension interface includes interventions, host attributes, and structured
transmission over [contact networks](tutorials/networks.md) and within
[households](tutorials/households.md), each a companion package built on the same
engine. See [Extending EpiBranch](tutorials/extending.md) for the extension points.

It started as a unification of five R packages (cf.
[ringbp](https://github.com/epiforecasts/ringbp),
[simulist](https://github.com/epiverse-trace/simulist),
[epichains](https://github.com/epiverse-trace/epichains),
[superspreading](https://github.com/epiverse-trace/superspreading) and
[pepbp](https://github.com/sophiemeakin/pepbp)) but adds type-structured
offspring and a single interface across simulation and analytical methods.

EpiBranch.jl separates the offspring draw from timing and interventions:
the branching process draws contacts, generation times are assigned
independently, and interventions act as competing risks that decide
whether each contact is actually infected. For more on the design,
see [Design](design.md).

## Quick start

```@example quickstart
using EpiBranch
using Distributions
using StableRNGs

iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
ct = ContactTracing(OnSymptomOnset(), 0.5, Exponential(1.5), Quarantine())

# A model is a process together with the population (attributes) and the
# policy in force (interventions); simulate reads both from it.
model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5);
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
)

rng = StableRNG(42)
results = simulate(model, 500; max_cases = 5000, rng = rng)

containment_probability(results)
```

## Installation

**EpiBranch.jl** is not yet registered. It can be installed from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/epiforecasts/EpiBranch.jl")
```
