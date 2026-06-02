# EpiBranch.jl

**EpiBranch.jl** is a [Julia](https://julialang.org/) package for stochastic branching process simulation
of infectious disease outbreaks. It unifies the functionality of
several R packages (cf.
[ringbp](https://github.com/epiforecasts/ringbp),
[simulist](https://github.com/epiverse-trace/simulist),
[epichains](https://github.com/epiverse-trace/epichains),
[superspreading](https://github.com/epiverse-trace/superspreading),
[pepbp](https://github.com/sophiemeakin/pepbp)) in
a single package with one shared simulation engine.

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

model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.5, delay = Exponential(1.5))

rng = StableRNG(42)
results = simulate_batch(model, 500;
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 5000),
    rng = rng,
)

containment_probability(results)
```

## Installation

**EpiBranch.jl** is not yet registered. The repository holds seven
Julia packages plus a main `EpiBranch` package that re-exports
everything they contain. For most users, the main package is what you
want:

```julia
using Pkg
Pkg.add(url = "https://github.com/epiforecasts/EpiBranch.jl", subdir = "EpiBranch")
```

The `subdir = "EpiBranch"` argument tells Pkg which sub-folder of the
repository holds the package, and is needed only while the package is
unregistered. Once `EpiBranch` is on the Julia General Registry, the
install command becomes the usual `Pkg.add("EpiBranch")` with no
`subdir`.

If you only need part of the functionality — for example, just the
closed-form likelihoods without the simulator, or just the intervention
types — you can install one of the smaller packages on its own with
the same `subdir = "..."` pattern. The available packages are
`EpiBranchCore`, `EpiBranchProcess`, `EpiBranchInterventions`,
`EpiBranchTransitions`, `EpiBranchObservation`, `EpiBranchOutput`, and `EpiBranchAnalytics`.
