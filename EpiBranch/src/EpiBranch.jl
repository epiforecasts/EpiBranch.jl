"""
EpiBranch — branching-process simulation and inference for
infectious-disease outbreaks.

The umbrella package. `using EpiBranch` re-exports the public surface
of every constituent package:

- [`EpiBranchCore`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchCore) — types, abstract types, hook protocols, helpers.
- [`EpiBranchProcess`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchProcess) — engine, `BranchingProcess`, stopping rules,
  attribute builders.
- [`EpiBranchInterventions`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchInterventions) — `Isolation`, `ContactTracing`, vaccinations,
  `Scheduled`.
- [`EpiBranchTransitions`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchTransitions) — `Reporting`, `Hospitalisation`, `Death`,
  `Recovery`.
- [`EpiBranchObservation`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchObservation) — `PerCaseObservation`, `Observed`,
  `ThinnedChainSize`.
- [`EpiBranchOutput`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchOutput) — `linelist`, `contacts`, `chain_statistics`,
  summary helpers.
- [`EpiBranchAnalytics`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchAnalytics) — chain-size distributions, likelihoods, fitting,
  end-of-outbreak probability.

Users who want only a subset (e.g. just adding a custom intervention
type, or doing closed-form analytics without the engine) can depend
on the relevant sub-packages directly.
"""
module EpiBranch

using Reexport

@reexport using EpiBranchCore
@reexport using EpiBranchProcess
@reexport using EpiBranchInterventions
@reexport using EpiBranchTransitions
@reexport using EpiBranchObservation
@reexport using EpiBranchOutput
@reexport using EpiBranchAnalytics

end # module
