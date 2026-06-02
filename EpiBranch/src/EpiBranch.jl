"""
EpiBranch — branching-process simulation and inference for
infectious-disease outbreaks.

The umbrella package. `using EpiBranch` re-exports the public surface
of every constituent package:

- [`EpiBranchCore`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchCore) — types, abstract types, hook protocols, helpers.
- [`EpiBranchProcess`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiBranchProcess) — engine, `BranchingProcess`, stopping rules,
  attribute builders.
- [`EpiInterventions`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiInterventions) — `Isolation`, `ContactTracing`, vaccinations,
  `Scheduled`.
- [`EpiTransitions`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiTransitions) — `Reporting`, `Hospitalisation`, `Death`,
  `Recovery`.
- [`EpiObservation`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiObservation) — `PerCaseObservation`, `Observed`,
  `ThinnedChainSize`.
- [`EpiOutput`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiOutput) — `linelist`, `contacts`, `chain_statistics`,
  summary helpers.
- [`EpiAnalytics`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/EpiAnalytics) — chain-size distributions, likelihoods, fitting,
  end-of-outbreak probability.

Users who want only a subset (e.g. just adding a custom intervention
type, or doing closed-form analytics without the engine) can depend
on the relevant sub-packages directly.
"""
module EpiBranch

using Reexport

@reexport using EpiBranchCore
@reexport using EpiBranchProcess
@reexport using EpiInterventions
@reexport using EpiTransitions
@reexport using EpiObservation
@reexport using EpiOutput
@reexport using EpiAnalytics

end # module
