**Breaking.** `PerCaseObservation`, the `quarantine_on_trace` keyword of
`ContactTracing`, the `AlwaysEligible` and `NoTracing` aliases and the
`generate_offspring(::BranchingProcess, ...)` method are removed. Report cases
with a `Reporting` step in `progression` instead of `PerCaseObservation`; pass
`action = Quarantine(duration = Inf)` for `quarantine_on_trace = true` and
`action = FlagOnly()` for `false`; use `TraceEveryone` and `TraceNobody` for the
two aliases. `generate_offspring` stays as the function a model of your own
extends. The offspring of a `BranchingProcess` come from its
`Infectiousness` windows.
