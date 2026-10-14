`PerCaseObservation` is deprecated in favour of a `Reporting` progression
step, which `simulate`, `progression_loglik`, the chain-size likelihood and
`end_of_outbreak_probability` all read directly, rather than splitting the
same reporting process across a `progression` step and an observation
model. A model may keep both while callers migrate.

Two behaviours differ between the two routes. `PerCaseObservation` falls
back to the infection time when its anchor (`:onset_time` by default) is
`NaN`, as for an asymptomatic case under `clinical_presentation`; `Reporting`
skips such a case instead, so it is never reported. Pass a `from` function to
`Reporting` for the fallback. `PerCaseObservation` also draws each case's
report after the simulation run, where `Reporting` draws it during the run
alongside the other progression transitions; migrating a model from one to
the other therefore changes its simulated results for a fixed seed.
