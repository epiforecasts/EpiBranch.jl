The Extending guide now documents three seams the package already relied on:
`AbstractStoppingRule`/`should_stop` for ending a run on a custom condition,
`AbstractEffectMode` for a new way of turning a vaccine dose's sampled
efficacy into a stored block probability, and `terminal_target` for a
terminal clinical transition to be seen by the `until`-coverage check. The
two `AbstractEffectMode` hooks, previously `_realised_efficacy` and
`_realise_prior_dose!`, drop their leading underscore to match
`terminal_target`'s and `should_stop`'s: a leading underscore marks a name
as private, and a seam has to be reachable by name from outside the
package.
