The chain-size likelihood now reads a model's detection probability from a
`Reporting` step in `progression`, the same step `simulate` and
`progression_loglik` already use, rather than only from a `PerCaseObservation`
observation. Previously a model that reported half its cases through
`Reporting` scored the same chain-size likelihood as one with full reporting:
the closed form ignored `Reporting` entirely, and the simulation-based route
counted every case regardless of whether it was reported. The same model
written with `PerCaseObservation` scored correctly. `ModelSpec` now refuses a
model that specifies both a `Reporting` step and a per-case observation,
since the two describe the same reporting process and `simulate` would have
silently let the observation's own draw overwrite `Reporting`'s.
