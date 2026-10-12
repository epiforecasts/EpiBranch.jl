The chain-size likelihood and `end_of_outbreak_probability` now read a
model's detection probability from a `Reporting` step in `progression`, the
same step `simulate` and `progression_loglik` already use. Previously a
model that reported half its cases through `Reporting` scored the same
chain-size likelihood as one with full reporting: the closed form ignored
`Reporting` entirely, and the simulation-based route counted every case
regardless of whether it was reported. `end_of_outbreak_probability` treated
it as fully reported too, since its closed form assumes `ρ = 1`.
