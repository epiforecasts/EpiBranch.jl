`MassVaccination` now has a closed-form `analytic_offspring_effect`, which lets
`extinction_probability`, `probability_contain`, `reproduction_number` and the
other analytical functions work on a `ModelSpec` holding one instead of
stopping with an error. For a campaign already complete before the simulation
starts (`eligibility_time` a number with
`eligibility_time + delay_to_immunity <= 0`), a `NegBin(R, k)` or `Poisson(R)`
offspring distribution is thinned to `R * (1 - coverage * efficacy)`, each
secondary case prevented independently. `MassVaccination` also takes a new
`coverage` keyword (default 1, as for `RingVaccination` and
`GroupVaccination`): a rollout reaching less than the whole eligible
population no longer needs a hand-written `eligibility_time` function to
express it.
