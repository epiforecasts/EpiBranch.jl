The pairwise likelihoods account for vaccination. `pairwise_surv_loglik` and
`pairwise_surv_loglik_by_component` take a `susceptibility` effect, such as a
candidate `VaccineEffect`, and the household and network
`loglikelihood(data, spec)` methods read it from the composed interventions.
From a host's immunity time on, a `LeakyMode` dose multiplies every hazard the
host faces by `1 - efficacy` (scaled by `waning` where given), and an
`AllOrNothingMode` dose makes the host's contribution a mixture over responder
status, smooth and so differentiable in `efficacy`. `household_infections` and
`network_infections` record each host's immunity time in `host_times` when the
model composes a vaccination. A custom effect declares itself through
`EpiBranch.susceptibility_components`, returning weighted
`EpiBranch.HazardScaling` components, and `EpiBranch.susceptibility_host_times`.
Previously the pairwise likelihoods evaluated a vaccinated population as if nobody
had been dosed.
