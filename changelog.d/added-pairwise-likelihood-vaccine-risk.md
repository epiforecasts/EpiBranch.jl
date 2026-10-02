`pairwise_surv_loglik` takes a `vaccine` argument (a candidate `VaccineEffect`)
that discounts a susceptible's escape probability and event hazard from its
vaccine-induced immunity time on: a constant `1 - efficacy` factor under
`LeakyMode` (continuously scaled by `waning` where given), and a two-component
mixture over responder status under `AllOrNothingMode`, smooth and so exactly
differentiable in `efficacy`. `InfectionLayer` gains a matching `immunity_time`
field, which `household_infections` and `network_infections` fill in
automatically; `HouseholdInfections`/`NetworkInfections` and the structured
`loglikelihood(data, spec)` methods read a compatible vaccination straight off
`interventions`. Previously the pairwise likelihoods ignored vaccination
state entirely, scoring a vaccinated population as if nobody had been dosed.
