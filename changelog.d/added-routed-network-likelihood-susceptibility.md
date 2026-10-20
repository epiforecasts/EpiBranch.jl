The routed-network likelihood accounts for vaccination.
`pairwise_surv_loglik(windows, data::RoutedNetworkInfections)` takes a
`susceptibility` effect, such as a candidate `VaccineEffect`, and
`loglikelihood(data, model)` reads it from the model's composed interventions,
exactly as the household and network likelihoods do. A susceptible the effect
modifies is scored as the weighted mixture of their escape and infection,
summed over every route and the external hazard, under each of the effect's
components. Previously `loglikelihood(data, model)` refused a `RoutedNetwork`
model with an intervention that changes susceptibility, such as
`RingVaccination`, so a case-triggered study could not be fitted with the
routes it was simulated on.
