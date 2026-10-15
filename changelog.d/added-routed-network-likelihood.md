`RoutedNetwork` has a likelihood. `routed_network_infections` reads a
simulated routed-network outbreak into a `RoutedNetworkInfections`, with one
infectious window per route, and `pairwise_surv_loglik(windows, data)` (and
`loglikelihood(data, model)`) scores it: survival sums every route's hazard
and an infected person's event term is a log-sum-exp over every route and
possible infector, so a model with household and community contact and
self-isolation at home can now be fitted as well as simulated. A lapsing
removal, such as an `Isolation` with a finite `duration`, is taken out of a
route's exposure only where that route lists
`EpiBranch.INTERVENTION_REMOVAL`, matching what the simulation does.

A route kernel that is a `PairKernel`, and an intervention that changes
susceptibility (such as vaccination), are not yet represented in this
likelihood and raise a clear error rather than a silently wrong answer.
