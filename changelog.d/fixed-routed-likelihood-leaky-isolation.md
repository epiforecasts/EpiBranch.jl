The `RoutedNetwork` likelihood now applies the `post_isolation_transmission` of a
leaky `Isolation`. The leaky stretches stay in the escape term and the event
term of each route that lists `INTERVENTION_REMOVAL`, as in the single-route
likelihoods, instead of being scored at the full hazard.
