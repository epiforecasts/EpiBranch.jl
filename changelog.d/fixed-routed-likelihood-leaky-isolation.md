The `RoutedNetwork` likelihood now applies a leaky `Isolation`'s
`post_isolation_transmission`: the leak's stretches stay in the escape term and
the event term of each route that lists `INTERVENTION_REMOVAL`, as the
single-route likelihoods do, rather than being scored at the full hazard.
