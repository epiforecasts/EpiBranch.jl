On the continuous-time race (`NetworkProcess`, `RoutedNetwork` and
`HouseholdProcess`), `ContactTracing` now reaches a neighbour infected
earlier than the case being traced, and grows the ring through it to its
own contacts, rather than skipping it as already settled. Quarantining
such a neighbour still blocks the transmissions it has not yet made, the
way `SourceTracing` already blocks an infector found through its own case.
Previously a case's already-infected neighbours, and anyone reachable only
through them, were never traced.
