`ContactTracing(depth = 2)` and beyond now reaches contacts-of-contacts on
`HouseholdProcess`, `NetworkProcess` and `RoutedNetwork`. The ring previously
grew past an uninfected contact only on the generation engine, whose
`keep_active` hook has no continuous-time counterpart: the race traces a
case's contacts once, when that case itself is settled, but an uninfected
ring member is never settled. The ring now grows breadth-first over the
model's own contact structure whenever a traced contact is left with ring
budget, matching the generation engine's reach.
