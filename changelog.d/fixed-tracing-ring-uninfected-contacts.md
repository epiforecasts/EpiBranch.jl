`ContactTracing(depth = 2)` and beyond now reaches contacts-of-contacts on
`HouseholdProcess`, `NetworkProcess` and `RoutedNetwork`. The ring previously
grew past an uninfected contact only on the generation engine: the race traces
a case's contacts once, when that case is settled, and an uninfected ring
member is never settled. The race now walks the model's own contact structure
breadth-first and asks `keep_active`, the same hook the generation engine uses,
which contacts the next hop starts from, so `ContactTracing` keeps the depth
semantics and a ring of another shape takes part by answering that hook.

A case first reached as someone else's contact is interviewed again once it
becomes a case in its own right, seeding the full-radius ring the documentation
describes. `!PreviouslyTraced()` in the eligibility expresses the
interview-once policy instead.
