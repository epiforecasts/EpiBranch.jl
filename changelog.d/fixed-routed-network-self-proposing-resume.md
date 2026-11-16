A `RoutedNetwork` route with a `reach` that proposes its own contacts, wrapped for
tracing in `_TracedProposals`, now continues with `simulate(model, state;
max_time)` the same way the bare route does. The wrapper already forwarded
`proposes_own_contacts` and the proposals themselves, but not the redraw a
window with no closing time needs beyond the time cap; continuing such a route stopped
with an error naming the route.
