A `RoutedNetwork` route whose `reach` proposes its own contacts, wrapped for
tracing in `_TracedProposals`, now continues with `simulate(model, state;
max_time)` the same way the bare route does. The wrapper already forwarded
`proposes_own_contacts` and the proposals themselves, but not the redraw a
capped, never-closing window still owes; continuing such a route was refused
with an error naming the route.
