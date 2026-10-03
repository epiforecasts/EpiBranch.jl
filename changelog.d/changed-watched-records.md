A pair kernel declares the host records its hazards depend on through
`EpiBranch.watched_records`, and a `PairKernel` whose `state` is a projection
now takes them as `watches`, which is required: an existing
`PairKernel(callback; state = project)` raises an `ArgumentError` until it
declares what the projection reads. A continuous-time race watches the union of
the records its routes declare and redraws only the pending contacts of the
routes reading a record that moved, so a `RoutedNetwork` can hold a
record-reading kernel on every route, and `NetworkProcess` is the one-route
case of the same mechanism.

A record that an attribute builder, a `Transition` or an observation model
moves is now followed as well: the engine no longer infers from the
intervention stack that nothing else can move one. A household model puts every
household on one clock when its kernel declares a record, whether or not an
intervention is in the stack.
