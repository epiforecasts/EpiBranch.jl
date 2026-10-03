A pair kernel declares the host records its hazards depend on through
`EpiBranch.watched_records`, and a `StatefulKernel` with a projection takes
them as `watches`. A continuous-time race watches the union of the records its
routes declare and redraws only the pending contacts of the routes reading a
record that moved, so a `RoutedNetwork` can carry a record-reading kernel on
every route, and `NetworkProcess` is the one-route case of the same mechanism.

A record that an attribute builder, a `Transition` or an observation model
moves is now followed as well: the engine no longer infers from the
intervention stack that nothing else can move one.
