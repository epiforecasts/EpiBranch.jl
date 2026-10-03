A `RoutedNetwork` route's `kernel` now accepts a covariate callable, a
`PairKernel` without `state` and a per-edge vector, each resolved per pair
exactly as `NetworkProcess` resolves its edge kernel, in both the route's
targets and the continuation drawn after a blocked contact. Previously every
such form other than a shared distribution was passed unresolved to the race
and raised a `MethodError`, so covariate and contextual kernels were unusable
on a route.

A kernel that reads host records, such as a `PairKernel` given `state`, is
refused at construction. Resolving one per pair is not enough on a route: the
race redraws pending contacts from a single set of watched records, and several
routes can carry several such kernels, so a route's contacts would be drawn
from whatever the records held when they were proposed.
