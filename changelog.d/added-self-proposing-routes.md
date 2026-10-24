A `RoutedNetwork` route's `reach` can now propose its own contacts instead of
listing them, for a route that is cheaper to sample than to list, such as a
spatial kernel over a large population or contact with everyone in an area.
Declare `EpiBranch.proposes_own_contacts(route) = true` and implement
`EpiBranch.propose_along` and `EpiBranch.route_pair_kernel` (and
`EpiBranch.pending_beyond` for a route sampled only up to the run's time
limit); `RoutedNetwork` then needs `n`, the node count, once no route has
an adjacency list to read it from. Contact tracing on such a route reads back
exactly the contacts its proposals used, rather than a fresh, costly walk of
its full reach. Mass action is rewritten on this seam.
