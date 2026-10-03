`RoutedNetwork` resolves a live `StatefulKernel` on any of its routes and
redraws its pending contacts as the records it reads change, as
`NetworkProcess` and `HouseholdProcess` already did; a route's kernel used
to be passed to the race unresolved, so a live kernel there silently never
refreshed. Giving two routes live kernels that read different records
raises an error rather than silently racing against only one of them.
