Setting `RoutedMassActionHazard` as a process's own `external_hazard` now gives
the same clear `ArgumentError` a plain `MassActionHazard` would, rather than a
bare `MethodError`: such a hazard is prepared with no outbreak to read, and now
delegates to the wrapped hazard's own rejection instead of falling through to
the generic calendar-schedule fallback. `routed_network_infections` likewise
rejects a route whose `reach` proposes its own contacts up front, as the
routed-network likelihood does not yet read such proposals back into a fixed
adjacency list.
