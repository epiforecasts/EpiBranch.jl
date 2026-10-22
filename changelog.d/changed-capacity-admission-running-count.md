`CapacityConstrained` now keeps a running count of who has already drawn on
each of its resources, grown as each action is admitted, rather than
rescanning the whole population every time an action is proposed. A run with
many settled cases and a resource only a few of them ever draw on no longer
pays for everyone else on every one of those cases. `capacity_usage` and
isolation's own capacity check still rescan the population, since neither is
called often enough for the scan to matter.
