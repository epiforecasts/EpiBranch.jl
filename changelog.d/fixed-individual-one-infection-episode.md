`Individual` no longer loses an episode's outcome when a second infection
overwrites its live fields. A new infection on a host that already carries
one now archives the closing episode onto `Individual.episodes` (its
`infection_time`, tree position, secondary cases so far, and a snapshot of
`state`) instead of silently dropping it.

This is the storage half of reinfection after waning. A model opts in by
having its `contacts_of` keep offering a host past its first infection: the
new `susceptible_again_time` accessor and the `HostImmunity` risk source
block every such exposure until a progression transition (for example
`Transition(:susceptible_again, from = :recovered, delay = ...)`) marks that
host eligible again, the same convention `infectious_time` and
`recovered_time` already follow. No built-in model offers a host again yet,
so existing simulations are unaffected.

The pairwise survival likelihood's `InfectionLayer` does not yet represent
more than one episode per host; fitting data from a reinfection-aware model
is left for later.
