`record_removal!` and `set_isolated!` record each removal as its own entry,
returning a handle for later revision or withdrawal.
`isolation_time`, `isolation_release_time` and `removal_stretches` read the
merged view back, recomputed whenever an entry is added, revised or
withdrawn. Two new verbs act on a handle directly: `revise_removal!` changes
the release of an entry, and `withdraw_removal!` undoes it, as though it had
never been recorded. `set_isolated!` also takes an `unrecorded` keyword, so a
caller states a removal's detection status up front instead of poking
`ind.state[:_isolation_unrecorded]` afterwards.

A quarantine and an isolation applied over it now keep separate entries.
The `reset!` methods of `Isolation` and `ContactTracing` each withdraw
only the entry they recorded.
The internal `_isolated_by_isolation`, `_quarantined_by_tracing` and
`_isolation_*_before_*` keys are gone; `CapacityConstrained`'s `capacity_key`
for `Isolation` and `Quarantine` now names the handle (or handles) it keeps
instead.
