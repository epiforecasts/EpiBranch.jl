`Reporting`, `Hospitalisation`, `Death` and `Recovery` now resolve `from` the
same way `Transition` does: `:infection` reads the case's infection time, and
any other state name resolves to its `:*_time` key, so `from = :onset` reads
`:onset_time` and `from = :infection` reads the infection time instead of a
state-dict key that is never set. Previously both read `from` as a raw
state-dict key: neither ever resolved, and the step silently never
happened for any case. The existing key form (`from = :onset_time`) still
works as before.

A `from` that names neither a state any step or attribute produces nor an
existing key now raises an error at start-up, since a step timed from it
would otherwise be skipped for every case without any warning.
