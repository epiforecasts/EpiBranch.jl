Assigning a new `infection_time` to an `Individual` that already has an
`:outcome_time` now throws an `ArgumentError` instead of silently overwriting
the episode and leaving the earlier outcome behind, where the record recovers
(or dies) before it is infected. `Individual` still holds only one infection
episode; this turns the resulting data corruption into a clear failure
rather than fixing reinfection itself.
