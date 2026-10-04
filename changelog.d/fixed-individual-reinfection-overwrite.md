Assigning a new `infection_time` to an `Individual` that already has an
`:outcome_time` now throws an `ArgumentError` instead of silently overwriting
the episode, which used to leave the earlier outcome behind with a record
that recovers (or dies) before it is infected. `Individual` still holds only
one infection episode; this turns the resulting data corruption into a clear
failure rather than fixing reinfection itself.
