`CapacityConstrained` takes an `on_refusal` policy for an action its budget
turns away: `DropRefused()` (the default, as before) discards it, and
`DeferRefused()` books it into the first later period with room. A new
`HeldCapacity` wrapper caps how many people hold a place at once, such as
quarantine or isolation beds: a place is taken when an action is admitted and
given back when its stay ends or `release_capacity!` is called, and
`held_usage` reports the places held. New policies are written as a
`RefusalPolicy` subtype with a `refused_action` method.
