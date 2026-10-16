`CapacityConstrained` now charges a dose to the budget of the day it is
actually given rather than the day the case proposing it was settled on, so a
daily (or other periodic) vaccination capacity can be modelled. Doses are
offered in the order the engine settles cases, but a dose dated to a day
already full is turned down even when a later-settled case's dose for an
earlier, still-open day is admitted first. `capacity_time_key` is now needed
on every capacity-limited intervention, not only one used with
`carry_over = false`.
