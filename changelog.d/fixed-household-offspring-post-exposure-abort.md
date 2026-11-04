`household_offspring` closed a case's infectious window at the earliest of
the `until` states on `process` and the removal time from the interventions,
missing a post-exposure abort (`post_exposure_efficacy` on `RingVaccination`,
say), which undoes the recovery transition and leaves `recovered_time` at
`Inf`. It threw an `ArgumentError` claiming the progression had no terminal
transition for every aborted case, even though it did. The window now closes
the same way `simulate` and the pairwise likelihood do, through
`window_open` and `window_close`, which also honour the abort.
