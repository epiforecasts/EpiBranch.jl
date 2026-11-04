`household_offspring` closed a case's infectious window at the earliest of
the process's `until` states and the interventions' removal time, missing a
post-exposure abort (`RingVaccination`'s `post_exposure_efficacy`, say), which
undoes the recovery transition and leaves `recovered_time` at `Inf`. It threw
an `ArgumentError` claiming the progression had no terminal transition for
every aborted case, even though it did. The window now closes the same way
`simulate` and the pairwise likelihood do, through `window_open` and
`window_close`, which also honour the abort.
