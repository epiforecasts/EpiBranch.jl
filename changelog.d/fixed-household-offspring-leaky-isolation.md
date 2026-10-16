`household_offspring` only subtracted isolated time from community
person-time for perfect isolation: `removal_gap_host_times` returned no
stretch once `post_isolation_transmission` was above zero, so leaky isolation
left R* close to its value with no isolation at all.

`Isolation` now records its stretch whenever it can hand the case back,
leaky, finite-duration, or both, and only skips a perfect isolation that
never releases, since that already closes the infectious window outright.
`household_offspring` now weights each recorded stretch by
`removal_block_probability` (1 by default, `1 - post_isolation_transmission`
for `Isolation`) instead of assuming full blocking, and checks every
recorded stretch's key against the model's interventions the same way the
pairwise likelihood does, rather than silently dropping one a custom
intervention forgot to declare.
