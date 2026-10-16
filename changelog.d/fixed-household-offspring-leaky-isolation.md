`household_offspring` only subtracted isolated time from community
person-time for complete isolation: `removal_gap_host_times` returned no
stretch once `post_isolation_transmission` was above zero, so leaky isolation
left R* close to its value with no isolation at all.

`Isolation` now records its stretch whenever it can hand the case back,
leaky, finite-duration, or both, and only skips a perfect isolation that
never releases, since that already closes the infectious window outright. A
new trait, `removal_block_fraction`, gives the fraction of transmission a
recorded stretch blocks (1 by default, `1 - post_isolation_transmission` for
`Isolation`); `household_offspring` uses it to weight each stretch instead of
assuming full blocking.
