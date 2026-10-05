Isolation and quarantine previously had a start time and no end, so a contact
quarantined and never infected, then infected later through another route,
stayed blocked from onward transmission by a quarantine that would have
lapsed long before. `Isolation` now takes an `isolation_duration` and
`ContactTracing`'s `Quarantine` action a `duration` (a value, a distribution,
or a function of the individual, matching `onset_to_isolation_delay`), giving
each removal a release time (`EpiBranch.isolation_release_time`) after which
the block lapses. `Quarantine`'s `duration` defaults to `Inf`, which never
releases the contact, reproducing the previous behaviour. `Risk` gained a
matching `release_time` field, and the
line list reports a `date_isolation_release` column once a finite duration is
in use, and its two-argument positional form still builds a risk that never
lapses.

On a generation-based process the release ends a per-contact block, so a
contact after it is not blocked. On the continuous-time (Sellke) models an
infectious window carries one closing time and cannot reopen, so the release
spares an infection acquired after a removal had lapsed and nothing else,
which is the case the duration exists for. Two removals layered on one case
are held as the interval covering both where they meet; where they do not, the
later one is kept, the earlier being spent before it begins. A duration of zero
removes nobody on either engine, and a negative one is refused.
