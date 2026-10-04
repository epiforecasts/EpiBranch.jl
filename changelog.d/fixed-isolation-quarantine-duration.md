Isolation and quarantine previously had a start time and no end, so a contact
quarantined and never infected, then infected later through another route,
stayed blocked from onward transmission by a quarantine that would have
lapsed long before. `Isolation` now takes an `isolation_duration` and
`ContactTracing`'s `Quarantine` action a `duration` (a value, a distribution,
or a function of the individual, matching `onset_to_isolation_delay`), giving
each removal a release time (`EpiBranch.isolation_release_time`) after which
the block lapses. Both default to `Inf`, reproducing the previous behaviour:
isolation lasts until the end of the infectious period unless a finite
duration is configured. `Risk` gained a matching `release_time` field, and the
line list reports a `date_isolation_release` column once a finite duration is
in use, and its two-argument positional form still builds a risk that never
lapses.

On a generation-based process the release ends a per-contact block, so a
contact after it is not blocked. On the continuous-time (Sellke) models an
infectious window carries one closing time, so a case isolated during its
infectious period stays removed for the rest of it; what the release changes
there is a case whose removal had already lapsed before it was infected,
which is the case the duration exists for.
