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
in use.
