**Breaking.** `Isolation`'s `isolation_duration` no longer defaults to `Inf`
and must be passed explicitly, as `onset_to_isolation_delay` already is,
since no policy isolates indefinitely and a silent default handed indefinite
isolation to a caller who did not know the parameter existed. Pass `Inf` to
keep the previous behaviour, which never releases the case, or a finite value
or distribution to give it a release time.
