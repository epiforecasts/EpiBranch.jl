**Breaking.** `Isolation`'s `isolation_duration` no longer defaults to `Inf`
and must be passed explicitly, as `onset_to_isolation_delay` already is. No
policy isolates indefinitely, and a silent default meant a caller who did not
know the parameter existed got it anyway. Pass `Inf` to keep the previous
behaviour (isolation to the end of the infectious period), or a finite value
or distribution to give it a release time.
