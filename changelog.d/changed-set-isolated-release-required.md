**Breaking.** `set_isolated!`'s `release_time` is now a required keyword rather
than defaulting to `Inf`. The accessor is where an intervention written outside
the package removes a host from transmission, so it was the last place a
removal could inherit "never released" without saying so, and a quarantine that
outlives its cause is the bug that follows. Pass `release_time = Inf` for a
removal that never ends, or the time the block lapses.
