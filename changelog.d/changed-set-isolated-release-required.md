**Breaking.** `set_isolated!` takes a required `release_time` keyword, where
0.1.0 took only an isolation time and recorded no release at all. The accessor
is where an intervention written outside the package removes a host from
transmission, so it is the last place a removal could inherit "never released"
without saying so, and a quarantine that outlives its cause is the bug that
follows. Pass `release_time = Inf` for a removal that never ends, or the time
the block lapses.
