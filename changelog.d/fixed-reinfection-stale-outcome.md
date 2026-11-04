A reinfection that reaches no terminal step of its own no longer keeps the
outcome from the episode it replaced. Previously, `:outcome` and
`:outcome_time` survived into the new episode untouched, so `outcome_time`
read a time from before the reinfection: `is_isolated` and isolation-based
tracing never started for the new episode, and `Reporting` and
`Hospitalisation` counted the new episode's events as happening after an
outcome it had not reached.
