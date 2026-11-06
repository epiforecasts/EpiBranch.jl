A reinfection that reaches no terminal step of its own no longer keeps the
outcome of the episode it replaced. Previously, `:outcome` and
`:outcome_time` survived into the new episode untouched, leaving
`outcome_time` at a time from before the reinfection. `is_isolated` and
isolation-based tracing never started for the new episode, and `Reporting`
and `Hospitalisation` counted events in the new episode as happening after
an outcome it had never reached.
