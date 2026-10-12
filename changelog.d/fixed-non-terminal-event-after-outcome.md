`Hospitalisation`, `Reporting`, and a non-terminal `Transition` no longer
record an event after the case's outcome: such an event cannot happen to a
person who has already died or recovered, so it is now reset to "did not
occur" (its own flag back to `false`, its own time back to `Inf`). The new
`EpiBranch.censor_after_outcome!` trait does this for every non-terminal
transition once `:outcome`/`:outcome_time` are set, and `transition_loglik`
reads such an event as censored at the outcome (the same way an aborted
infection is already censored at the abort), whether or not a hand-built
individual's own state agrees.

A custom non-terminal transition picks this up by overriding
`censor_after_outcome!` alongside the flag/time keys its own
`resolve_individual!` writes; the clinical transitions tutorial shows the
pattern, and also how to chain a terminal transition onto `Hospitalisation`
with `from = :admission_time` where admission should change the outcome
rather than sit alongside it.
