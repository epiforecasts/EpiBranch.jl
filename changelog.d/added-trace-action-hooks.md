`ContactTracing` forwards two further hooks to the `TraceAction` it carries,
each with a no-op default: `EpiBranch.intervention_actions`, so a trace action
can propose its own `InterventionAction`s, rationed by `CapacityConstrained`
through its own `capacity_key`; and `EpiBranch.on_infection_settled!`, so one
can act once a contact's own infection time is fixed, rather than at the trace
itself. `EpiBranch.shorten_quarantine!(ind, start, old_release, new_release)`
ends a quarantine `Quarantine`'s `apply_trace!` recorded, before the duration
it drew has run, keeping the quarantine's own stretches and the shared removal
history the likelihood reads consistent with each other. Together these let a
trace action release a contact early on information only available after the
trace, such as a negative test, without editing the package; see "A trace
action that proposes its own actions" in the extending guide for
`TestAndRelease`, a worked example.
