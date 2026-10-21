`TestAndRelease` is a new `TraceAction` for `ContactTracing`: it quarantines a
traced contact as `Quarantine` does, for up to a backstop `duration`, but
releases them early once a test taken `test_day` days after the trace comes
back negative. The test is read once the contact's own infection time is
settled, rather than when the duration would otherwise have been drawn; that
way it sees the contact's true infection status rather than whatever was
known at the trace. The test itself is an `InterventionAction`, so wrapping the
`ContactTracing` in `CapacityConstrained` rations how many tests are
available per day; a contact turned down keeps its full `duration`.
