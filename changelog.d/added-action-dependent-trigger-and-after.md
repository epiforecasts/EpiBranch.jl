`is_triggered!` takes the `InterventionAction` `Scheduled` is deciding on as
an optional third argument, so a `start_after` trigger can gate an action by
whom it is for, such as a group's own case count, rather than only by the
population as a whole. `After(trigger, delay)` opens a fixed delay after
`trigger` itself opened, as returned by the new `opened_at`, which
`Infections` and `ReportedCases` both implement.
