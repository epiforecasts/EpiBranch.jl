`ContactTracing` forwards `intervention_actions` and `on_infection_settled!`
to the `TraceAction` it holds, so a trace action can propose a further
follow-up, such as a test some days after the trace, through the ordinary
schedule and capacity funnel, and read what settles once the contact's own
infection is fixed. A removal `apply_trace!` recorded can then be shortened
with `revise_removal!` from `on_infection_settled!`, before the contact has
made any transmissions of its own, which is how a negative test can release
a quarantine earlier than its recorded duration. The Writing an intervention
guide works through this as a `TestAndRelease` action.
