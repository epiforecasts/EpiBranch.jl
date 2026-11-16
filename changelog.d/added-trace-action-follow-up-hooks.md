`ContactTracing` forwards `intervention_actions` and `on_infection_settled!`
to its `TraceAction`. A trace action can propose a follow-up, such as a test
some days after the trace, subject to the usual schedule and capacity
constraints. It can then read the result once the infection status of the
contact is known. A removal `apply_trace!` recorded can then be shortened
with `revise_removal!` from `on_infection_settled!`, before the contact has
made any transmissions of its own, which is how a negative test can release
a quarantine earlier than its recorded duration. The Writing an intervention
guide works through this as a `TestAndRelease` action.
