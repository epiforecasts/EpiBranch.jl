`apply_post_transmission!` is deprecated. An intervention implements
`trace_contacts!` for what it does to the contacts a case reached and
`intervention_actions` for actions with a delivery time, and both engines call
these once per case. The generation engine still hands a whole generation's
contacts to an intervention with an `apply_post_transmission!` method of its
own, at its place in the list of interventions, and warns that the hook is
deprecated. Calling
it directly on a built-in intervention traces each contact from its
`parent_id` and admits the intervention's actions, with the same warning.
