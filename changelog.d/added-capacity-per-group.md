`CapacityConstrained` can now split an intervention's budget into more than
one, rather than holding a single pool for everyone: a new dispatched method,
`EpiBranch.resource_key(intervention, individual, state)`, names which budget
each person draws on — a district, a treatment centre — and
`capacity_usage(cc, state, key)` reads one group's own usage. The default,
`nothing`, keeps every existing use of `CapacityConstrained` on one shared
budget, exactly as before.
