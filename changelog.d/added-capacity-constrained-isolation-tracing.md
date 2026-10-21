`Isolation` and the `Quarantine` action of `ContactTracing` can both be
wrapped in `CapacityConstrained`: isolation as a number of admissions per
period, and quarantine as a number of contacts followed up per period. Once
the budget runs out, a contact is still found and still grows a ring;
`CapacityConstrained` no longer disables tracing outright.
