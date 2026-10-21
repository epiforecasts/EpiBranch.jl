`Isolation` and `ContactTracing`'s `Quarantine` action can both be wrapped
in `CapacityConstrained`: isolation as a number of admissions per period,
and a quarantine as a number of contacts followed up per period, with a
contact still found and still growing a ring once the budget runs out.
`CapacityConstrained` no longer disables tracing outright.
