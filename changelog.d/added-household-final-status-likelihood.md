`HouseholdFinalStatuses`, dispatched through `loglikelihood(data, model; window, susceptibility, infectiousness)`,
scores a household study's final statuses under a `HouseholdProcess`'s
community hazard and within-household kernel: who was ever infected, with
`missing` for anyone not followed up, and no infection times at all. A
`ModelSpec` reads the infectious window from its progression instead of
taking it as an argument.
