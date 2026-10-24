`HouseholdFinalStatuses`, dispatched through `loglikelihood(data, model; window, susceptibility, infectiousness)`,
scores a household study's final statuses (who was ever infected, with
`missing` for anyone not followed up, and no infection times at all) under the
community hazard and within-household kernel of a `HouseholdProcess`. A
`ModelSpec` reads the infectious window from its progression instead of
taking it as an argument.
