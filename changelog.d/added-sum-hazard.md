`pairwise_surv_loglik`'s `external_hazard` accepts a `SumHazard`, built with
`+` from a `MassActionHazard` and any other source, such as a constant or
calendar importation hazard. Index cases are explained by the sum when any
active source can explain them, so mass action plus an importation rate scores
them as importations instead of conditioning on them.
