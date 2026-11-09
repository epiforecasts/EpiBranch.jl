`pairwise_surv_loglik`'s `external_hazard` accepts a `SumHazard`, built with
`+` from a `MassActionHazard` and any other source, such as a constant or
calendar importation hazard. The sum explains index cases when any active
source can, so mass action plus an importation rate scores them as importations
instead of conditioning on them.
