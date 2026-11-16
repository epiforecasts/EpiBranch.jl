`pairwise_surv_loglik` accepts a `SumHazard` as its `external_hazard`, built with
`+` from a `MassActionHazard` and any other source, such as a constant or
calendar importation hazard. The sum explains index cases when any active
source can. Mass action plus an importation rate scores them as importations
instead of conditioning on them.
