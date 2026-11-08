`pairwise_surv_loglik` no longer returns `NaN` when a row lies beyond the upper bound of a truncated kernel. Such a row now contributes zero hazard and drops out of the sum over infectors.
