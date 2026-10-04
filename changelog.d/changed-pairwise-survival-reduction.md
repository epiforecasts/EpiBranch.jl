`pairwise_surv_loglik` and `pairwise_surv_loglik_by_component` now share one
implementation of their two accumulation passes, dispatched on a
`PairwiseReduction`. A grouping other than the built-in total and
per-component ones is written from outside the package as a `PairwiseReduction`
subtype with `ngroups` and `group` methods.
