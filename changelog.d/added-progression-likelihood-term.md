`progression_loglik` scores a `ModelSpec`'s `progression` against a case's
clinical timeline, the natural-history counterpart to
`pairwise_surv_loglik`. It works out of the box for the built-in
transitions; a custom `AbstractClinicalTransition` needs its own
`EpiBranch.transition_loglik` method to be scored this way.
