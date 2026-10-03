`progression_loglik`, the natural-history counterpart to
`pairwise_surv_loglik`, gives the log-density of a case's clinical timeline
under a `ModelSpec`'s `progression`. It works out of the box for the built-in
transitions; a custom `AbstractClinicalTransition` needs its own
`EpiBranch.transition_loglik` method to be evaluated this way.
