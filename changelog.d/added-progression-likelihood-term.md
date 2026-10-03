`progression_loglik`, the natural-history counterpart to
`pairwise_surv_loglik`, gives the log-density of a case's clinical timeline
under a `ModelSpec`'s `progression`. It works out of the box for the built-in
transitions; a custom `AbstractClinicalTransition` needs its own
`EpiBranch.transition_loglik` method to be evaluated this way.

A transition an abort undid is censored at the abort, and a shared-draw group
built by `exclusive_probabilities` scores the bucket its draw selected, so an
exact case-fatality ratio reaches the likelihood.
