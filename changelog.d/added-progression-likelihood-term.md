`progression_loglik`, the natural-history counterpart to
`pairwise_surv_loglik`, gives the log-density of a case's clinical timeline
under a `ModelSpec`'s `progression`. It works out of the box for the built-in
transitions; a custom `AbstractClinicalTransition` needs its own
`EpiBranch.transition_loglik` method to be evaluated this way.

A transition an abort undid is censored at the abort rather than read as a gate
that failed, and a shared-draw group built by `exclusive_probabilities` keeps
the width of the bucket its draw selected, there as anywhere else, so an exact
case-fatality ratio reaches the likelihood. The draw itself now survives the
abort, so the group still partitions a case the abort cut short.

`EpiBranch.transition_term` is the gate term a custom `transition_loglik`
calls: it handles an aborted infection and a shared-draw gate, which reading
`probability` directly would get wrong.
