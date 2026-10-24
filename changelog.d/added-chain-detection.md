A new observation model, `ChainDetection`, records a whole chain with a
probability that depends on its size, as when a cluster is found through
investigation rather than case by case. The common case is built from a
single number: each case independently brings its chain to attention with
probability `q`. Any other size-dependent detection rule can be passed as a
function instead. `observe` turns it into a `DetectedChainSize`, usable in a
closed-form likelihood like the other observation models.
