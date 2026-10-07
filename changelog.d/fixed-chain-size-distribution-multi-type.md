`chain_size_distribution` no longer errors on a multi-type or multi-window
model when called with no keyword arguments. It now falls back to the
simulation-based wrapper in that case, the same as
`loglikelihood(ChainSizes, model)` already did.
