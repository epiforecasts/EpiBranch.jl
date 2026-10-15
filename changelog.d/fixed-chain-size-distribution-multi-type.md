`chain_size_distribution` no longer errors on a multi-type or multi-window
model, or on a single-type offspring with no closed-form chain-size law
(e.g. `Binomial`), when called with no keyword arguments. It now falls back
to the simulation-based wrapper in those cases, the same as
`loglikelihood(ChainSizes, model)` already did.
