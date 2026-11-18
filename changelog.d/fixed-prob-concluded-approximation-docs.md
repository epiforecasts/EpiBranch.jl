The docs for `prob_concluded` in `loglikelihood(::ChainSizes, ...)`,
`chain_size_distribution` and the chains tutorial now say that the mixture
`π P(X = x) + (1 − π) P(X ≥ x)` is an approximation to the exact probability of
the data, and that `π` depends on the offspring parameters and so has to be
recomputed at each parameter value in a fit.
