`chain_size_distribution` now works for any discrete offspring law, not just
Poisson, negative binomial and `ClusterMixed(Poisson, ::Gamma)`: for anything
else, such as a `DiscreteNonParametric` law or a `GeneratingFunctionOffspring`
built from a probability mass function, it gives the exact chain-size
distribution from the hitting-time theorem, at a cost that grows with the
cube of the largest chain size needed. `IndexChainSize` and
`extinction_probability` pick this up for the same offspring laws.
