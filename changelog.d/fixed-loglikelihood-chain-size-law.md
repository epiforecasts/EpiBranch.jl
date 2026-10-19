`loglikelihood(::ChainSizes, ...)` now accepts any chain-size law directly,
including a `ThinnedChainSize` such as `observe` returns for under-reporting.
Previously only `IndexChainSize` and `TruncatedChainSize` had their own
method; anything else fell through to the generic offspring-distribution
method and raised a `MethodError`. Every built-in chain-size law (`Borel`,
`GammaBorel`, `PoissonGammaChainSize`, `ChainSizeMixture`, `IndexChainSize`,
`TruncatedChainSize`, `ThinnedChainSize`) now shares the parent type
`ChainSizeLaw`, and a user-defined law gets the same `loglikelihood` method
by subtyping it.
