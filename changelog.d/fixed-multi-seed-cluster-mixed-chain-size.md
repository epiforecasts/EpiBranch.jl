Multi-seed chain-size likelihoods for `ClusterMixed` now sum independent
single-seed chains, each with its own mixed parameter, matching simulation.
This corrects `PoissonGammaChainSize` and adds multi-seed support for
`ChainSizeMixture`, including its use within `IndexChainSize`.
