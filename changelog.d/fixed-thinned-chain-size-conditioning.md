`ThinnedChainSize`'s `logpdf` now conditions on at least one case being
detected, matching the chains a simulation actually records. A chain with no
detected case leaves no trace in the data. The law over observed sizes therefore
sums to 1. The analytical chain-size likelihood under a `Reporting` step now
agrees with the simulation route, which already dropped undetected chains.
