`ThinnedChainSize`'s `logpdf` now conditions on at least one case being
detected, matching the chains a `PerCaseObservation` actually records: a
chain with no detected case leaves no trace in the data, so its law over
observed sizes sums to 1 rather than to the probability of any detection.
The analytical chain-size likelihood under `PerCaseObservation` now agrees
with the simulation route, which already dropped undetected chains.
