The pairwise survival likelihoods, including the routed network likelihood,
compute the cumulative hazard and log hazard of `Exponential`, `Weibull`,
`Gamma` and `LogNormal` kernels from `logccdf` and `logpdf` instead of through
`log(ccdf)`. This is faster, and the cumulative hazard stays finite in the far
tail, where `ccdf` underflows to zero and the generic form gave `Inf`. Other
kernels still use `cumhazard` and `loghazard`.
