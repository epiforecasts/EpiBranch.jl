`fit(data, Poisson)` and `fit(data, NegativeBinomial)` give a maximum-likelihood
`R` (and, for the Negative Binomial, dispersion `k`) from `OffspringCounts`,
`ChainSizes` or `ChainLengths`, with a profile-likelihood confidence interval
per parameter. A side the search cannot bound — typically the upper side of
`k`, where the likelihood flattens towards the Poisson limit — is reported as
`Inf`. An optional parametric bootstrap gives a percentile interval alongside
the profile one.
