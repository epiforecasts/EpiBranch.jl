The analytical `loglikelihood(::ChainLengths, model)` now honours
`n_initial`: a cluster with several independent seeds has the length of the
longest of them, so its CDF is the single-seed CDF raised to the power
`n_initial`, rather than every cluster being scored as if it had one seed.
