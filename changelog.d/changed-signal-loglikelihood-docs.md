The `loglikelihood(::SignalData, model)` docstrings now say that the result is
a one-sample Monte Carlo estimate whose log is biased downwards, that it suits
pseudo-marginal MCMC rather than optimisers or gradient-based samplers, and
that longer series call for a particle method.
