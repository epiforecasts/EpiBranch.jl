`ThinnedChainSize` preserves the numeric type of its reporting probability.
Fitting a `Reporting` probability with ForwardDiff now uses the exact
chain-size likelihood and its gradient. Errors within exact chain-size laws
now propagate instead of silently falling back to a simulated likelihood.
