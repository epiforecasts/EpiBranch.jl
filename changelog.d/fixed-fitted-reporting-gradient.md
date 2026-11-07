`ThinnedChainSize` preserves the numeric type of its reporting probability,
so fitting a `Reporting` probability with ForwardDiff uses the exact
chain-size likelihood and its gradient. Errors within exact chain-size laws
now propagate instead of silently selecting a simulated likelihood.
