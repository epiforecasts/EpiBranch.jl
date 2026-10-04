`MinimumSize(k)` is an observation model for chain sizes recorded only once a
cluster reaches `k` cases. `observe` turns a chain-size law into the
conditional `P(N = n | N >= k)` through `TruncatedChainSize`, and the
simulation path drops simulated clusters below `k`, so both evaluate against
the same distribution.
