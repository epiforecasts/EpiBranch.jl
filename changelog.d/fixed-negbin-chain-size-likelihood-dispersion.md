The chain-size log-likelihood under `NegBin(R, k)` offspring could return
`NaN` or throw at extreme dispersion. `_gammaborel_logpdf` regroups the terms
fed to `logabsgamma` so they are bit identical at `x == s` and cancel to
exactly 0, instead of hitting a spurious Gamma-function pole when `k` is very
small. It also skips the `(x - s) * log(R / (k + R))` term at `x == s`
rather than evaluating it as `0 * log(0)` when `k` is very large. `GammaBorel` now
accepts `R == 0`, the degenerate no-transmission boundary that
`mean(NegativeBinomial)` rounds to once `k` swamps `R`, so
`chain_size_distribution` and the branching-process likelihood take the
analytical route there instead of falling back to simulation. Beyond
`k ≈ 1e16 · R`, where `R` cannot be recovered from the stored
`NegativeBinomial(k, p)`, the likelihood now returns the correct degenerate
answer (chain size fixed at the seed count) instead of `NaN` or an uncaught
error, while still tracking the Poisson limit smoothly up to that boundary.
