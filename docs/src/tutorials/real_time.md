# Real-time cluster-size inference

The chain-size likelihood assumes every cluster has finished
generating cases. Mid-outbreak data is a snapshot at some calendar
time, with some clusters still active. The likelihood needs an
adjustment.

EpiBranch uses a state-space framing: a *process model* (the latent
transmission dynamics — `BranchingProcess`) and an *observation
model* (how the latent state generates data — [`PerCaseObservation`](@ref)
covers per-case detection probability and reporting delay), plus an
optional cluster-level [`Snapshot`](@ref) carrying the per-cluster
observation timing. All three combine via [`Observed`](@ref) so the
same `loglikelihood(data, model)` dispatch handles every regime.

## What you observe per cluster

For each cluster you may have:

- the cluster size `x_i`,
- the time since the most recent case `τ_i`,
- or the times since each known case `(τ_{i,1}, …, τ_{i,x_i})`,
- or only a binary classification "ongoing / concluded" with no
  timing (the Endo-style threshold rule).

[`Snapshot`](@ref) encodes all of these in one type:

| Inner vector for cluster `i` | Meaning |
|---|---|
| `[Inf]` | concluded — no further reports possible |
| `[]` (empty) | ongoing, no timing — right-tail likelihood `P(X ≥ x)` |
| `[τ]` | single most-recent case observed at lag `τ` |
| `[τ_1, …, τ_x]` | every case observed, exact per-case product |

`Snapshot([3.0, 7.0, Inf])` builds a single-most-recent-case snapshot
of three clusters; the third is concluded. `Snapshot([[3.0], [],
[Inf]])` does the same with explicit per-cluster vectors and marks
the second cluster as ongoing-no-timing.

## The end-of-outbreak mixture

Let `π_i` be the probability that no further cases will be reported.
Each cluster contributes:

```
L_i = π_i · P(X = x_i | s_i, R, k)
    + (1 - π_i) · P(X ≥ x_i | s_i, R, k)
```

For a NegBin offspring distribution, the per-case "no more reports"
probability is `(1 + S(τ) · R/k)^(-k)`, where `S(τ) = P(G + D > τ)`
is the survival of the convolved generation-time + reporting-delay
distribution. Multi-case clusters use the product:

```
π_i = ∏_j (1 + S(τ_{i,j}) · R / k)^(-k)
```

`τ = ∞` collapses to π = 1 (chain-PMF only). Empty inner vector
collapses to π = 0 (right-tail only). Finite `τ` gives the
continuous mixture.

## Closed-outbreak case

When all clusters are concluded, no `Snapshot` is needed. The
two-argument `Observed` form (or just the bare offspring) works:

```julia
loglikelihood(ChainSizes(sizes; seeds), NegBin(R, k))
loglikelihood(
    ChainSizes(sizes; seeds),
    Observed(BranchingProcess(NegBin(R, k), gt), PerCaseObservation()))
```

## Mid-outbreak case

Pass per-cluster timing through `Snapshot`:

```julia
snap = Snapshot(taus)   # τ_i for each cluster
loglikelihood(
    ChainSizes(sizes; seeds),
    Observed(BranchingProcess(NegBin(R, k), gt), PerCaseObservation(),
        snap))
```

For a non-trivial reporting delay `D`, configure
`PerCaseObservation`:

```julia
loglikelihood(
    ChainSizes(sizes; seeds),
    Observed(BranchingProcess(NegBin(R, k), gt),
        PerCaseObservation(delay = LogNormal(1.6, 0.4)),
        snap))
```

For under-reporting, set `detection_prob < 1`. The chain-size PMF
becomes `ThinnedChainSize`, and the per-case rate in `π(τ)` becomes
`ρ·R` (the *direct-offspring approximation*). The approximation is
tight for `ρ` near 1; for low `ρ` it underestimates residual hazard
from unobserved descendants. The exact recursive form (Thompson,
Morgan & Jansen, *Phil Trans B*, 2019) requires a numerical
fixed-point solver and isn't implemented here.

## The threshold rule (Endo et al. 2020)

The 7-day threshold rule classifies each cluster as concluded
(silent ≥ 7 days) or ongoing (silent < 7 days), then uses the
chain-size PMF for concluded and the right-tail for ongoing. Encode
this directly as a Snapshot:

```julia
snap = Snapshot([τ >= 7.0 ? [Inf] : Float64[] for τ in taus])
loglikelihood(
    ChainSizes(sizes; seeds),
    Observed(BranchingProcess(NegBin(R, k), gt), PerCaseObservation(),
        snap))
```

Concluded clusters get `[Inf]` (π=1, chain-PMF). Ongoing clusters
get `[]` (no timing, right-tail). The likelihood reduces to Endo's
binary mixture.

The continuous-`τ` version is more rigorous (no ad-hoc cutoff,
acknowledges that even active clusters may have stopped). The
threshold rule is the right tool when timing data is unreliable, you
want robustness to GT distribution choice, or you want to match a
published threshold-rule analysis.

## Worked example

`benchmarks/reproduce_endo2020.jl` runs both on the WHO 27-Feb-2020
situation report data. Threshold (7-day rule) gives R=2.9, k=0.10 —
matching Endo's published joint MLE region. The single-`τ`
analytical mixture on the same dataset drifts upward in `k` because
most countries have several recent cases that the
most-recent-case-only π underweights.

## References

- Endo et al. (2020), *Estimating the overdispersion in COVID-19
  transmission using outbreak sizes outside China*, Wellcome Open
  Res 5:67.
- Nishiura, Miyamatsu & Mizumoto (2016), *Objective determination of
  end of MERS outbreak, South Korea, 2015*.
