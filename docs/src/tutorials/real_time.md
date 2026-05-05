# Real-time cluster-size inference

The chain-size likelihood assumes every cluster has finished
generating cases. Mid-outbreak data is a snapshot at some calendar
time, with some clusters still active. The likelihood needs an
adjustment.

EpiBranch uses a state-space framing: a *process model* (the latent
transmission dynamics — `BranchingProcess`) and an *observation
model* (how the latent state generates data — [`PerCaseObservation`](@ref)
covers per-case detection probability and reporting delay). The two
combine via [`Surveilled`](@ref) so the same `loglikelihood(data,
model)` dispatch handles every combination.

## The observation model

Cluster `i` was seeded by `s_i` imported cases at some time before
the snapshot. At the snapshot the observer has two numbers per
cluster:

- `x_i`: cases reported so far,
- `τ_i`: time since the most recent case.

A long `τ_i` means the cluster has likely run its course; a short
one means more cases are probably coming.

The data type is [`RealTimeChainSizes`](@ref):

```julia
data = RealTimeChainSizes(sizes, taus; seeds = seeds)
```

## The end-of-outbreak mixture

Let `π_i` denote the probability that no further cases will be
reported, given everything observed so far. Each cluster
contributes:

```
L_i = π_i · P(X = x_i | s_i, R, k)
    + (1 - π_i) · P(X ≥ x_i | s_i, R, k)
```

The chain-size PMF `P(X | s)` comes from
`chain_size_distribution(NegBin(R, k))`. For a NegBin offspring
distribution, marginalising the per-case Poisson rate over its Gamma
mixing prior gives:

```
π(τ) = (1 + S(τ) · R / k)^(-k)
```

where `S(τ) = P(G + D > τ)` is the survival function of the
convolved generation-time and reporting-delay distributions. As
`k → ∞` (Poisson limit) this reduces to the form Nishiura (2016)
used for end-of-outbreak declaration: `π(τ) = exp(-R · S(τ))`.

The single-`τ` form uses only the most recent case. With per-case
times available, the per-case product is exact:

```
π = ∏_i (1 + S(τ_i) · R / k)^(-k)
```

Pass `case_ages = [...]` to [`RealTimeChainSizes`](@ref) and the
likelihood will use the exact form.

```julia
model = BranchingProcess(NegBin(R, k), generation_time)
loglikelihood(data, model)
```

For a non-trivial reporting delay `D`, combine the process with a
`PerCaseObservation` via `Surveilled`:

```julia
obs = PerCaseObservation(detection_prob = 1.0, delay = LogNormal(1.6, 0.4))
loglikelihood(data, Surveilled(model, obs))
```

The convenience constructor [`Reported`](@ref) wraps this:
`Reported(model, delay)` is exactly `Surveilled(model,
PerCaseObservation(1.0, delay))`. The bare-model likelihood is the
special case `D = Dirac(0.0)`.

For under-reporting, set `detection_prob < 1`. The likelihood then
uses the *direct-offspring approximation* — the per-case report rate
is `ρ·R` in `π(τ)`, and the chain-size PMF in the mixture becomes
`ThinnedChainSize` against the underlying chain-size distribution:

```julia
obs = PerCaseObservation(detection_prob = 0.7, delay = LogNormal(1.6, 0.4))
loglikelihood(data, Surveilled(model, obs))
```

The approximation ignores hazard from unobserved descendants. It's
tight for `ρ` near 1 and underestimates residual hazard at low `ρ`.
The exact recursive form (Thompson, Morgan & Jansen, *Phil Trans B*
2019) requires a numerical fixed-point and isn't implemented here.

The single-`τ` approximation underestimates extinction when older
silent cases haven't wound down. A cluster that is quiet now but had
several cases recently is treated more optimistically than the
per-case form would treat it. The per-case product fixes this when
the data permits.

## Comparison with the threshold rule

The threshold rule used in Endo et al. (2020) is the degenerate case
where `π_i ∈ {0, 1}` from a hard cutoff (e.g. `concluded_i = τ_i ≥ 7`
days). It's expressed in EpiBranch via the `concluded` flag on
[`ChainSizes`](@ref):

```julia
ChainSizes(sizes; seeds = seeds, concluded = taus .>= 7.0)
```

The continuous π form is the natural generalisation. The threshold
rule is the right tool when the classification comes from outside
the model — official outbreak-end declarations, contact tracing
exhaustion, lab confirmation of the last case. The continuous form
is the right tool when you only have the case timing and want the
likelihood to handle the uncertainty in "is it really over".

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
