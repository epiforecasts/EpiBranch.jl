# Benchmarks

This page documents performance comparisons between EpiBranch.jl and equivalent R packages. The benchmark scripts are in [`benchmarks/`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/benchmarks) and can be run locally to reproduce results on your own hardware.

These are indicative timings, not rigorous benchmarks. Both the R and Julia implementations are under active development, so numbers will change over time.

## How to run

Julia benchmarks (requires BenchmarkTools and StableRNGs in your default environment):

```bash
julia benchmarks/benchmark_julia.jl
```

R benchmarks (requires [epichains](https://github.com/epiverse-trace/epichains) and [ringbp](https://github.com/epiforecasts/ringbp)):

```bash
Rscript benchmarks/benchmark_r.R
Rscript benchmarks/benchmark_r_ringbp.R
```

## Chain simulation (vs epichains)

Simulating 1000 transmission chains to completion, comparing EpiBranch.jl with R's [epichains](https://github.com/epiverse-trace/epichains) package.

| Scenario | R (epichains) | Julia (EpiBranch) |
|---|---|---|
| 1000 chains, Poisson(0.9) | 22.4 ms | 1.6 ms |
| 1000 chains, NegBin(0.8, 0.5) | 11.3 ms | 1.0 ms |
| 1000 chains + generation time | 30.2 ms | 2.0 ms |
| Chain statistics | 0.45 ms | 0.26 ms |
| Log-likelihood (Poisson → Borel) | 151 μs | 0.18 μs |
| Log-likelihood (Poisson + Gamma mixing → gamma-Borel) | 165 μs | 0.49 μs |

The last row exercises the cluster-level mixing closed form (`ClusterMixed(Poisson, Gamma(k, R/k))` in Julia, `rgborel` in epichains).

## Intervention scenarios (vs ringbp)

Simulating 500 outbreaks with NegBin(2.5, 0.16) offspring, isolation, and contact tracing, capped at 5000 cases. Comparing EpiBranch.jl with R's [ringbp](https://github.com/epiforecasts/ringbp) package.

| Scenario | R (ringbp) | Julia (EpiBranch) |
|---|---|---|
| No interventions | 9,893 ms | 707 ms |
| 50% contact tracing | 10,374 ms | 707 ms |
| 50% tracing + quarantine | 9,319 ms | 707 ms |

The Julia column uses the same scenario (isolation + 50% contact tracing, scenario 7 in `benchmark_julia.jl`). The ringbp scenarios differ slightly in parameterisation (incubation-linked generation time, presymptomatic transmission fraction), so these are order-of-magnitude comparisons rather than like-for-like.

## Other benchmarks

| Scenario | Julia (EpiBranch) |
|---|---|
| Line list generation (200 cases) | 0.008 ms |
| NegBin fit from 1000 offspring counts | 0.71 ms |

No direct R comparison is included for line list generation (simulist requires epiparameter database setup) or offspring fitting.

## Notes

- Julia timings exclude compilation (measured after warm-up using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl))
- R timings use [microbenchmark](https://cran.r-project.org/package=microbenchmark)
- All timings are medians from multiple runs
- Hardware differences will affect absolute numbers; ratios are more informative
- Neither implementation is specifically optimised for speed
- Last run: April 2026 (compositional likelihood rows added)
