# Benchmarks

How much faster is EpiBranch.jl than the R packages it draws on? Speed matters when an analysis needs thousands of simulated outbreaks, for example to map the containment probability across values of R, k and tracing coverage, or to fit a model by repeated likelihood evaluation. This page compares timings for the same tasks in EpiBranch.jl and in R. The benchmark scripts are in [`benchmarks/`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/benchmarks), so you can rerun them on your own computer.

These timings are indicative only. Both the R and Julia implementations are under active development, so numbers will change over time.

## How to run

The Julia benchmarks run on the copy of EpiBranch in the repository folder and need three other packages. Start Julia in that folder and set up a project for them once:

```julia
using Pkg
Pkg.activate("benchmarks-env")  # create the project folder benchmarks-env
Pkg.develop(path = ".")  # use the EpiBranch in this folder
Pkg.add(["BenchmarkTools", "Distributions", "StableRNGs"])
```

then run, from a terminal in the repository folder:

```bash
julia --project=benchmarks-env benchmarks/benchmark_julia.jl
```

R benchmarks (requires [epichains](https://github.com/epiverse-trace/epichains) and [ringbp](https://github.com/epiforecasts/ringbp)):

```bash
Rscript benchmarks/benchmark_r.R
Rscript benchmarks/benchmark_r_ringbp.R
```

## Chain simulation (vs epichains)

Simulating 1000 transmission chains until each dies out, and calculating the likelihood of observed chain sizes, in EpiBranch.jl and in R's [epichains](https://github.com/epiverse-trace/epichains) package. Times are in milliseconds (ms, thousandths of a second) or microseconds (μs, millionths of a second).

| Scenario | R (epichains) | Julia (EpiBranch) |
|---|---|---|
| 1000 chains, Poisson(0.9) | 22.4 ms | 1.6 ms |
| 1000 chains, NegBin(0.8, 0.5) | 11.3 ms | 1.0 ms |
| 1000 chains + generation time | 30.2 ms | 2.0 ms |
| Chain statistics | 0.45 ms | 0.26 ms |
| Log-likelihood, Poisson offspring (Borel) | 151 μs | 0.18 μs |
| Log-likelihood, Poisson offspring with gamma-distributed R (gamma-Borel) | 165 μs | 0.49 μs |

The two likelihood rows give the probability of observed chain sizes. With Poisson offspring the chain size follows the Borel distribution. In the last row R varies from chain to chain, following a gamma distribution with shape k and scale R/k (mean R), and chain sizes then follow the gamma-Borel distribution (`ClusterMixed(Poisson, Gamma(k, R/k))` in Julia, `rgborel` in epichains).

## Intervention scenarios (vs ringbp)

Simulating 500 outbreaks with R = 2.5 and dispersion k = 0.16 (`NegBin(2.5, 0.16)`), each stopped at 5000 cases, in EpiBranch.jl and in R's [ringbp](https://github.com/epiforecasts/ringbp) package.

| Scenario | R (ringbp) | Julia (EpiBranch) |
|---|---|---|
| No interventions | 9,893 ms | 707 ms |
| 50% contact tracing | 10,374 ms | 707 ms |
| 50% tracing + quarantine | 9,319 ms | 707 ms |

!!! note "One Julia scenario against three R scenarios"
    The Julia column repeats one timing, for isolation plus 50% contact tracing (scenario 7 in `benchmark_julia.jl`), against all three ringbp scenarios. The models also differ: ringbp links each generation time to the case's incubation period and sets the fraction of transmission before symptoms, which the EpiBranch run does not. Read these as order-of-magnitude comparisons only.

## Other benchmarks

| Scenario | Julia (EpiBranch) |
|---|---|
| Line list generation (200 cases) | 0.008 ms |
| NegBin fit from 1000 offspring counts | 0.71 ms |

No direct R comparison is included for line list generation (simulist requires epiparameter database setup) or offspring fitting.

## Notes

- Julia timings exclude compilation. The first call to a function in a new Julia session takes longer, from a few seconds up to a minute, because Julia compiles the code first; later calls run at the speeds shown. The timings were measured after this first call with [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
- R timings use [microbenchmark](https://cran.r-project.org/package=microbenchmark)
- All timings are medians from multiple runs
- Hardware differences will affect absolute numbers; ratios are more informative
- Neither implementation is specifically optimised for speed
- Last run: April 2026
