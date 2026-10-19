# Benchmarks

How much faster is EpiBranch.jl than the R packages it draws on? Speed matters when an analysis needs thousands of simulated outbreaks, for example to map the containment probability across values of R, k and tracing coverage, or to fit a model by repeated likelihood evaluation. This page compares timings for the same tasks in EpiBranch.jl and in R. The benchmark scripts are in [`benchmarks/`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/benchmarks), so you can rerun them on your own computer.

These timings are indicative only. Both the R and Julia implementations are under active development, so numbers will change over time.

## How to run

The Julia benchmarks run on the copy of EpiBranch in the repository folder and need four other packages. Start Julia in that folder and set up a project for them once, outside the repository so that it leaves nothing behind in your checkout:

```julia
using Pkg
env = joinpath(homedir(), "epibranch-benchmarks")
Pkg.activate(env)
Pkg.develop(path = ".")  # use the EpiBranch in this folder, creating the project
Pkg.add(["BenchmarkTools", "Distributions", "Optim", "StableRNGs"])
```

then run, from a terminal in the repository folder:

```bash
julia --project=~/epibranch-benchmarks benchmarks/benchmark_julia.jl
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
| 1000 chains, Poisson(0.9) | 39.4 ms | 2.5 ms |
| 1000 chains, NegBin(0.8, 0.5) | 18.6 ms | 1.6 ms |
| 1000 chains + generation time | 50.8 ms | 3.3 ms |
| Chain statistics | 0.73 ms | 0.39 ms |
| Log-likelihood, Poisson offspring (Borel) | 296 μs | 0.37 μs |
| Log-likelihood, Poisson offspring with gamma-distributed R (gamma-Borel) | 305 μs | 1.06 μs |

The two likelihood rows give the probability of observed chain sizes. With Poisson offspring the chain size follows the Borel distribution. In the last row R varies from chain to chain, following a gamma distribution with shape k and scale R/k (mean R), and chain sizes then follow the gamma-Borel distribution (`ClusterMixed(Poisson, Gamma(k, R/k))` in Julia, `rgborel` in epichains).

## Intervention scenarios (vs ringbp)

Simulating 500 outbreaks with R = 2.5 and dispersion k = 0.16 (`NegBin(2.5, 0.16)`), each stopped at 5000 cases, in EpiBranch.jl and in R's [ringbp](https://github.com/epiforecasts/ringbp) package.

| Scenario | R (ringbp) | Julia (EpiBranch) |
|---|---|---|
| No interventions | 22,573 ms | 859 ms |
| 50% contact tracing | 25,842 ms | 859 ms |
| 50% tracing + quarantine | 21,091 ms | 859 ms |

!!! note "One Julia scenario against three R scenarios"
    The Julia column repeats one timing, for isolation plus 50% contact tracing (scenario 7 in `benchmark_julia.jl`), against all three ringbp scenarios. The models also differ: ringbp links each generation time to the case's incubation period and sets the fraction of transmission before symptoms, which the EpiBranch run does not. Read these as order-of-magnitude comparisons only.

## Other benchmarks

| Scenario | Julia (EpiBranch) |
|---|---|
| Line list generation (220 cases) | 0.06 ms |
| NegBin fit from 1000 offspring counts | 4.2 ms |

The script holds both rows to the size they name. The line list is built from a run stopped at a cap of 220 cases, and a branching process stops at exactly its cap. `condition = 220:220` rejects the runs that die out before reaching it, about one in five of them. A seed alone would be weaker: it can pick a run that dies out, and the run it produces changes with any change upstream of the first draw.

The fit is a maximum-likelihood search, EpiBranch's `loglikelihood` paired with an optimiser, because a negative binomial has no closed-form fit. The script cuts the counts the 1,000 simulated chains hold down to the 1,000 the row names, and starts the search away from the values the data were drawn from. Starting at them reaches the optimum in about two thirds of the iterations, which would understate what fitting costs.

No direct R comparison is included for line list generation (simulist requires epiparameter database setup) or offspring fitting.

## Notes

- Julia timings exclude compilation. The first call to a function in a new Julia session takes longer, from a few seconds up to a minute, because Julia compiles the code first; later calls run at the speeds shown. The timings were measured after this first call with [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
- R timings use [microbenchmark](https://cran.r-project.org/package=microbenchmark)
- All timings are medians from multiple runs
- Hardware differences will affect absolute numbers; ratios are more informative
- Neither implementation is specifically optimised for speed
- Last run: October 2026, every table on the same computer
