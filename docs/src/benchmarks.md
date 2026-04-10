# Benchmarks

This page documents performance comparisons between EpiBranch.jl and equivalent R packages. The benchmark scripts are in [`benchmarks/`](https://github.com/epiforecasts/EpiBranch.jl/tree/main/benchmarks) and can be run locally to reproduce results on your own hardware.

These are indicative timings, not rigorous benchmarks. Both the R and Julia implementations are under active development, so numbers will change over time.

## How to run

Julia benchmarks:

```bash
julia --project benchmarks/benchmark_julia.jl
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
| 1000 chains, Poisson(0.9) | 23.9 ms | 2.0 ms |
| 1000 chains, NegBin(0.8, 0.5) | 11.8 ms | 1.2 ms |
| 1000 chains + generation time | 31.6 ms | 2.6 ms |
| Chain statistics | 0.48 ms | 0.27 ms |
| Analytical log-likelihood | 0.16 ms | 0.008 ms |

## Intervention scenarios (vs ringbp)

Simulating 500 outbreaks with interventions (isolation, contact tracing), comparing EpiBranch.jl with R's [ringbp](https://github.com/epiforecasts/ringbp) package. Both use NegBin(2.5, 0.16) offspring with a cap of 5000 cases.

*Results pending — run `benchmarks/benchmark_r_ringbp.R` and `benchmarks/benchmark_julia.jl` (scenario 7) to generate.*

## Notes

- Julia timings exclude compilation (measured after warm-up using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl))
- R timings use [microbenchmark](https://cran.r-project.org/package=microbenchmark)
- All timings are medians from multiple runs
- Hardware differences will affect absolute numbers; ratios are more informative
- Neither implementation is specifically optimised for speed
