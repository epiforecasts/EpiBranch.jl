#!/usr/bin/env Rscript
# R benchmarks for ringbp comparison with EpiBranch.jl
# Requires the ringbp package: remotes::install_github("epiforecasts/ringbp")

library(ringbp)
library(microbenchmark)

set.seed(42)
cat("=== R ringbp Benchmarks ===\n\n")

# ── 1. Basic containment scenario (no interventions) ───────────────

cat("1. 500 sims, NegBin(2.5, 0.16), no interventions\n")
res <- microbenchmark(
  scenario_sim(
    n.sim = 500,
    num.initial.cases = 1,
    prop.asym = 0,
    prop.ascertain = 0,
    cap_cases = 5000,
    cap_max_days = 350,
    r0isolated = 0,
    r0community = 2.5,
    disp.com = 0.16,
    disp.iso = 1,
    delay_shape = 1.651524,
    delay_scale = 4.287786,
    k = 0,
    quaression = 0
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

# ── 2. Isolation only ──────────────────────────────────────────────

cat("2. 500 sims, NegBin(2.5, 0.16), isolation\n")
res <- microbenchmark(
  scenario_sim(
    n.sim = 500,
    num.initial.cases = 1,
    prop.asym = 0,
    prop.ascertain = 1.0,
    cap_cases = 5000,
    cap_max_days = 350,
    r0isolated = 0,
    r0community = 2.5,
    disp.com = 0.16,
    disp.iso = 1,
    delay_shape = 1.651524,
    delay_scale = 4.287786,
    k = 0,
    quaression = 0
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

# ── 3. Isolation + contact tracing ────────────────────────────────

cat("3. 500 sims, NegBin(2.5, 0.16), isolation + contact tracing\n")
res <- microbenchmark(
  scenario_sim(
    n.sim = 500,
    num.initial.cases = 1,
    prop.asym = 0,
    prop.ascertain = 1.0,
    cap_cases = 5000,
    cap_max_days = 350,
    r0isolated = 0,
    r0community = 2.5,
    disp.com = 0.16,
    disp.iso = 1,
    delay_shape = 1.651524,
    delay_scale = 4.287786,
    k = 0.5,
    quaression = 0
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

cat("=== Done ===\n")
