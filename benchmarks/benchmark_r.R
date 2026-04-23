#!/usr/bin/env Rscript
# R benchmarks for comparison with EpiBranch.jl
# Uses epichains and simulist

library(epichains)
library(simulist)
library(microbenchmark)

set.seed(42)
cat("=== R Benchmarks ===\n\n")

# ── 1. Chain simulation (1000 chains, Poisson offspring) ─────────────

cat("1. Simulate 1000 chains (Poisson, R=0.9)\n")
res <- microbenchmark(
  simulate_chains(
    n_chains = 1000,
    statistic = "size",
    offspring_dist = rpois,
    lambda = 0.9
  ),
  times = 20
)
cat(sprintf("   Median: %.1f ms\n\n", median(res$time) / 1e6))

# ── 2. Chain simulation (NegBin offspring, overdispersed) ────────────

cat("2. Simulate 1000 chains (NegBin, R=0.8, k=0.5)\n")
res <- microbenchmark(
  simulate_chains(
    n_chains = 1000,
    statistic = "size",
    offspring_dist = rnbinom,
    mu = 0.8,
    size = 0.5
  ),
  times = 20
)
cat(sprintf("   Median: %.1f ms\n\n", median(res$time) / 1e6))

# ── 3. Chain simulation with generation time ─────────────────────────

cat("3. Simulate 1000 chains with generation time\n")
serial_interval <- function(x) rgamma(x, shape = 2, rate = 0.4)
res <- microbenchmark(
  simulate_chains(
    n_chains = 1000,
    statistic = "size",
    offspring_dist = rpois,
    lambda = 0.9,
    generation_time = serial_interval
  ),
  times = 20
)
cat(sprintf("   Median: %.1f ms\n\n", median(res$time) / 1e6))

# ── 4. Chain statistics (size and length) ────────────────────────────

cat("4. Chain statistics (from pre-simulated chains)\n")
chains <- simulate_chains(
  n_chains = 1000,
  statistic = "size",
  offspring_dist = rpois,
  lambda = 0.9
)
res <- microbenchmark(
  summary(chains),
  times = 50
)
cat(sprintf("   Median: %.3f ms\n\n", median(res$time) / 1e6))

# ── 5. Likelihood evaluation ─────────────────────────────────────────

cat("5. Chain size log-likelihood (analytical, Poisson)\n")
observed <- c(1L, 1L, 2L, 1L, 3L, 1L, 1L, 5L, 1L, 2L)
res <- microbenchmark(
  likelihood(
    chains = observed,
    statistic = "size",
    offspring_dist = rpois,
    lambda = 0.9
  ),
  times = 100
)
cat(sprintf("   Median: %.3f ms\n\n", median(res$time) / 1e6))

# ── 5b. Cluster-mixed likelihood (gamma-Borel, epichains rgborel) ────

cat("5b. Chain size log-likelihood (Poisson offspring, Gamma-mixed rate)\n")
# epichains rgborel: size = Gamma shape (k), mu = Gamma mean (R0).
res <- microbenchmark(
  likelihood(
    chains = observed,
    statistic = "size",
    offspring_dist = rgborel,
    size = 0.5,
    mu = 0.9
  ),
  times = 100
)
cat(sprintf("   Median: %.3f ms\n\n", median(res$time) / 1e6))

# ── 6. Line list simulation (simulist) ──────────────────────────────

cat("6. Line list simulation (simulist, ~200 cases)\n")
cat("   Skipped — simulist API requires epiparameter database setup\n\n")

cat("=== Done ===\n")
