#!/usr/bin/env Rscript
# R benchmarks for ringbp comparison with EpiBranch.jl
# Requires: remotes::install_github("epiforecasts/ringbp")

library(ringbp)
library(microbenchmark)

set.seed(42)
cat("=== R ringbp Benchmarks ===\n\n")

offspring <- offspring_opts(
  community = function(n) rnbinom(n, mu = 2.5, size = 0.16),
  isolated = function(n) rep(0L, n)
)
delays <- delay_opts(
  incubation_period = function(n) rlnorm(n, meanlog = 1.6, sdlog = 0.5),
  onset_to_isolation = function(n) rweibull(n, shape = 1.651524, scale = 4.287786)
)
sim <- sim_opts(cap_cases = 5000, cap_max_days = 350)

# ── 1. No interventions ───────────────────────────────────────────

cat("1. 500 sims, NegBin(2.5, 0.16), no interventions\n")
res <- microbenchmark(
  scenario_sim(
    n = 500, initial_cases = 1,
    offspring = offspring, delays = delays,
    event_probs = event_prob_opts(asymptomatic = 0, presymptomatic_transmission = 0.15, symptomatic_traced = 0),
    interventions = intervention_opts(quarantine = FALSE),
    sim = sim
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

# ── 2. 50% contact tracing ──────────────────────────────────────

cat("2. 500 sims, NegBin(2.5, 0.16), 50% contact tracing\n")
res <- microbenchmark(
  scenario_sim(
    n = 500, initial_cases = 1,
    offspring = offspring, delays = delays,
    event_probs = event_prob_opts(asymptomatic = 0, presymptomatic_transmission = 0.15, symptomatic_traced = 0.5),
    interventions = intervention_opts(quarantine = FALSE),
    sim = sim
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

# ── 3. 50% contact tracing + quarantine ─────────────────────────

cat("3. 500 sims, NegBin(2.5, 0.16), 50% tracing + quarantine\n")
res <- microbenchmark(
  scenario_sim(
    n = 500, initial_cases = 1,
    offspring = offspring, delays = delays,
    event_probs = event_prob_opts(asymptomatic = 0, presymptomatic_transmission = 0.15, symptomatic_traced = 0.5),
    interventions = intervention_opts(quarantine = TRUE),
    sim = sim
  ),
  times = 5
)
cat(sprintf("   Median: %.0f ms\n\n", median(res$time) / 1e6))

cat("=== Done ===\n")
