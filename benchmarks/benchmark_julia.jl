#!/usr/bin/env julia
# Julia benchmarks for EpiBranch.jl
# Equivalent scenarios to benchmark_r.R

using EpiBranch
using Dates
using Distributions
using BenchmarkTools
using StableRNGs

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

println("=== Julia Benchmarks ===\n")

# ── 1. Chain simulation (1000 chains, Poisson offspring) ─────────────

println("1. Simulate 1000 chains (Poisson, R=0.9)")
model_pois = BranchingProcess(Poisson(0.9))
b = @benchmark simulate_batch($model_pois, 1000, rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1e6, digits=1)) ms\n")

# ── 2. Chain simulation (NegBin offspring, overdispersed) ────────────

println("2. Simulate 1000 chains (NegBin, R=0.8, k=0.5)")
model_nb = BranchingProcess(NegBin(0.8, 0.5))
b = @benchmark simulate_batch($model_nb, 1000; rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1e6, digits=1)) ms\n")

# ── 3. Chain simulation with generation time ─────────────────────────

println("3. Simulate 1000 chains with generation time")
model_gt = BranchingProcess(Poisson(0.9), Exponential(5.0))
b = @benchmark simulate_batch($model_gt, 1000; rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1e6, digits=1)) ms\n")

# ── 4. Chain statistics ──────────────────────────────────────────────

println("4. Chain statistics (from pre-simulated chains)")
state = simulate(model_gt;
    sim_opts = SimOpts(n_initial = 1000, max_cases = 50_000), rng = StableRNG(42))
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
b = @benchmark chain_statistics($state)
println("   Median: $(round(median(b.times) / 1e6, digits=3)) ms\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 5. Likelihood evaluation ─────────────────────────────────────────

println("5. Chain size log-likelihood (analytical, Poisson)")
observed = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
b = @benchmark loglikelihood($observed, $(Poisson(0.9)))
println("   Median: $(round(median(b.times) / 1000, digits=2)) μs\n")

# ── 5b. Cluster-mixed likelihood (closed form via dispatch) ─────────

println("5b. Chain size log-likelihood (Poisson offspring, Gamma-mixed rate)")
# ClusterMixed(Poisson, Gamma) dispatches to the closed-form
# PoissonGammaChainSize, matching epichains' gborel likelihood.
cm = ClusterMixed(Poisson, Gamma(0.5, 0.9 / 0.5))
b = @benchmark loglikelihood($observed, $cm)
println("   Median: $(round(median(b.times) / 1000, digits=2)) μs\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 6. Line list simulation ─────────────────────────────────────────

println("6. Line list generation (from 200-case state)")
model_ll = BranchingProcess(Poisson(2.0), LogNormal(1.6, 0.5))
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
ll_state = simulate(model_ll;
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 200),
    rng = StableRNG(42))
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
b = @benchmark linelist($ll_state; reference_date = Date(2024, 1, 1), rng = StableRNG(99))
println("   Median: $(round(median(b.times) / 1e6, digits=3)) ms\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 7. Intervention scenario (batch, isolation + contact tracing) ────

println("7. Intervention scenario (500 sims, isolation + CT)")
model_int = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
iso = Isolation(delay = LogNormal(1.0, 0.5))
ct = ContactTracing(probability = 0.5, delay = Exponential(2.0))
b = @benchmark simulate_batch($model_int, 500;
    interventions = [$iso, $ct], attributes = $clinical,
    sim_opts = SimOpts(max_cases = 5000, max_generations = 50),
    rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1e6, digits=1)) ms\n")

# ── 8. Chain-size fitting ────────────────────────────────────────────

println("8. Fit NegBin offspring distribution from 1000 chain sizes")
rng = StableRNG(42)
fit_model = BranchingProcess(NegBin(0.7, 0.5))
fit_states = simulate_batch(fit_model, 1000; rng = rng)
fit_sizes = Int[]
for s in fit_states
    cs = chain_statistics(s)
    append!(fit_sizes, cs.size)
end
fit_data = ChainSizes(fit_sizes)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
b = @benchmark fit(NegativeBinomial, $fit_data)
println("   Median: $(round(median(b.times) / 1e6, digits=3)) ms\n")

println("=== Done ===")
