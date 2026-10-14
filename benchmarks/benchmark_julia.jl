#!/usr/bin/env julia
# Julia benchmarks for EpiBranch.jl
# Equivalent scenarios to benchmark_r.R

using EpiBranch
using Dates
using Distributions
using BenchmarkTools
using StableRNGs
using Optim

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

println("=== Julia Benchmarks ===\n")

# ── 1. Chain simulation (1000 chains, Poisson offspring) ─────────────

println("1. Simulate 1000 chains (Poisson, R=0.9)")
model_pois = BranchingProcess(Poisson(0.9))
b = @benchmark simulate($model_pois, 1000, rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1.0e6, digits = 1)) ms\n")

# ── 2. Chain simulation (NegBin offspring, overdispersed) ────────────

println("2. Simulate 1000 chains (NegBin, R=0.8, k=0.5)")
model_nb = BranchingProcess(NegBin(0.8, 0.5))
b = @benchmark simulate($model_nb, 1000; rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1.0e6, digits = 1)) ms\n")

# ── 3. Chain simulation with generation time ─────────────────────────

println("3. Simulate 1000 chains with generation time")
model_gt = BranchingProcess(Poisson(0.9), Exponential(5.0))
b = @benchmark simulate($model_gt, 1000; rng = StableRNG(42))
println("   Median: $(round(median(b.times) / 1.0e6, digits = 1)) ms\n")

# ── 4. Chain statistics ──────────────────────────────────────────────

println("4. Chain statistics (from pre-simulated chains)")
state = simulate(
    model_gt;
    n_initial = 1000, max_cases = 50_000, rng = StableRNG(42)
)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
b = @benchmark chain_statistics($state)
println("   Median: $(round(median(b.times) / 1.0e6, digits = 3)) ms\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 5. Likelihood evaluation ─────────────────────────────────────────

println("5. Chain size log-likelihood (analytical, Poisson)")
observed = ChainSizes([1, 1, 2, 1, 3, 1, 1, 5, 1, 2])
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
b = @benchmark loglikelihood($observed, $(Poisson(0.9)))
println("   Median: $(round(median(b.times) / 1000, digits = 2)) μs\n")

# ── 5b. Cluster-mixed likelihood (closed form via dispatch) ─────────

println("5b. Chain size log-likelihood (Poisson offspring, Gamma-mixed rate)")
# ClusterMixed(Poisson, Gamma) dispatches to the closed-form
# PoissonGammaChainSize, matching epichains' gborel likelihood.
cm = ClusterMixed(Poisson, Gamma(0.5, 0.9 / 0.5))
b = @benchmark loglikelihood($observed, $cm)
println("   Median: $(round(median(b.times) / 1000, digits = 2)) μs\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 6. Line list simulation ─────────────────────────────────────────

println("6. Line list generation (from a 200 to 260 case state)")
model_ll = BranchingProcess(Poisson(2.0), LogNormal(1.6, 0.5))
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
# `condition` holds the state to a window, which is the most that is available
# here: the cap does not set the size, because the generation that crosses it
# completes, and a capped run of this process lands anywhere from 220 to about
# 490 cases, median around 310. So of the runs rejected, three in four are for
# overshooting 260 and the rest for dying out before 200, and nothing lands
# between 200 and 219 at all. A seed on its own would pin nothing: the run a
# given seed produces changes with any change upstream of the first draw, and
# the row would then name a size the state no longer has.
#
# The accepted state is a capped run rather than a finished outbreak, which is
# what a line list of a given size needs; a conditioned run meant to give a
# final size wants a cap above the range instead.
ll_state = simulate(
    ModelSpec(model_ll; attributes = clinical);
    max_cases = 220,
    condition = 200:260,
    rng = StableRNG(42)
)
println("   Cases in the state: $(ll_state.cumulative_cases)")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
b = @benchmark linelist($ll_state; reference_date = Date(2024, 1, 1))
println("   Median: $(round(median(b.times) / 1.0e6, digits = 3)) ms\n")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# ── 7. Intervention scenario (batch, isolation + contact tracing) ────

println("7. Intervention scenario (500 sims, isolation + CT)")
model_int = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
# benchmark_r.R never releases isolation, so this matches with Inf.
iso = Isolation(onset_to_isolation_delay = LogNormal(1.0, 0.5), duration = Inf)
ct = ContactTracing(
    probability = 0.5, isolation_to_trace_delay = Exponential(2.0),
    action = Quarantine(duration = Inf)
)
spec_int = ModelSpec(model_int; interventions = [iso, ct], attributes = clinical)
b = @benchmark simulate(
    $spec_int, 500;
    max_cases = 5000, max_generations = 50,
    rng = StableRNG(42)
)
println("   Median: $(round(median(b.times) / 1.0e6, digits = 1)) ms\n")

# ── 8. Chain-size fitting ────────────────────────────────────────────

println("8. Fit NegBin offspring distribution from 1000 offspring counts")
rng = StableRNG(42)
fit_model = BranchingProcess(NegBin(0.7, 0.5))
fit_states = simulate(fit_model, 1000; rng = rng)
fit_counts = Int[]
for s in fit_states
    tally = Dict{Int, Int}()
    for ind in s.individuals
        is_infected(ind) || continue
        tally[ind.id] = get(tally, ind.id, 0)
        ind.parent_id == 0 && continue
        tally[ind.parent_id] = get(tally, ind.parent_id, 0) + 1
    end
    append!(fit_counts, values(tally))
end
# A negative binomial has no closed-form maximum-likelihood fit, so this is
# the route the documentation gives: EpiBranch's `loglikelihood` paired with an
# optimiser. The counts are cut to 1,000 so that the timing is for a dataset of
# a stated size, comparable with an R implementation on the same number of
# counts; 1,000 chains of mean size about 3.3 give more than that.
fit_data = OffspringCounts(fit_counts[1:1000])
nll(theta) = -loglikelihood(fit_data, NegBin(exp(theta[1]), exp(theta[2])))
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
# Started away from the data-generating values, because starting at them finds
# the optimum in about two thirds of the iterations and would understate what
# the fit costs.
b = @benchmark optimize($nll, [log(1.0), log(1.0)])
println("   Median: $(round(median(b.times) / 1.0e6, digits = 3)) ms\n")

println("=== Done ===")
