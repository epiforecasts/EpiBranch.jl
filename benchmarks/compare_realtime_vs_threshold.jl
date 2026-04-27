#!/usr/bin/env julia
# Parameter-recovery comparison: threshold rule vs continuous real-time
# likelihood, on simulated cluster-size data with mid-outbreak snapshots.
#
# Setup:
#   - Generate clusters from a known (R, k) under a Gamma generation time.
#   - Snapshot each at a fixed time so a fraction is still ongoing.
#   - For each cluster, record observed size, time-since-last-case τ.
#   - Estimate (R, k) by MLE over a grid for each method:
#       (a) threshold rule with several window choices,
#       (b) real-time end-of-outbreak likelihood with the true GT.
#
# Reports: MLE point estimate and log-likelihood at the truth, for each
# method. The real-time method is expected to recover the truth more
# closely when the snapshot time leaves many clusters mid-outbreak.

using EpiBranch
using Distributions
using StableRNGs

const R_TRUTH = 0.6
const K_TRUTH = 0.3
const GT = Gamma(2.0, 2.5)             # mean 5 days
const SNAP = 15.0                      # data cutoff (~3 generations)
const N_CLUSTERS = 1000

const R_GRID = 0.2:0.025:1.2
const K_GRID = 0.05:0.025:1.0

function simulate_dataset(rng)
    model = BranchingProcess(NegBin(R_TRUTH, K_TRUTH), GT)
    sizes = Int[]
    last_times = Float64[]
    case_ages = Vector{Vector{Float64}}()
    while length(sizes) < N_CLUSTERS
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 200, max_time = SNAP + 200),
            rng = rng)
        times = [ind.infection_time
                 for ind in state.individuals
                 if ind.infection_time <= SNAP]
        isempty(times) && continue
        push!(sizes, length(times))
        push!(last_times, maximum(times))
        push!(case_ages, SNAP .- times)
    end
    τ = SNAP .- last_times
    return sizes, τ, case_ages
end

function threshold_loglik(sizes, τ, R, k; window)
    concluded = τ .>= window
    data = ChainSizes(sizes; concluded = concluded)
    return loglikelihood(data, NegBin(R, k))
end

function realtime_loglik(sizes, τ, R, k)
    data = RealTimeChainSizes(sizes, τ)
    model = BranchingProcess(NegBin(R, k), GT)
    return loglikelihood(data, model)
end

function realtime_percase_loglik(case_ages, R, k)
    data = RealTimeChainSizes(case_ages)
    model = BranchingProcess(NegBin(R, k), GT)
    return loglikelihood(data, model)
end

function grid_mle(loglik_fn)
    best_R, best_k, best_ll = R_GRID[1], K_GRID[1], -Inf
    for R in R_GRID, k in K_GRID

        ll = loglik_fn(R, k)
        if ll > best_ll
            best_R, best_k, best_ll = R, k, ll
        end
    end
    return (R = best_R, k = best_k, ll = best_ll)
end

function main()
    rng = StableRNG(42)
    sizes, τ, case_ages = simulate_dataset(rng)
    n_concluded_3 = count(>=(3.0), τ)
    n_concluded_7 = count(>=(7.0), τ)
    n_concluded_14 = count(>=(14.0), τ)
    println("Truth: R=$R_TRUTH, k=$K_TRUTH; SNAP=$SNAP; N=$N_CLUSTERS")
    println("Concluded under 3/7/14 day rule: " *
            "$n_concluded_3 / $n_concluded_7 / $n_concluded_14")
    println()

    methods = Dict(
        "Threshold τ≥3" => (R, k) -> threshold_loglik(sizes, τ, R, k; window = 3.0),
        "Threshold τ≥7" => (R, k) -> threshold_loglik(sizes, τ, R, k; window = 7.0),
        "Threshold τ≥14" => (R, k) -> threshold_loglik(sizes, τ, R, k; window = 14.0),
        "RT (last-case)" => (R, k) -> realtime_loglik(sizes, τ, R, k),
        "RT (per-case)" => (R, k) -> realtime_percase_loglik(case_ages, R, k))

    println("Method            | MLE R   MLE k   LL@truth   LL@MLE")
    println("―" ^ 60)
    for name in [
        "Threshold τ≥3", "Threshold τ≥7", "Threshold τ≥14",
        "RT (last-case)", "RT (per-case)"]
        fn = methods[name]
        mle = grid_mle(fn)
        ll_truth = fn(R_TRUTH, K_TRUTH)
        println(rpad(name, 18) * "| " *
                rpad(string(round(mle.R, digits = 2)), 8) *
                rpad(string(round(mle.k, digits = 2)), 8) *
                rpad(string(round(ll_truth, digits = 1)), 11) *
                string(round(mle.ll, digits = 1)))
    end
end

main()
