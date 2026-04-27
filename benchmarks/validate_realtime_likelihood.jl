#!/usr/bin/env julia
# Forward-simulation validation of `end_of_outbreak_probability`.
#
# Simulates many subcritical clusters, freezes each at a snapshot
# time, bins by τ = time-since-last-case at the snapshot, and compares
# the empirical fraction of clusters that go on to produce no more
# cases against the analytical formula.
#
# Expected outcome: agreement for τ greater than ~one generation
# interval; the formula overestimates extinction at small τ because
# the single-most-recent-case approximation ignores residual hazard
# from older cases in the same cluster.

using EpiBranch
using Distributions
using StableRNGs

const R_TRUTH = 0.6
const K_TRUTH = 0.3
const GT = Gamma(2.0, 2.5)            # mean 5 days
const SNAP = 30.0                     # snapshot time after seed
const N_SIMS = 50_000

const MODEL = BranchingProcess(NegBin(R_TRUTH, K_TRUTH), GT)

function run_validation(rng = StableRNG(42))
    τ_bins = 0.0:5.0:40.0
    n_bins = length(τ_bins) - 1
    n_quiet = zeros(Int, n_bins)
    n_extinct = zeros(Int, n_bins)
    cluster_sizes = [Int[] for _ in 1:n_bins]
    # For per-case validation: collect (per-case-ages, extinct?) pairs.
    per_case_data = Vector{Tuple{Vector{Float64}, Bool}}()

    for _ in 1:N_SIMS
        state = simulate(MODEL;
            sim_opts = SimOpts(max_cases = 200, max_time = SNAP + 200),
            rng = rng)
        times = [ind.infection_time
                 for ind in state.individuals
                 if ind.infection_time <= SNAP]
        isempty(times) && continue
        τ = SNAP - maximum(times)
        bin = searchsortedfirst(τ_bins, τ) - 1
        (1 <= bin < n_bins + 1) || continue
        bin > n_bins && continue
        any_future = any(ind.infection_time > SNAP for ind in state.individuals)
        n_quiet[bin] += 1
        n_extinct[bin] += !any_future
        push!(cluster_sizes[bin], length(times))
        push!(per_case_data, (SNAP .- times, !any_future))
    end

    println("Single-most-recent-case formula validation")
    println("Truth: R=$R_TRUTH, k=$K_TRUTH, Gamma(2, 2.5)")
    println()
    println("τ_bin       | n_quiet | mean size | empirical π | formula π | diff")
    println("―" ^ 75)
    for i in 1:n_bins
        n_quiet[i] < 50 && continue
        τ_mid = (τ_bins[i] + τ_bins[i + 1]) / 2
        emp = n_extinct[i] / n_quiet[i]
        formula = end_of_outbreak_probability(R_TRUTH, K_TRUTH, GT, Dirac(0.0);
            tau = τ_mid)
        size = round(sum(cluster_sizes[i]) / max(length(cluster_sizes[i]), 1),
            digits = 2)
        diff = round(formula - emp, digits = 3)
        println(rpad("[$(τ_bins[i]), $(τ_bins[i + 1]))", 12) * "| " *
                rpad(string(n_quiet[i]), 8) * "| " *
                rpad(string(size), 10) * "| " *
                rpad(string(round(emp, digits = 3)), 12) * "| " *
                rpad(string(round(formula, digits = 3)), 10) * "| " *
                string(diff))
    end

    println()
    println("Per-case formula validation")
    println("(Each case contributes exp(-R · S(τ_i)); cluster prob is the product.)")
    println()
    # Bin clusters by their predicted per-case π and check empirical
    # extinction within each bin.
    π_pred = [EpiBranch._per_case_extinction_probability(
                  R_TRUTH, K_TRUTH, GT, Dirac(0.0), ages)
              for (ages, _) in per_case_data]
    π_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    println("π predicted bin     | n     | empirical π | mean predicted | diff")
    println("―" ^ 70)
    for j in 1:(length(π_bins) - 1)
        idx = findall(p -> π_bins[j] <= p < π_bins[j + 1], π_pred)
        length(idx) < 50 && continue
        emp = sum(per_case_data[i][2] for i in idx) / length(idx)
        pred = sum(π_pred[i] for i in idx) / length(idx)
        diff = round(pred - emp, digits = 3)
        println(rpad("[$(π_bins[j]), $(π_bins[j + 1]))", 20) * "| " *
                rpad(string(length(idx)), 6) * "| " *
                rpad(string(round(emp, digits = 3)), 12) * "| " *
                rpad(string(round(pred, digits = 3)), 15) * "| " *
                string(diff))
    end
end

run_validation()
