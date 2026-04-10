#!/usr/bin/env julia
#
# ringbp clone — reproducing Hellewell et al. 2020 with EpiBranch.jl
#
# This replicates the core ringbp workflow:
# 1. Define scenarios (R0, k, tracing probability, delays)
# 2. Run batch simulations per scenario
# 3. Compute containment probability
# 4. Sweep over parameters and produce a results table

using EpiBranch
using Distributions
using DataFrames
using StableRNGs

# ── Scenario parameters (matching ringbp vignette) ───────────────────

# Offspring: NegBin(R0, k) with R0 ∈ {1.5, 2.5} and k = 0.16
# Isolation: Weibull delay from symptom onset
# Contact tracing: probability ∈ {0, 0.5, 1.0}
# Initial cases: 5, 20

# Two delay profiles from Hellewell et al.
# "SARS-like": Weibull(shape=1.65, scale=4.29)
# "Wuhan-like": Weibull(shape=2.31, scale=9.48)
sars_delay = Weibull(1.65, 4.29)
wuhan_delay = Weibull(2.31, 9.48)

# Incubation period (Backer et al. 2020)
incubation = LogNormal(1.57, 0.65)

# Generation time: ringbp-style incubation-linked model
# 15% presymptomatic transmission
gt = ringbp_generation_time(presymptomatic_fraction = 0.15)

# Clinical attributes: 10% asymptomatic
clinical = Disease(
    incubation_period = incubation,
    prob_asymptomatic = 0.1
)

println("=== ringbp clone: Hellewell et al. 2020 scenarios ===\n")

# ── Run scenarios ────────────────────────────────────────────────────

results = DataFrame(
    delay_group = String[],
    R0 = Float64[],
    k = Float64[],
    tracing_prob = Float64[],
    initial_cases = Int[],
    containment_prob = Float64[]
)

n_sim = 500
rng = StableRNG(2020)

for (delay_name, delay_dist) in [("SARS-like", sars_delay), ("Wuhan-like", wuhan_delay)]
    for R0 in [1.5, 2.5]
        for tracing_prob in [0.0, 0.5, 1.0]
            for initial_cases in [5, 20]
                k = 0.16
                model = BranchingProcess(NegBin(R0, k), gt)
                iso = Isolation(delay = delay_dist, test_sensitivity = 1.0)

                interventions = if tracing_prob > 0
                    ct = ContactTracing(
                        probability = tracing_prob,
                        delay = Exponential(1.0),
                        quarantine_on_trace = false
                    )
                    [iso, ct]
                else
                    [iso]
                end

                batch = simulate_batch(model, n_sim;
                    interventions = interventions,
                    attributes = clinical,
                    sim_opts = SimOpts(
                        max_cases = 5000,
                        max_time = 350.0,
                        n_initial = initial_cases
                    ),
                    rng = rng
                )

                cp = containment_probability(batch)

                push!(results,
                    (
                        delay_group = delay_name,
                        R0 = R0,
                        k = k,
                        tracing_prob = tracing_prob,
                        initial_cases = initial_cases,
                        containment_prob = round(cp, digits = 3)
                    ))
            end
        end
    end
end

println(results)
println()

# ── Key findings ─────────────────────────────────────────────────────

println("Key findings:")
println()

# Filter for main comparison: 5 initial cases, SARS-like delays
main = filter(r -> r.initial_cases == 5 && r.delay_group == "SARS-like", results)
println("SARS-like delays, 5 initial cases:")
println(main[:, [:R0, :tracing_prob, :containment_prob]])
println()

# Compare delay groups at R0=2.5, 100% tracing
compare_delays = filter(r -> r.R0 == 2.5 && r.tracing_prob == 1.0 && r.initial_cases == 5, results)
println("R0=2.5, 100% tracing, 5 initial cases:")
println(compare_delays[:, [:delay_group, :containment_prob]])
println()

# ── Analytical comparison ────────────────────────────────────────────

println("Analytical extinction probabilities (no interventions):")
for R0 in [1.5, 2.5]
    q = extinction_probability(R0, 0.16)
    println("  R0=$R0, k=0.16: P(extinct|1 case) = $(round(q, digits=3))")
    println("                   P(extinct|5 cases) = $(round(q^5, digits=3))")
end

println("\n=== Done ===")
