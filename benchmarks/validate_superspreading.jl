#!/usr/bin/env julia
# Validate EpiBranch.jl against R superspreading reference values

using EpiBranch
using Distributions
using Test

println("=== EpiBranch.jl vs R superspreading validation ===\n")

# 1. probability_extinct (R=1.5, k=0.1, n_initial=10)
# R: 0.4963112
println("1. extinction_probability(1.5, 0.1)^10")
q = extinction_probability(1.5, 0.1)
p = q^10
println("   Julia: $(round(p, digits=7)), R: 0.4963112")
@test p ≈ 0.4963112 atol=1e-5

# 2. probability_epidemic (R=1.5, k=0.1, n_initial=10)
# R: 0.5036888
println("2. 1 - extinction^10")
pe = 1 - p
println("   Julia: $(round(pe, digits=7)), R: 0.5036888")
@test pe ≈ 0.5036888 atol=1e-5

# 3. probability_contain (R=1.5, k=0.5, pop_control=0.1)
# R: 0.8213172
println("3. probability_contain(1.5, 0.5; pop_control=0.1)")
pc = probability_contain(1.5, 0.5; pop_control=0.1)
println("   Julia: $(round(pc, digits=7)), R: 0.8213172")
@test pc ≈ 0.8213172 atol=1e-5

# 4. probability_contain (R=1.5, k=0.5, ind_control=0.1)
# R: 0.8391855
println("4. probability_contain(1.5, 0.5; ind_control=0.1)")
pc = probability_contain(1.5, 0.5; ind_control=0.1)
println("   Julia: $(round(pc, digits=7)), R: 0.8391855")
@test pc ≈ 0.8391855 atol=1e-5

# 5. probability_contain (both controls)
# R: 0.8915076
println("5. probability_contain(1.5, 0.5; ind_control=0.1, pop_control=0.1)")
pc = probability_contain(1.5, 0.5; ind_control=0.1, pop_control=0.1)
println("   Julia: $(round(pc, digits=7)), R: 0.8915076")
@test pc ≈ 0.8915076 atol=1e-5

# 6. probability_contain (5 introductions)
# R: 0.3737271
println("6. probability_contain(1.5, 0.5; n_initial=5, pop_control=0.1)")
pc = probability_contain(1.5, 0.5; n_initial=5, pop_control=0.1)
println("   Julia: $(round(pc, digits=7)), R: 0.3737271")
@test pc ≈ 0.3737271 atol=1e-5

# 7. proportion_cluster_size (R=2, k=0.1, cluster_size=10)
println("7. proportion_cluster_size(2.0, 0.1; cluster_size=10)")
pcs = proportion_cluster_size(2.0, 0.1; cluster_size=10)
println("   Julia: $(round(pcs, digits=4))")
@test 0.0 < pcs < 1.0

# 8. proportion_transmission (R=2, k=0.5, prop_cases=0.2) — "top 20% cause X%"
println("8. proportion_transmission(2.0, 0.5; prop_cases=0.2)")
pt = proportion_transmission(2.0, 0.5; prop_cases=0.2)
println("   Julia: $(round(pt, digits=4))")
@test 0.0 < pt < 1.0

# 9. network_R (NATSAL data)
# R: R=0.2431034, R_net=6.166508
println("9. network_R(14.1, 69.6, 1.0, 1.0)")
# R divides by age range (74-16=58), we pass pre-scaled values
mean_c = 14.1 / (74 - 16)
sd_c = 69.6 / (74 - 16)
res = network_R(mean_c, sd_c, 1.0, 1.0)
println("   Julia: R=$(round(res.R, digits=4)), R_net=$(round(res.R_net, digits=4))")
println("   R ref: R=0.2431, R_net=6.1665")
@test res.R ≈ 0.2431034 atol=1e-3
@test res.R_net ≈ 6.166508 atol=1e-2

# 10. probability_epidemic varying k (R=1.5)
# R: k=1: 0.3333, k=0.5: 0.2324, k=0.1: 0.0677
println("10. epidemic_probability varying k (R=1.5)")
r_vals = Dict(1.0 => 0.3333333, 0.5 => 0.2324081, 0.1 => 0.06765766)
for (k, r_ref) in sort(collect(r_vals), by=first)
    jl = epidemic_probability(1.5, k)
    println("    k=$k: Julia=$(round(jl, digits=7)), R=$r_ref")
    @test jl ≈ r_ref atol=1e-5
end

# 11. probability_epidemic varying R (k=1)
# R: 0.5→0, 1→0, 1.5→0.3333, 5→0.8
println("11. epidemic_probability varying R (k=1)")
r_vals = Dict(0.5 => 0.0, 1.0 => 0.0, 1.5 => 0.3333333, 5.0 => 0.8)
for (R, r_ref) in sort(collect(r_vals), by=first)
    jl = epidemic_probability(R, 1.0)
    println("    R=$R: Julia=$(round(jl, digits=7)), R=$r_ref")
    @test jl ≈ r_ref atol=1e-5
end

# 12. Extinction probability (R=3, various k)
println("12. extinction_probability(3, k) for various k")
r_vals = Dict(0.01=>0.9813, 0.1=>0.8379, 0.5=>0.5, 1.0=>0.3333, 4.0=>0.1354)
for (k, r_ref) in sort(collect(r_vals), by=first)
    jl = extinction_probability(3.0, k)
    println("    k=$k: Julia=$(round(jl, digits=4)), R=$r_ref")
    @test jl ≈ r_ref atol=1e-3
end

# 13. Containment under pop control (R=3)
println("13. probability_contain(3, k; pop_control=ctrl)")
r_vals = [
    (0.1, 0.25, 0.8745), (0.1, 0.5, 0.9323), (0.1, 0.75, 0.999),
    (0.5, 0.25, 0.5954), (0.5, 0.5, 0.7676), (0.5, 0.75, 0.999),
    (1.0, 0.25, 0.4444), (1.0, 0.5, 0.6667), (1.0, 0.75, 0.999),
]
for (k, ctrl, r_ref) in r_vals
    jl = probability_contain(3.0, k; pop_control=ctrl)
    println("    k=$k, ctrl=$ctrl: Julia=$(round(jl, digits=4)), R=$r_ref")
    @test jl ≈ r_ref atol=2e-3
end

println("\n=== All superspreading validations passed ===")
