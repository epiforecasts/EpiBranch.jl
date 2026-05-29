# Tests against R package reference values
#
# These tests verify that EpiBranch.jl deterministic/analytical results match
# reference values computed by the R packages epichains and superspreading.
# Each test documents the R function call that produced the expected value.

@testset "R conversion targets" begin

    # ── Borel distribution density ──────────────────────────────────────
    # R: dborel(1:5, 1)  (epichains)
    # Formula: P(X=n) = exp(-mu*n + (n-1)*log(mu*n) - lgamma(n+1))
    # Computed in R 4.4 with epichains::dborel(1:5, 1):
    #   0.36787944 0.13533528 0.07468060 0.04884170 0.03509347
    @testset "Borel density matches R dborel" begin
        d = Borel(1.0)
        r_values = [0.36787944, 0.13533528, 0.07468060, 0.04884170, 0.03509347]
        for (n, r_ref) in zip(1:5, r_values)
            @test pdf(d, n) ≈ r_ref atol = 1e-6
        end
    end

    # R: dborel(1:5, 0.5)
    @testset "Borel density matches R dborel (mu=0.5)" begin
        d = Borel(0.5)
        r_values = [0.60653066, 0.18393972, 0.08367381, 0.04511176, 0.02672038]
        for (n, r_ref) in zip(1:5, r_values)
            @test pdf(d, n) ≈ r_ref atol = 1e-6
        end
    end

    # ── Chain size log-likelihood ───────────────────────────────────────
    # R: likelihood(chains=c(4,7,1,2,7,2,3,1,5,6,1,10,5,10,6,8,8,6,7,10),
    #              statistic="size", offspring_dist=rpois, lambda=0.5)
    # R uses simulation-based likelihood; our analytical Borel-based likelihood
    # should give a consistent finite negative value.
    @testset "Chain size log-likelihood is finite and negative" begin
        chain_sizes = [4, 7, 1, 2, 7, 2, 3, 1, 5, 6, 1, 10, 5, 10, 6, 8, 8, 6, 7, 10]
        ll = loglikelihood(ChainSizes(chain_sizes), Poisson(0.5))
        @test isfinite(ll)
        @test ll < 0.0
    end

    # ── Extinction probability ──────────────────────────────────────────
    # R: superspreading::probability_extinct(R=1.5, k=0.1, num_init_infect=10)
    # Expected: 0.4963112
    @testset "Extinction probability matches R probability_extinct" begin
        q = extinction_probability(1.5, 0.1)
        p = q^10
        @test p ≈ 0.4963112 atol = 1e-5
    end

    # ── Epidemic probability ────────────────────────────────────────────
    # R: 1 - superspreading::probability_extinct(R=1.5, k=0.1, num_init_infect=10)
    # Expected: 0.5036888
    @testset "Epidemic probability matches R probability_epidemic" begin
        q = extinction_probability(1.5, 0.1)
        pe = 1 - q^10
        @test pe ≈ 0.5036888 atol = 1e-5
    end

    # ── Extinction probability for various k (R=3) ─────────────────────
    # R: superspreading::probability_extinct(R=3, k=...)
    @testset "Extinction probability R=3, various k" begin
        r_vals = Dict(
            0.01 => 0.9813,
            0.1 => 0.8379,
            0.5 => 0.5,
            1.0 => 0.3333,
            4.0 => 0.1354
        )
        for (k, r_ref) in r_vals
            @test extinction_probability(3.0, k) ≈ r_ref atol = 1e-3
        end
    end

    # ── Epidemic probability varying k (R=1.5) ─────────────────────────
    # R: superspreading::probability_epidemic(R=1.5, k=...)
    @testset "Epidemic probability varying k, R=1.5" begin
        r_vals = Dict(1.0 => 0.3333333, 0.5 => 0.2324081, 0.1 => 0.06765766)
        for (k, r_ref) in r_vals
            @test epidemic_probability(1.5, k) ≈ r_ref atol = 1e-5
        end
    end

    # ── Epidemic probability varying R (k=1) ────────────────────────────
    # R: superspreading::probability_epidemic(R=..., k=1)
    @testset "Epidemic probability varying R, k=1" begin
        r_vals = Dict(0.5 => 0.0, 1.0 => 0.0, 1.5 => 0.3333333, 5.0 => 0.8)
        for (R, r_ref) in r_vals
            @test epidemic_probability(R, 1.0) ≈ r_ref atol = 1e-5
        end
    end

    # ── probability_contain ─────────────────────────────────────────────
    # R: superspreading::probability_contain(R, k, ...)
    @testset "probability_contain matches R" begin
        @testset "pop_control=0.1" begin
            # R: probability_contain(R=1.5, k=0.5, theta=0.1)
            # Expected: 0.8213172
            @test probability_contain(1.5, 0.5; pop_control = 0.1) ≈ 0.8213172 atol = 1e-5
        end

        @testset "ind_control=0.1" begin
            # R: probability_contain(R=1.5, k=0.5, theta_ind=0.1)
            # Expected: 0.8391855
            @test probability_contain(1.5, 0.5; ind_control = 0.1) ≈ 0.8391855 atol = 1e-5
        end

        @testset "both controls" begin
            # R: probability_contain(R=1.5, k=0.5, theta=0.1, theta_ind=0.1)
            # Expected: 0.8915076
            @test probability_contain(1.5, 0.5; ind_control = 0.1, pop_control = 0.1) ≈
                  0.8915076 atol = 1e-5
        end

        @testset "5 introductions with pop_control" begin
            # R: probability_contain(R=1.5, k=0.5, num_init_infect=5, theta=0.1)
            # Expected: 0.3737271
            @test probability_contain(1.5, 0.5; n_initial = 5, pop_control = 0.1) ≈
                  0.3737271 atol = 1e-5
        end

        @testset "R=1.2, pop_control=0.25 (subcritical effective R)" begin
            # R_eff = 1.2 * 0.75 = 0.9, subcritical -> containment = 1.0
            @test probability_contain(1.2, 0.5; pop_control = 0.25) == 1.0
        end
    end

    # ── Containment under pop control grid (R=3) ───────────────────────
    # R: superspreading::probability_contain(R=3, k=..., theta=...)
    @testset "probability_contain R=3 grid" begin
        r_vals = [
            (0.1, 0.25, 0.8745),
            (0.1, 0.5, 0.9323),
            (0.1, 0.75, 0.999),
            (0.5, 0.25, 0.5954),
            (0.5, 0.5, 0.7676),
            (0.5, 0.75, 0.999),
            (1.0, 0.25, 0.4444),
            (1.0, 0.5, 0.6667),
            (1.0, 0.75, 0.999)
        ]
        for (k, ctrl, r_ref) in r_vals
            jl = probability_contain(3.0, k; pop_control = ctrl)
            @test jl ≈ r_ref atol = 2e-3
        end
    end

    # ── Superspreading: proportion_transmission ─────────────────────────
    # R: superspreading::proportion_transmission(R=2.5, k=0.16, prop=0.2)
    # Known: with high overdispersion, top 20% cause ~80% of transmission
    @testset "proportion_transmission high overdispersion" begin
        prop = proportion_transmission(2.5, 0.16; prop_cases = 0.2)
        @test prop > 0.7
    end

    # ── network_R (NATSAL data) ─────────────────────────────────────────
    # R: superspreading::calc_network_R(mean_num_contact=14.1/(74-16),
    #                                   sd_num_contact=69.6/(74-16),
    #                                   infect_duration=1, p_trans=1)
    # Expected: R=0.2431034, R_net=6.166508
    @testset "network_R matches R calc_network_R" begin
        mean_c = 14.1 / (74 - 16)
        sd_c = 69.6 / (74 - 16)
        res = network_R(mean_c, sd_c, 1.0, 1.0)
        @test res.R ≈ 0.2431034 atol = 1e-3
        @test res.R_net ≈ 6.166508 atol = 1e-2
    end
end
