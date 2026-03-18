@testset "Analytical" begin
    @testset "Extinction probability" begin
        @testset "Subcritical always extinct" begin
            @test extinction_probability(0.5, 0.1) == 1.0
            @test extinction_probability(0.99, 1.0) == 1.0
            @test extinction_probability(1.0, 0.5) == 1.0
        end

        @testset "Supercritical extinction < 1" begin
            q = extinction_probability(2.0, 0.5)
            @test 0.0 < q < 1.0
        end

        @testset "Known value: R=2, k=1 (Geometric)" begin
            # For Geometric offspring (k=1, mean=R), extinction prob = 1/R
            q = extinction_probability(2.0, 1.0)
            @test q ≈ 0.5 atol=1e-8
        end

        @testset "Higher R → lower extinction" begin
            q1 = extinction_probability(1.5, 0.5)
            q2 = extinction_probability(3.0, 0.5)
            @test q1 > q2
        end

        @testset "Higher k → lower extinction (for same R)" begin
            # More dispersion (lower k) → higher extinction probability
            q_low_k = extinction_probability(2.0, 0.1)
            q_high_k = extinction_probability(2.0, 10.0)
            @test q_low_k > q_high_k
        end

        @testset "Poisson offspring" begin
            q = extinction_probability(Poisson(0.5))
            @test q == 1.0

            q = extinction_probability(Poisson(2.0))
            @test 0.0 < q < 1.0

            # Known: for Poisson(λ), extinction prob is the solution to q = exp(λ(q-1))
            # For λ=2: q ≈ 0.2032
            @test q ≈ 0.2032 atol=0.001
        end

        @testset "NegativeBinomial distribution method" begin
            d = NegBin(2.5, 0.16)
            q = extinction_probability(d)
            q2 = extinction_probability(2.5, 0.16)
            @test q ≈ q2
        end
    end

    @testset "Epidemic probability" begin
        @test epidemic_probability(0.5, 0.5) == 0.0
        @test epidemic_probability(2.0, 1.0) ≈ 0.5 atol=1e-8
    end

    @testset "Borel distribution" begin
        @testset "Construction" begin
            d = Borel(0.5)
            @test d.μ == 0.5
            @test_throws ArgumentError Borel(-0.1)
            @test_throws ArgumentError Borel(1.5)
        end

        @testset "PMF sums to ≈ 1" begin
            d = Borel(0.5)
            total = sum(pdf(d, n) for n in 1:100)
            @test total ≈ 1.0 atol=1e-6
        end

        @testset "Mean" begin
            d = Borel(0.5)
            @test mean(d) ≈ 2.0 atol=1e-10  # 1/(1-μ)
        end

        @testset "Borel(1.0) has infinite mean" begin
            d = Borel(1.0)
            @test mean(d) == Inf
        end
    end

    @testset "GammaBorel distribution" begin
        @testset "Construction" begin
            d = GammaBorel(0.5, 0.8)
            @test d.k == 0.5
            @test d.R == 0.8
            @test_throws ArgumentError GammaBorel(-1.0, 0.5)
            @test_throws ArgumentError GammaBorel(0.5, -0.5)
        end

        @testset "PMF sums to ≈ 1 for subcritical" begin
            d = GammaBorel(0.5, 0.8)
            total = sum(pdf(d, n) for n in 1:500)
            @test total ≈ 1.0 atol=1e-2
        end
    end

    @testset "Chain size distribution" begin
        @testset "Poisson offspring → Borel" begin
            d = chain_size_distribution(Poisson(0.5))
            @test d isa Borel
            @test d.μ == 0.5
        end

        @testset "NegBin offspring → GammaBorel" begin
            d = chain_size_distribution(NegBin(0.8, 0.5))
            @test d isa GammaBorel
        end
    end

    @testset "Chain size likelihood" begin
        @testset "Basic evaluation" begin
            data = [1, 1, 2, 1, 3]
            ll = chain_size_ll(data, Poisson(0.5))
            @test isfinite(ll)
            @test ll < 0.0  # log-likelihood is negative
        end

        @testset "Higher λ gives different likelihood" begin
            data = [1, 1, 1, 2, 1]
            ll1 = chain_size_ll(data, Poisson(0.3))
            ll2 = chain_size_ll(data, Poisson(0.8))
            @test ll1 != ll2
        end

        @testset "NegBin offspring" begin
            data = [1, 2, 1, 3, 1]
            ll = chain_size_ll(data, NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "With observation probability" begin
            data = [1, 1, 2]
            ll = chain_size_ll(data, Poisson(0.5), 0.8)
            @test isfinite(ll)
        end
    end

    @testset "Chain length likelihood" begin
        @testset "Poisson offspring" begin
            data = [0, 1, 0, 2, 1]
            ll = chain_length_ll(data, Poisson(0.5))
            @test isfinite(ll)
        end

        @testset "NegBin offspring" begin
            data = [0, 1, 0, 2, 1]
            ll = chain_length_ll(data, NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "Supercritical throws" begin
            @test_throws ArgumentError chain_length_ll([1, 2], Poisson(1.5))
            @test_throws ArgumentError chain_length_ll([1, 2], NegBin(1.5, 0.5))
        end
    end

    @testset "Simulation-based chain size likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            data = [1, 1, 2, 1, 3]
            ll = chain_size_ll(data, model; n_sim=5000, rng=StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end

        @testset "Consistent with analytical for Poisson" begin
            data = [1, 1, 1, 2, 1]
            ll_analytical = chain_size_ll(data, Poisson(0.5))
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll_simulated = chain_size_ll(data, model; n_sim=50_000, rng=StableRNG(42))
            # Should be in the same ballpark (simulation noise)
            @test abs(ll_analytical - ll_simulated) < 1.0
        end

        @testset "With interventions" begin
            model = BranchingProcess(Poisson(2.0), Exponential(5.0))
            iso = Isolation(delay=Exponential(1.0))
            data = [1, 1, 2, 1]
            ll = chain_size_ll(data, model;
                interventions=[iso],
                attributes=clinical_presentation(incubation_period=LogNormal(1.5, 0.5)),
                n_sim=5000, rng=StableRNG(42))
            @test isfinite(ll)
        end
    end

    @testset "Simulation-based chain length likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            data = [0, 1, 0, 2, 1]
            ll = chain_length_ll(data, model; n_sim=5000, rng=StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end
    end

    @testset "Proportion transmission (superspreading)" begin
        @testset "Basic 80/20" begin
            # With high R and low k, top 20% should cause most transmission
            prop = proportion_transmission(2.5, 0.16; prop_cases=0.2)
            @test 0.0 < prop < 1.0
            @test prop > 0.5  # high overdispersion → top 20% cause > 50%
        end

        @testset "No overdispersion → more equal" begin
            # High k means little overdispersion
            prop_low_k = proportion_transmission(2.0, 0.1; prop_cases=0.2)
            prop_high_k = proportion_transmission(2.0, 100.0; prop_cases=0.2)
            # Lower k → more unequal → top 20% cause more
            @test prop_low_k > prop_high_k
        end

        @testset "Argument validation" begin
            @test_throws ArgumentError proportion_transmission(-1.0, 0.5)
            @test_throws ArgumentError proportion_transmission(2.0, -0.5)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases=0.0)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases=1.0)
        end

        @testset "Known approximate values" begin
            # For k → ∞ (Poisson limit), distribution is nearly uniform
            # Top 20% should cause ≈ 20% of transmission
            prop = proportion_transmission(2.0, 1000.0; prop_cases=0.2)
            @test prop ≈ 0.2 atol=0.05
        end
    end
end
