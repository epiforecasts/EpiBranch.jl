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
            ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3]), Poisson(0.5))
            @test isfinite(ll)
            @test ll < 0.0
        end

        @testset "Higher λ gives different likelihood" begin
            data = ChainSizes([1, 1, 1, 2, 1])
            ll1 = loglikelihood(data, Poisson(0.3))
            ll2 = loglikelihood(data, Poisson(0.8))
            @test ll1 != ll2
        end

        @testset "NegBin offspring" begin
            ll = loglikelihood(ChainSizes([1, 2, 1, 3, 1]), NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "With observation probability" begin
            ll = loglikelihood(ChainSizes([1, 1, 2]; obs_prob = 0.8), Poisson(0.5))
            @test isfinite(ll)
        end
    end

    @testset "Chain length likelihood" begin
        @testset "Poisson offspring" begin
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), Poisson(0.5))
            @test isfinite(ll)
        end

        @testset "NegBin offspring" begin
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "Supercritical throws" begin
            @test_throws ArgumentError loglikelihood(ChainLengths([1, 2]), Poisson(1.5))
            @test_throws ArgumentError loglikelihood(ChainLengths([1, 2]), NegBin(1.5, 0.5))
        end
    end

    @testset "Simulation-based chain size likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3]), model; n_sim = 5000, rng = StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end

        @testset "Consistent with analytical for Poisson" begin
            data = ChainSizes([1, 1, 1, 2, 1])
            ll_analytical = loglikelihood(data, Poisson(0.5))
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll_simulated = loglikelihood(data, model; n_sim = 10_000, rng = StableRNG(42))
            @test abs(ll_analytical - ll_simulated) < 1.0
        end

        @testset "With interventions" begin
            model = BranchingProcess(Poisson(2.0), Exponential(5.0))
            iso = Isolation(delay = Exponential(1.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1]), model;
                interventions = [iso],
                attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
                sim_opts = SimOpts(max_cases = 500),
                n_sim = 500, rng = StableRNG(42))
            @test isfinite(ll)
        end
    end

    @testset "Simulation-based chain length likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), model; n_sim = 5000, rng = StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end
    end

    @testset "Proportion transmission (superspreading)" begin
        @testset "Basic 80/20" begin
            # With high R and low k, top 20% should cause most transmission
            prop = proportion_transmission(2.5, 0.16; prop_cases = 0.2)
            @test 0.0 < prop < 1.0
            @test prop > 0.5  # high overdispersion → top 20% cause > 50%
        end

        @testset "No overdispersion → more equal" begin
            # High k means little overdispersion
            prop_low_k = proportion_transmission(2.0, 0.1; prop_cases = 0.2)
            prop_high_k = proportion_transmission(2.0, 100.0; prop_cases = 0.2)
            # Lower k → more unequal → top 20% cause more
            @test prop_low_k > prop_high_k
        end

        @testset "Argument validation" begin
            @test_throws ArgumentError proportion_transmission(-1.0, 0.5)
            @test_throws ArgumentError proportion_transmission(2.0, -0.5)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases = 0.0)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases = 1.0)
        end

        @testset "Known approximate values" begin
            # For k → ∞ (Poisson limit), distribution is nearly uniform
            # Top 20% should cause ≈ 20% of transmission
            prop = proportion_transmission(2.0, 1000.0; prop_cases = 0.2)
            @test prop ≈ 0.2 atol=0.05
        end
    end

    @testset "Proportion cluster size" begin
        @testset "High overdispersion concentrates cases" begin
            # k=0.1 → most cases from large clusters
            prop = proportion_cluster_size(2.0, 0.1; cluster_size = 5)
            @test 0.0 < prop < 1.0
            @test prop > 0.5
        end

        @testset "Low overdispersion spreads cases" begin
            prop_low_k = proportion_cluster_size(2.0, 0.1; cluster_size = 5)
            prop_high_k = proportion_cluster_size(2.0, 10.0; cluster_size = 5)
            @test prop_low_k > prop_high_k
        end

        @testset "cluster_size=1 captures everything" begin
            prop = proportion_cluster_size(2.0, 0.5; cluster_size = 1)
            @test prop ≈ 1.0 atol=1e-6
        end
    end

    @testset "Containment probability" begin
        @testset "Subcritical always contained" begin
            @test probability_contain(0.5, 0.5) == 1.0
        end

        @testset "Supercritical partially contained" begin
            p = probability_contain(2.0, 0.5)
            @test 0.0 < p < 1.0
        end

        @testset "Matches extinction_probability for defaults" begin
            q = extinction_probability(2.0, 0.5)
            p = probability_contain(2.0, 0.5)
            @test p ≈ q atol=1e-8
        end

        @testset "Multiple introductions reduce containment" begin
            p1 = probability_contain(2.0, 0.5; n_initial = 1)
            p5 = probability_contain(2.0, 0.5; n_initial = 5)
            @test p5 < p1
            @test p5 ≈ p1^5 atol=1e-8
        end

        @testset "Population control increases containment" begin
            p_none = probability_contain(2.0, 0.5; pop_control = 0.0)
            p_half = probability_contain(2.0, 0.5; pop_control = 0.5)
            @test p_half > p_none
        end

        @testset "Individual control increases containment" begin
            p_none = probability_contain(2.0, 0.5; ind_control = 0.0)
            p_half = probability_contain(2.0, 0.5; ind_control = 0.5)
            @test p_half > p_none
        end

        @testset "Full control → certain containment" begin
            @test probability_contain(5.0, 0.1; pop_control = 0.9) ≈ 1.0 atol=1e-6
        end
    end

    @testset "Network R" begin
        @testset "Homogeneous contacts" begin
            result = network_R(10.0, 0.0, 1.0, 0.1)
            @test result.R ≈ 1.0
            @test result.R_net ≈ 1.0  # no variance → no adjustment
        end

        @testset "Heterogeneous contacts amplify R" begin
            result = network_R(10.0, 20.0, 1.0, 0.1)
            @test result.R_net > result.R
        end

        @testset "Zero contacts" begin
            result = network_R(0.0, 0.0, 1.0, 0.5)
            @test result.R == 0.0
            @test result.R_net == 0.0
        end
    end

    @testset "Offspring likelihood" begin
        @testset "Basic evaluation" begin
            ll = loglikelihood(OffspringCounts([0, 1, 2, 0, 3]), Poisson(1.2))
            @test isfinite(ll)
            @test ll < 0.0
        end
    end

    @testset "fit" begin
        @testset "Poisson from offspring counts" begin
            rng = StableRNG(42)
            data = OffspringCounts(rand(rng, Poisson(2.5), 1000))
            d = fit(Poisson, data)
            @test d isa Poisson
            @test mean(d) ≈ 2.5 atol=0.2
        end

        @testset "NegBin from offspring counts" begin
            rng = StableRNG(42)
            d_true = NegBin(2.0, 0.5)
            raw = rand(rng, Distributions.NegativeBinomial(d_true.r, d_true.p), 2000)
            d = fit(NegativeBinomial, OffspringCounts(raw))
            @test mean(d) ≈ 2.0 atol=0.3
            @test d.r ≈ 0.5 atol=0.2
        end

        @testset "NegBin with low overdispersion returns high k" begin
            rng = StableRNG(42)
            data = OffspringCounts(rand(rng, Poisson(3.0), 500))
            d = fit(NegativeBinomial, data)
            @test mean(d) ≈ 3.0 atol=0.3
            @test d.r > 10.0
        end

        @testset "All zeros" begin
            d = fit(Poisson, OffspringCounts(zeros(Int, 100)))
            @test mean(d) == 0.0
        end

        @testset "Poisson from chain sizes" begin
            rng = StableRNG(42)
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            states = simulate_batch(model, 500; rng = rng)
            sizes = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(sizes, cs.size)
            end
            d = fit(Poisson, ChainSizes(sizes))
            @test d isa Poisson
            @test mean(d) ≈ 0.5 atol=0.2
        end

        @testset "NegBin from chain lengths" begin
            rng = StableRNG(42)
            true_R, true_k = 0.6, 0.5
            model = BranchingProcess(NegBin(true_R, true_k), Exponential(5.0))
            states = simulate_batch(model, 500; rng = rng)
            lengths = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(lengths, cs.length)
            end
            d = fit(NegativeBinomial, ChainLengths(lengths))
            @test d isa NegativeBinomial
            @test mean(d) ≈ true_R atol=0.3
        end

        @testset "MLE maximises likelihood" begin
            rng = StableRNG(42)
            data = OffspringCounts(rand(rng, Poisson(2.0), 200))
            d_mle = fit(Poisson, data)
            ll_mle = loglikelihood(data, d_mle)
            ll_other = loglikelihood(data, Poisson(5.0))
            @test ll_mle >= ll_other
        end
    end

    @testset "Data wrapper validation" begin
        @test_throws ArgumentError OffspringCounts(Int[])
        @test_throws ArgumentError OffspringCounts([-1, 0, 1])
        @test_throws ArgumentError ChainSizes(Int[])
        @test_throws ArgumentError ChainSizes([0, 1, 2])
        @test_throws ArgumentError ChainLengths(Int[])
        @test_throws ArgumentError ChainLengths([-1, 0])
    end
end
