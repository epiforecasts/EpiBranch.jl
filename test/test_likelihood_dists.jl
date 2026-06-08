@testset "Likelihood distributions" begin
    @testset "offspring_distribution returns the bare offspring" begin
        bp = BranchingProcess(Poisson(1.0))
        d = offspring_distribution(bp)
        @test d isa Poisson
        data = [0, 1, 2, 0, 3, 1, 0]
        @test loglikelihood(d, data) ≈ loglikelihood(OffspringCounts(data), Poisson(1.0))
    end

    @testset "chain_size_distribution — bare model gives the analytical form" begin
        bp = BranchingProcess(Poisson(0.5))
        d = chain_size_distribution(bp)
        @test d isa Borel
        data = [1, 2, 1, 3, 1, 5, 2]
        @test sum(logpdf(d, n) for n in data) ≈
              loglikelihood(ChainSizes(data), Poisson(0.5))

        bp_nb = BranchingProcess(NegBin(0.5, 0.5))
        d_nb = chain_size_distribution(bp_nb)
        @test d_nb isa GammaBorel
    end

    @testset "chain_size_distribution — kwargs trigger the wrapper" begin
        bp = BranchingProcess(Poisson(0.5))
        data = [3, 5, 10, 2]
        seeds = [1, 2, 1, 1]

        d = chain_size_distribution(bp; seeds = seeds)
        @test logpdf(d, data) ≈
              loglikelihood(ChainSizes(data; seeds = seeds), Poisson(0.5))

        pi = [1.0, 0.5, 0.0, 0.7]
        d_pi = chain_size_distribution(bp; seeds = seeds, pi = pi)
        @test logpdf(d_pi, data) ≈
              loglikelihood(ChainSizes(data; seeds = seeds), Poisson(0.5); pi = pi)
    end

    @testset "chain_length_distribution" begin
        bp = BranchingProcess(Poisson(0.5))
        d = chain_length_distribution(bp)
        data = [0, 1, 2, 0, 3, 1]
        @test logpdf(d, data) ≈ loglikelihood(ChainLengths(data), Poisson(0.5))
    end

    @testset "chain_size_distribution — Observed wrapper" begin
        data = [1, 2, 1, 3, 1]
        bp = BranchingProcess(Poisson(0.5))
        obs = Observed(bp, PerCaseObservation(; detection_prob = 0.7))
        d = chain_size_distribution(obs; n_sim = 0)
        # n_sim=0 forces the wrapper path; logpdf should still match the
        # analytical loglikelihood with no kwargs since there are no
        # interventions.
        @test logpdf(d, data) ≈ loglikelihood(ChainSizes(data), obs)
    end

    @testset "rand draws chain sizes from the underlying model" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        d = chain_size_distribution(bp; sim_opts = SimOpts())
        samples = rand(StableRNG(7), d, 50)
        @test length(samples) == 50
        @test all(s -> s >= 1, samples)
    end
end
