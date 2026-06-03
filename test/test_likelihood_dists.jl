@testset "Likelihood distributions" begin
    @testset "OffspringCountLikelihood" begin
        data = [0, 1, 2, 0, 3, 1, 0]
        d = OffspringCountLikelihood(Poisson(1.0))
        @test logpdf(d, data) ≈ loglikelihood(OffspringCounts(data), Poisson(1.0))
        @test Distributions.loglikelihood(d, data) ≈ logpdf(d, data)
    end

    @testset "ChainSizeLikelihood — analytical fast path" begin
        data = [1, 2, 1, 3, 1, 5, 2]
        d = ChainSizeLikelihood(Poisson(0.5))
        @test logpdf(d, data) ≈ loglikelihood(ChainSizes(data), Poisson(0.5))

        d_nb = ChainSizeLikelihood(NegBin(0.5, 0.5))
        @test logpdf(d_nb, data) ≈ loglikelihood(ChainSizes(data), NegBin(0.5, 0.5))
    end

    @testset "ChainSizeLikelihood — seeds and pi" begin
        data = [3, 5, 10, 2]
        seeds = [1, 2, 1, 1]
        d = ChainSizeLikelihood(Poisson(0.5); seeds = seeds)
        @test logpdf(d, data) ≈
              loglikelihood(ChainSizes(data; seeds = seeds), Poisson(0.5))

        pi = [1.0, 0.5, 0.0, 0.7]
        d_pi = ChainSizeLikelihood(Poisson(0.5); seeds = seeds, pi = pi)
        @test logpdf(d_pi, data) ≈
              loglikelihood(ChainSizes(data; seeds = seeds), Poisson(0.5); pi = pi)
    end

    @testset "ChainLengthLikelihood" begin
        data = [0, 1, 2, 0, 3, 1]
        d = ChainLengthLikelihood(Poisson(0.5))
        @test logpdf(d, data) ≈ loglikelihood(ChainLengths(data), Poisson(0.5))

        d_nb = ChainLengthLikelihood(NegBin(0.5, 0.5))
        @test logpdf(d_nb, data) ≈ loglikelihood(ChainLengths(data), NegBin(0.5, 0.5))
    end

    @testset "ChainSizeLikelihood — Observed wrapper" begin
        data = [1, 2, 1, 3, 1]
        bp = BranchingProcess(Poisson(0.5))
        obs = Observed(bp, PerCaseObservation(; detection_prob = 0.7))
        d = ChainSizeLikelihood(obs)
        @test logpdf(d, data) ≈ loglikelihood(ChainSizes(data), obs)
    end
end
