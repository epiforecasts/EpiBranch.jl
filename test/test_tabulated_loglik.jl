@testset "Chain-size likelihoods tabulate repeated sizes" begin
    sizes = [1, 1, 2, 1, 3, 2, 1, 5, 1, 2, 3, 1]
    seeds = [1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1]

    @testset "single-seed fast paths match the direct sum" begin
        data = ChainSizes(sizes)
        @test loglikelihood(data, Poisson(0.7)) ≈
            sum(logpdf(Borel(0.7), n) for n in sizes)
        nb = NegBin(0.8, 0.3)
        @test loglikelihood(data, nb) ≈
            sum(logpdf(chain_size_distribution(nb), n) for n in sizes)
    end

    @testset "general path counts (size, seeds) pairs" begin
        data = ChainSizes(sizes; seeds = seeds)
        law = chain_size_distribution(NegBin(0.8, 0.3))
        expected = sum(
            EpiBranch._chain_size_logpdf(law, n, s) for (n, s) in zip(sizes, seeds)
        )
        @test loglikelihood(data, law) ≈ expected
        @test loglikelihood(data, law; prob_concluded = ones(length(sizes))) ≈ expected
    end
end
