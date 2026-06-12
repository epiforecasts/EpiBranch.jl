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

    @testset "chain_size_distribution — per-case observation" begin
        data = [1, 2, 1, 3, 1]
        obs = BranchingProcess(Poisson(0.5); observation = PerCaseObservation(;
            detection_prob = 0.7))
        d = chain_size_distribution(obs; n_sim = 0)
        # n_sim=0 forces the wrapper path; logpdf should still match the
        # analytical loglikelihood with no kwargs since there are no
        # interventions.
        @test logpdf(d, data) ≈ loglikelihood(ChainSizes(data), obs)
    end

    @testset "rand draws chain sizes from the underlying model" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        d = chain_size_distribution(bp)
        samples = rand(StableRNG(7), d, 50)
        @test length(samples) == 50
        @test all(s -> s >= 1, samples)
    end

    @testset "rand sums multi-seed clusters rather than dropping chains" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        # Three seeds per cluster: each draw is the total infected across
        # all three seed chains, so every cluster size is at least 3.
        d = chain_size_distribution(bp; n_initial = 3)
        samples = rand(StableRNG(7), d, 50)
        @test length(samples) == 50
        @test all(s -> s >= 3, samples)

        # Chain length aggregates with `maximum`, so deeper seed chains
        # are not lost.
        dl = chain_length_distribution(bp; n_initial = 3)
        lengths = rand(StableRNG(7), dl, 50)
        @test length(lengths) == 50
        @test all(l -> l >= 0, lengths)
    end

    @testset "coverage matrix: every cell is a Distribution or a clear ArgumentError" begin
        # Locks the model × data-type matrix so a gap can't silently
        # reopen: each entry point either returns something usable with
        # Turing's `~` (a `Distribution`) or refuses with a user-facing
        # `ArgumentError`. No bare `MethodError`s, no developer-only hints.
        bp = BranchingProcess(Poisson(0.5))
        bp_obs = BranchingProcess(Poisson(0.5);
            observation = PerCaseObservation(detection_prob = 0.6))
        bp_cm = BranchingProcess(ClusterMixed(Poisson, Gamma(2.0, 0.4)))
        bp_mt = BranchingProcess([1.0 0.5; 0.5 1.0],
            R -> NegBin(R, 0.16), LogNormal(1.6, 0.5))
        net = NetworkProcess([[2], [1, 3], [2]], 0.5)

        @testset "chain_size_distribution" begin
            @test chain_size_distribution(bp) isa Distribution
            @test chain_size_distribution(bp_obs) isa Distribution
            @test chain_size_distribution(bp_cm) isa Distribution
            @test_throws ArgumentError chain_size_distribution(bp_mt)
            @test_throws ArgumentError chain_size_distribution(net)
        end

        @testset "chain_length_distribution" begin
            @test chain_length_distribution(bp) isa Distribution
            @test chain_length_distribution(net) isa Distribution
            # per-case detection has no well-defined chain length; the
            # refusal surfaces when the law is evaluated, not at construction.
            dl = chain_length_distribution(bp_obs)
            @test_throws ArgumentError logpdf(dl, [0, 1, 2])
        end

        @testset "offspring_distribution" begin
            @test offspring_distribution(bp) isa Distribution
            @test_throws ArgumentError offspring_distribution(bp_cm)
            @test_throws ArgumentError offspring_distribution(bp_mt)
            @test_throws ArgumentError offspring_distribution(net)
        end
    end
end
