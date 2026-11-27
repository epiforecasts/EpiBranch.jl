using EpiBranch, Distributions, Random, Statistics

@testset "Count simulation with immunity" begin
    gt = DiscreteNonParametric([1], [1.0])
    model = BranchingProcess(Poisson(2.0), gt; population = Population(100; immune = 40))
    state = simulate(model; n_initial = 10, max_time = 0, max_generations = nothing, rng = Xoshiro(1))
    @test state isa EpiBranch.CountSimulationState
    @test state.susceptibles == [50]
    @test_throws ArgumentError simulate(model; n_initial = 61, max_time = 0, max_generations = nothing, rng = Xoshiro(1))

    M = [1.0 0.0; 0.0 1.0]
    grouped = BranchingProcess(M, R -> Poisson(R), gt; population = Population([100, 80]; immune = [40, 30]))
    state = simulate(grouped; index_types = [1, 2], max_time = 0, max_generations = nothing, rng = Xoshiro(2))
    @test state isa EpiBranch.CountSimulationState
    @test state.susceptibles == [59, 49]

    # One day's expected infections follow sequential depletion of susceptibles.
    runs = [simulate(model; n_initial = 10, max_time = 1, max_generations = nothing, rng = Xoshiro(i)) for i in 1:2000]
    infections = [s.counts[2] for s in runs]
    expected = 50 * (1 - exp(-20 / 100))
    @test abs(mean(infections) - expected) < 5 * std(infections) / sqrt(length(infections))
end
