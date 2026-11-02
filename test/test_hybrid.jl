using Distributions
using StableRNGs

@testset "HybridProcess" begin
    @testset "construction validates its two models" begin
        exact = HomogeneousProcess(; transmission_rate = 0.4, population_size = 500)
        approx = MeanFieldProcess(;
            population_size = 500, transmission_rate = 0.4, removal_rate = 0.2
        )
        mismatched = MeanFieldProcess(;
            population_size = 400, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test_throws ArgumentError HybridProcess(exact, mismatched; threshold = 10)
        @test_throws ArgumentError HybridProcess(
            BranchingProcess(NegBin(2.0, 0.2), Gamma(2.5, 2.0)), approx; threshold = 10
        )
        @test HybridProcess(exact, approx; threshold = 10) isa HybridProcess
    end

    @testset "never crosses the threshold on a small population: agrees with exact simulation" begin
        exact = ModelSpec(
            HomogeneousProcess(; transmission_rate = 0.4, population_size = 50);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        approx = MeanFieldProcess(;
            population_size = 50, transmission_rate = 0.4, removal_rate = 0.2
        )
        model = HybridProcess(exact, approx; threshold = 1000, check_interval = 2.0)

        hybrid = simulate(model; n_initial = 2, max_time = 60.0, rng = StableRNG(1))
        @test hybrid.phase === :exact
        @test hybrid.approx === nothing

        direct = simulate(exact; n_initial = 2, max_time = 60.0, rng = StableRNG(1))
        @test hybrid.exact.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(hybrid.exact.individuals, direct.individuals)
        )
    end

    @testset "crosses the threshold on a large population: agrees with the mean-field path" begin
        N = 20_000
        β, γ = 0.6, 0.2
        exact = ModelSpec(
            HomogeneousProcess(; transmission_rate = β, population_size = N);
            progression = [Transition(:recovered; rate = γ, terminal = true)]
        )
        approx = MeanFieldProcess(; population_size = N, transmission_rate = β, removal_rate = γ)
        model = HybridProcess(exact, approx; threshold = 10, check_interval = 1.0)

        hybrid = simulate(model; n_initial = 5, max_time = 150.0, rng = StableRNG(2))
        @test hybrid.phase === :approx
        @test hybrid.approx.t == 150.0

        # The pure mean-field path from the same initial condition: once an
        # outbreak this large has taken off, its final size is close to the
        # deterministic model's, by the usual law-of-large-numbers argument
        # for a near-deterministic epidemic (Kurtz's theorem), whatever the
        # (random) time it took to cross the threshold.
        pure = simulate(approx; n_initial = 5, max_time = 150.0)
        @test isapprox(hybrid.approx.S, pure.S; atol = 0.03 * N)
        @test isapprox(hybrid.approx.R, pure.R; atol = 0.03 * N)
    end

    @testset "a continued hybrid state picks up where it stopped" begin
        N = 20_000
        β, γ = 0.6, 0.2
        exact = ModelSpec(
            HomogeneousProcess(; transmission_rate = β, population_size = N);
            progression = [Transition(:recovered; rate = γ, terminal = true)]
        )
        approx = MeanFieldProcess(; population_size = N, transmission_rate = β, removal_rate = γ)
        model = HybridProcess(exact, approx; threshold = 10, check_interval = 1.0)

        chained = simulate(model; n_initial = 5, max_time = 50.0, rng = StableRNG(3))
        chained = simulate(model, chained; max_time = 150.0)
        direct = simulate(model; n_initial = 5, max_time = 150.0, rng = StableRNG(3))

        @test chained.phase === direct.phase === :approx
        @test chained.approx.S ≈ direct.approx.S
        @test chained.approx.I ≈ direct.approx.I
        @test chained.approx.R ≈ direct.approx.R
    end

    @testset "an already-extinct exact phase is left alone" begin
        exact = ModelSpec(
            HomogeneousProcess(; transmission_rate = 0.4, population_size = 20);
            progression = [Transition(:recovered; rate = 5.0, terminal = true)]
        )
        approx = MeanFieldProcess(;
            population_size = 20, transmission_rate = 0.4, removal_rate = 5.0
        )
        model = HybridProcess(exact, approx; threshold = 1000, check_interval = 2.0)
        state = simulate(model; n_initial = 2, max_time = 50.0, rng = StableRNG(5))
        @test state.exact.extinct
        continued = simulate(model, state; max_time = 200.0)
        @test continued.exact === state.exact
    end
end
