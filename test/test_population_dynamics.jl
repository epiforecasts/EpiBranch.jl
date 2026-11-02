using Distributions
using LinearAlgebra
using StableRNGs

@testset "MeanFieldProcess" begin
    @testset "conserves population and matches direct integration when split" begin
        model = MeanFieldProcess(;
            population_size = 10_000, transmission_rate = 0.4, removal_rate = 0.2
        )
        direct = simulate(model; n_initial = 10, max_time = 40.0)
        @test direct.S + direct.I + direct.R ≈ 10_000
        @test direct.S < 10_000 - 10
        @test direct.I > 0

        split = simulate(model; n_initial = 10, max_time = 15.0)
        split = simulate(model, split; max_time = 40.0)
        @test split.S ≈ direct.S rtol = 1.0e-3
        @test split.I ≈ direct.I rtol = 1.0e-3
        @test split.R ≈ direct.R rtol = 1.0e-3
    end

    @testset "epidemic grows then fades for R0 > 1" begin
        model = MeanFieldProcess(;
            population_size = 5000, transmission_rate = 0.6, removal_rate = 0.2
        )
        state = simulate(model; n_initial = 5, max_time = 200.0)
        @test state.I < 1.0
        @test state.S < 5000 - 5   # some final size, not the whole population untouched
    end

    @testset "max_time must move past the state's own" begin
        model = MeanFieldProcess(;
            population_size = 500, transmission_rate = 0.4, removal_rate = 0.2
        )
        state = simulate(model; max_time = 10.0)
        @test_throws ArgumentError simulate(model, state; max_time = 10.0)
        @test_throws ArgumentError simulate(model, state; max_time = 5.0)
    end

    @testset "rejects a bad n_initial" begin
        model = MeanFieldProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test_throws ArgumentError simulate(model; n_initial = 0, max_time = 1.0)
        @test_throws ArgumentError simulate(model; n_initial = 11, max_time = 1.0)
    end

    @testset "rejects invalid construction" begin
        @test_throws ArgumentError MeanFieldProcess(;
            population_size = 0, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test_throws ArgumentError MeanFieldProcess(;
            population_size = 10, transmission_rate = -0.4, removal_rate = 0.2
        )
        @test_throws ArgumentError MeanFieldProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = -0.2
        )
        @test_throws ArgumentError MeanFieldProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = 0.2, step = 0.0
        )
    end

    @testset "show" begin
        model = MeanFieldProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test occursin("MeanFieldProcess(population_size=10", sprint(show, model))
        state = simulate(model; max_time = 1.0)
        @test occursin("MeanFieldState(S=", sprint(show, state))
    end
end

@testset "LinearNoiseProcess" begin
    @testset "mean path matches MeanFieldProcess" begin
        mean_field = MeanFieldProcess(;
            population_size = 8000, transmission_rate = 0.5, removal_rate = 0.2
        )
        lna = LinearNoiseProcess(;
            population_size = 8000, transmission_rate = 0.5, removal_rate = 0.2
        )
        mf_state = simulate(mean_field; n_initial = 8, max_time = 50.0)
        lna_state = simulate(lna; n_initial = 8, max_time = 50.0)
        @test lna_state.S ≈ mf_state.S
        @test lna_state.I ≈ mf_state.I
        @test lna_state.R ≈ mf_state.R
    end

    @testset "covariance grows from zero and stays symmetric positive semi-definite" begin
        # Early in the outbreak, while I is still a small share of the
        # population (the epidemic has not yet turned over), the covariance
        # grows monotonically; it shrinks back once the epidemic nears its
        # deterministic endpoint and few events remain to drive it.
        model = LinearNoiseProcess(;
            population_size = 8000, transmission_rate = 0.5, removal_rate = 0.2
        )
        state = simulate(model; n_initial = 8, max_time = 5.0)
        @test state.covariance ≈ state.covariance'
        @test all(>=(-1.0e-6), eigvals(Symmetric(state.covariance)))
        @test tr(state.covariance) > 0

        continued = simulate(model, state; max_time = 10.0)
        @test state.I < 0.1 * 8000   # still well before the epidemic turns over
        @test tr(continued.covariance) > tr(state.covariance)
    end

    @testset "max_time must move past the state's own" begin
        model = LinearNoiseProcess(;
            population_size = 500, transmission_rate = 0.4, removal_rate = 0.2
        )
        state = simulate(model; max_time = 10.0)
        @test_throws ArgumentError simulate(model, state; max_time = 10.0)
    end

    @testset "rejects a bad n_initial" begin
        model = LinearNoiseProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test_throws ArgumentError simulate(model; n_initial = 0, max_time = 1.0)
        @test_throws ArgumentError simulate(model; n_initial = 11, max_time = 1.0)
    end

    @testset "rejects invalid construction" begin
        @test_throws ArgumentError LinearNoiseProcess(;
            population_size = 0, transmission_rate = 0.4, removal_rate = 0.2
        )
    end

    @testset "show" begin
        model = LinearNoiseProcess(;
            population_size = 10, transmission_rate = 0.4, removal_rate = 0.2
        )
        @test occursin("LinearNoiseProcess(population_size=10", sprint(show, model))
        state = simulate(model; max_time = 1.0)
        @test occursin("LinearNoiseState(S=", sprint(show, state))
    end
end
