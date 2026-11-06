using Distributions
using LinearAlgebra
using StableRNGs

# Two age bands of fixed, deterministic size, exactly as the issue's example
# assigns them (`ind.id <= n[1] ? 1 : 2`), so the group sizes the approximation
# derives match what the exact race realises every time.
function _two_band_spec(; n = [2_000, 8_000], within = 0.3, between = 0.1, γ = 0.2, calendar = nothing)
    band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
    rate = (from, to) -> (from == to ? within : between) / n[to[1]]
    return ModelSpec(
        MixingProcess(;
            population_size = sum(n), mixing_by = (:age_band,), rate, calendar
        );
        progression = [Transition(:recovered; rate = γ, terminal = true)],
        attributes = band
    )
end

@testset "MixingMeanFieldProcess" begin
    @testset "reads the group sizes and the between-group rate from the spec" begin
        spec = _two_band_spec()
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        @test EpiBranch.population_size(model) == 10_000
        @test model.groups == [(1,), (2,)]
        @test model.group_sizes == [2_000, 8_000]
        # Λ[g, h] = rate(h, g): within-group rate on the diagonal, between-group
        # rate off it, each already divided by the target group's own size.
        @test model.contact_matrix[1, 1] ≈ 0.3 / 2_000
        @test model.contact_matrix[2, 2] ≈ 0.3 / 8_000
        @test model.contact_matrix[1, 2] ≈ 0.1 / 2_000
        @test model.contact_matrix[2, 1] ≈ 0.1 / 8_000
    end

    @testset "mixing_by = () reproduces MeanFieldProcess exactly" begin
        N = 5_000
        β, γ = 0.4, 0.2
        spec = ModelSpec(
            MixingProcess(; population_size = N, rate = (from, to) -> β / N);
            progression = [Transition(:recovered; rate = γ, terminal = true)]
        )
        structured = MixingMeanFieldProcess(spec; removal_rate = γ)
        homogeneous = MeanFieldProcess(;
            population_size = N, transmission_rate = β, removal_rate = γ
        )

        s_state = simulate(structured; n_initial = 10, max_time = 40.0)
        h_state = simulate(homogeneous; n_initial = 10, max_time = 40.0)
        @test only(s_state.S) ≈ h_state.S
        @test only(s_state.I) ≈ h_state.I
        @test only(s_state.R) ≈ h_state.R
    end

    @testset "conserves population per group and matches direct integration when split" begin
        spec = _two_band_spec()
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)

        direct = simulate(model; n_initial = 10, max_time = 40.0)
        @test direct.S + direct.I + direct.R ≈ model.group_sizes
        @test sum(direct.S) < 10_000 - 10
        @test sum(direct.I) > 0

        split = simulate(model; n_initial = 10, max_time = 15.0)
        split = simulate(model, split; max_time = 40.0)
        @test split.S ≈ direct.S rtol = 1.0e-3
        @test split.I ≈ direct.I rtol = 1.0e-3
        @test split.R ≈ direct.R rtol = 1.0e-3
    end

    @testset "the more mixed group suffers the larger attack rate" begin
        # Band 1 is both smaller and mixes within itself three times as hard
        # as band 2 does, so it should burn through a larger share of its own
        # members once the epidemic has run its course.
        spec = _two_band_spec(; n = [2_000, 8_000], within = 0.9, between = 0.1)
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        state = simulate(model; n_initial = 10, max_time = 300.0)
        attack_rate = (model.group_sizes .- state.S) ./ model.group_sizes
        @test attack_rate[1] > attack_rate[2]
    end

    @testset "a calendar scales the contact matrix" begin
        schedule = Steps([10.0], [1.0, 0.0])
        spec = _two_band_spec(; calendar = schedule)
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        before = simulate(model; n_initial = 10, max_time = 10.0)
        after = simulate(model, before; max_time = 40.0)
        # Transmission is switched off from day 10 onwards, so nobody new is
        # infected and the epidemic simply decays by removal from then on.
        @test sum(after.S) ≈ sum(before.S)
        @test sum(after.I) < sum(before.I)
    end

    @testset "agrees with exact simulation of the same structured model in a large population" begin
        N = [20_000, 80_000]
        spec = _two_band_spec(; n = N, within = 0.5, between = 0.1, γ = 0.2)
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)

        approx = simulate(model; n_initial = 20, max_time = 120.0)
        exact = simulate(spec; n_initial = 20, max_time = 120.0, rng = StableRNG(7))
        exact_final_size = count(i -> isfinite(i.infection_time), exact.individuals)

        # A near-deterministic epidemic this large settles close to the
        # mean-field path from the same initial condition (Kurtz's theorem),
        # whatever randomness decided exactly how it unfolded.
        @test isapprox(sum(approx.S) + sum(approx.I), sum(N) - exact_final_size; atol = 0.03 * sum(N))
    end

    @testset "rejects a model it cannot be built from" begin
        bare = MixingProcess(; population_size = 10, rate = (from, to) -> 0.1)
        @test_throws ArgumentError MixingMeanFieldProcess(bare; removal_rate = 0.2)

        homogeneous_spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 0.4, population_size = 10);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        @test_throws ArgumentError MixingMeanFieldProcess(homogeneous_spec; removal_rate = 0.2)

        vaccinated_spec = ModelSpec(
            _two_band_spec();
            interventions = [MassVaccination(; efficacy = 0.7, eligibility_time = 30.0)]
        )
        @test_throws ArgumentError MixingMeanFieldProcess(vaccinated_spec; removal_rate = 0.2)
    end

    @testset "rejects invalid construction" begin
        spec = _two_band_spec()
        @test_throws ArgumentError MixingMeanFieldProcess(spec; removal_rate = -0.2)
        @test_throws ArgumentError MixingMeanFieldProcess(spec; removal_rate = 0.2, step = 0.0)
    end

    @testset "rejects a bad n_initial" begin
        model = MixingMeanFieldProcess(_two_band_spec(); removal_rate = 0.2)
        @test_throws ArgumentError simulate(model; n_initial = 0, max_time = 1.0)
        @test_throws ArgumentError simulate(model; n_initial = 10_001, max_time = 1.0)
    end

    @testset "max_time must move past the state's own" begin
        model = MixingMeanFieldProcess(_two_band_spec(); removal_rate = 0.2)
        state = simulate(model; max_time = 10.0)
        @test_throws ArgumentError simulate(model, state; max_time = 10.0)
    end

    @testset "show" begin
        model = MixingMeanFieldProcess(_two_band_spec(); removal_rate = 0.2)
        @test occursin("MixingMeanFieldProcess(population_size=10000", sprint(show, model))
        state = simulate(model; max_time = 1.0)
        @test occursin("MixingMeanFieldState(S=", sprint(show, state))
    end
end

@testset "MixingLinearNoiseProcess" begin
    @testset "mean path matches MixingMeanFieldProcess" begin
        spec = _two_band_spec()
        mean_field = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        lna = MixingLinearNoiseProcess(spec; removal_rate = 0.2)

        mf_state = simulate(mean_field; n_initial = 8, max_time = 50.0)
        lna_state = simulate(lna; n_initial = 8, max_time = 50.0)
        @test lna_state.S ≈ mf_state.S
        @test lna_state.I ≈ mf_state.I
        @test lna_state.R ≈ mf_state.R
    end

    @testset "covariance grows from zero and stays symmetric positive semi-definite" begin
        spec = _two_band_spec()
        model = MixingLinearNoiseProcess(spec; removal_rate = 0.2)
        state = simulate(model; n_initial = 8, max_time = 5.0)
        @test state.covariance ≈ state.covariance'
        @test all(>=(-1.0e-6), eigvals(Symmetric(state.covariance)))
        @test tr(state.covariance) > 0

        continued = simulate(model, state; max_time = 10.0)
        @test tr(continued.covariance) > tr(state.covariance)
    end

    @testset "rejects invalid construction" begin
        bare = MixingProcess(; population_size = 10, rate = (from, to) -> 0.1)
        @test_throws ArgumentError MixingLinearNoiseProcess(bare; removal_rate = 0.2)
    end

    @testset "show" begin
        model = MixingLinearNoiseProcess(_two_band_spec(); removal_rate = 0.2)
        @test occursin("MixingLinearNoiseProcess(population_size=10000", sprint(show, model))
        state = simulate(model; max_time = 1.0)
        @test occursin("MixingLinearNoiseState(S=", sprint(show, state))
    end
end
