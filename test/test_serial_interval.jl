# Tests for the CensoredDistributions.jl extension: the distribution of the
# serial interval between a linked pair, and its use with
# `double_interval_censored`. Loading CensoredDistributions.jl and
# ConvolvedDistributions.jl together activates `serial_interval`'s only
# method.

@testset "Serial interval (CensoredDistributions.jl extension)" begin
    gt = Gamma(5.95, 1 / 0.77)
    incubation = LogNormal(1.6, 0.5)
    clinical = clinical_presentation(incubation_period = incubation)

    @testset "mean is the generation time's: the incubation periods cancel" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt); attributes = clinical)
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        infector, infectee = state.individuals[1], state.individuals[2]
        si = serial_interval(spec, infector, infectee)
        @test mean(si) ≈ mean(gt)
        # Variances add: the generation time's, and the two incubation
        # periods' through the convolution and the difference alike.
        @test var(si) ≈ var(gt) + 2 * var(incubation)
    end

    @testset "double_interval_censored gives a finite likelihood" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt); attributes = clinical)
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        infector, infectee = state.individuals[1], state.individuals[2]
        si = serial_interval(spec, infector, infectee)
        obs = double_interval_censored(si; interval = 1, upper = 21.0)
        @test isfinite(logpdf(obs, 6))
        @test 0 < pdf(obs, 6) < 1
    end

    @testset "a bare BranchingProcess has no attributes to read the incubation period from" begin
        # clinical_presentation attaches through ModelSpec's `attributes`; a
        # bare process alone can never supply the incubation period.
        bare = BranchingProcess(NegBin(2.5, 0.16), gt)
        spec = ModelSpec(bare; attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        infector, infectee = state.individuals[1], state.individuals[2]
        @test_throws "needs the incubation period" serial_interval(bare, infector, infectee)
    end

    @testset "NoGenerationTime has no serial interval" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16)); attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws "needs a fixed generation-time distribution" serial_interval(
            spec, state.individuals[1], state.individuals[2]
        )
    end

    @testset "a per-case generation time has no general convolution" begin
        spec = ModelSpec(
            BranchingProcess(NegBin(2.5, 0.16), incubation_linked_generation_time());
            attributes = clinical
        )
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        @test_throws "not a per-case function" serial_interval(
            spec, state.individuals[1], state.individuals[2]
        )
    end

    @testset "a model with no clinical_presentation has no incubation period" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt))
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws "needs the incubation period" serial_interval(
            spec, state.individuals[1], state.individuals[2]
        )
    end

    @testset "a multi-window model has no single generation time" begin
        windows = (
            Infectiousness(NegBin(1.0, 0.5); kernel = gt),
            Infectiousness(NegBin(1.0, 0.5); kernel = gt, name = :other),
        )
        spec = ModelSpec(BranchingProcess(windows...); attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws ArgumentError serial_interval(
            spec, state.individuals[1], state.individuals[2]
        )
    end
end
