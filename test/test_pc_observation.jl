@testset "PerCaseObservation trait seams" begin
    @testset "Default keyword constructor reproduces previous behaviour" begin
        o = PerCaseObservation()
        @test o.detection_prob == 1.0
        @test o.delay == Dirac(0.0)
    end

    @testset "Scalar detection_prob is validated" begin
        @test_throws ArgumentError PerCaseObservation(detection_prob = 0.0)
        @test_throws ArgumentError PerCaseObservation(detection_prob = 1.5)
        @test_throws ArgumentError PerCaseObservation(detection_prob = -0.1)
    end

    @testset "Function detection_prob varies per individual" begin
        # Age-conditional detection: 50+ always reported, under-50 never.
        attrs = compose(
            clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0),
            demographics(age_distribution = Uniform(0, 90)))
        obs = PerCaseObservation(
            detection_prob = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0)
        rng = StableRNG(11)
        m = Observed(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        state = simulate(m; attributes = attrs,
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        for ind in state.individuals
            expected = ind.state[:age] >= 50
            @test ind.state[:reported] == expected
        end
    end

    @testset "Function delay varies per individual" begin
        # Per-individual delay drawn from a state-dependent distribution.
        attrs = compose(
            clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0),
            demographics(age_distribution = Uniform(0, 90)))
        obs = PerCaseObservation(
            delay = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 7.0)
        rng = StableRNG(23)
        m = Observed(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        state = simulate(m; attributes = attrs,
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        for ind in state.individuals
            expected_lag = ind.state[:age] >= 50 ? 1.0 : 7.0
            @test ind.state[:report_time] ≈ ind.infection_time + expected_lag
        end
    end

    @testset "Distribution detection_prob varies per individual" begin
        # Beta-distributed reporting probability: aggregate over many runs.
        obs = PerCaseObservation(detection_prob = Beta(2.0, 2.0))
        m = Observed(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        rng = StableRNG(31)
        states = simulate_batch(m, 50; sim_opts = SimOpts(max_cases = 50),
            rng = rng)
        reported = [ind.state[:reported]
                    for s in states for ind in s.individuals]
        @test !isempty(reported)
        frac = count(reported) / length(reported)
        # Beta(2,2) has mean 0.5; with hundreds of draws we should be in (0.3, 0.7).
        @test 0.3 < frac < 0.7
    end

    @testset "scalar_detection_prob rejects non-scalar fields" begin
        scalar = PerCaseObservation(detection_prob = 0.7)
        @test EpiBranch.EpiBranchObservation.scalar_detection_prob(scalar) == 0.7

        fn = PerCaseObservation(
            detection_prob = (rng, ind) -> 0.5)
        @test_throws ArgumentError EpiBranch.EpiBranchObservation.scalar_detection_prob(fn)

        dist = PerCaseObservation(detection_prob = Beta(2.0, 2.0))
        @test_throws ArgumentError EpiBranch.EpiBranchObservation.scalar_detection_prob(dist)
    end

    @testset "Closed-form chain_size_distribution refuses non-scalar" begin
        base = BranchingProcess(Poisson(0.5), Exponential(5.0))
        m = Observed(base, PerCaseObservation(
            detection_prob = (rng, ind) -> 0.5))
        @test_throws ArgumentError chain_size_distribution(m)
    end
end
