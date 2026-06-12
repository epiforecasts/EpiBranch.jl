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
        m = with_observation(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        state = simulate(with_attributes(m, attrs); max_cases = 100, rng = rng)
        for ind in state.individuals
            expected = ind.state[:age] >= 50
            @test ind.state[:reported] == expected
        end
    end

    @testset "Function delay varies per individual" begin
        # Per-individual delay drawn from a state-dependent distribution.
        # Default anchor is :onset_time.
        attrs = compose(
            clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0),
            demographics(age_distribution = Uniform(0, 90)))
        obs = PerCaseObservation(
            delay = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 7.0)
        rng = StableRNG(23)
        m = with_observation(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        state = simulate(with_attributes(m, attrs); max_cases = 100, rng = rng)
        for ind in state.individuals
            expected_lag = ind.state[:age] >= 50 ? 1.0 : 7.0
            @test ind.state[:report_time] ≈ ind.state[:onset_time] + expected_lag
        end
    end

    @testset "from anchor: infection time fallback for asymptomatic" begin
        # Asymptomatic cases have NaN onset_time; report_time should fall
        # back to infection_time rather than NaN.
        attrs = compose(
            clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 1.0))
        obs = PerCaseObservation(delay = 2.0)
        rng = StableRNG(99)
        m = with_observation(BranchingProcess(Poisson(1.5), Exponential(5.0)), obs)
        state = simulate(with_attributes(m, attrs); max_cases = 50, rng = rng)
        for ind in state.individuals
            @test !isnan(ind.state[:report_time])
            @test ind.state[:report_time] ≈ ind.infection_time + 2.0
        end
    end

    @testset "from = infection_time anchors on infection" begin
        attrs = compose(
            clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0))
        obs = PerCaseObservation(delay = 3.0,
            from = ind -> ind.infection_time)
        rng = StableRNG(7)
        m = with_observation(BranchingProcess(Poisson(1.5), Exponential(5.0)), obs)
        state = simulate(with_attributes(m, attrs); max_cases = 50, rng = rng)
        for ind in state.individuals
            @test ind.state[:report_time] ≈ ind.infection_time + 3.0
        end
    end

    @testset "Distribution detection_prob varies per individual" begin
        # Beta-distributed reporting probability: aggregate over many runs.
        obs = PerCaseObservation(detection_prob = Beta(2.0, 2.0))
        m = with_observation(BranchingProcess(Poisson(2.0), Exponential(5.0)), obs)
        rng = StableRNG(31)
        states = simulate(m, 50; max_cases = 50,
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
        @test EpiBranch.scalar_detection_prob(scalar) == 0.7

        fn = PerCaseObservation(
            detection_prob = (rng, ind) -> 0.5)
        @test_throws ArgumentError EpiBranch.scalar_detection_prob(fn)

        dist = PerCaseObservation(detection_prob = Beta(2.0, 2.0))
        @test_throws ArgumentError EpiBranch.scalar_detection_prob(dist)
    end

    @testset "Closed-form chain_size_distribution refuses non-scalar" begin
        base = BranchingProcess(Poisson(0.5), Exponential(5.0))
        m = with_observation(base, PerCaseObservation(
            detection_prob = (rng, ind) -> 0.5))
        @test_throws ArgumentError chain_size_distribution(m)
    end
end
