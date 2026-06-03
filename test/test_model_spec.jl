@testset "ModelSpec" begin
    @testset "Construction and defaults" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        spec = ModelSpec(process = bp)

        @test spec.process === bp
        @test isempty(spec.interventions)
        @test isempty(spec.transitions)
        @test spec.attributes isa NoAttributes
        @test spec.observation isa NoObservation
        @test spec.sim_opts isa SimOpts
    end

    @testset "simulate(spec) ≡ simulate(model; kwargs)" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        iso = Isolation(delay = Exponential(2.0))
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        opts = SimOpts(max_cases = 200)

        spec = ModelSpec(process = bp, interventions = [iso],
            attributes = clinical, sim_opts = opts)

        rng_a = StableRNG(7)
        rng_b = StableRNG(7)
        state_spec = simulate(spec; rng = rng_a)
        state_kwarg = simulate(bp; interventions = [iso],
            attributes = clinical, sim_opts = opts, rng = rng_b)

        @test state_spec.cumulative_cases == state_kwarg.cumulative_cases
        @test state_spec.extinct == state_kwarg.extinct
    end

    @testset "simulate(spec, n) ≡ simulate_batch(model, n; kwargs)" begin
        bp = BranchingProcess(Poisson(0.6), Exponential(5.0))
        spec = ModelSpec(process = bp)

        rng_a = StableRNG(11)
        rng_b = StableRNG(11)
        states_spec = simulate(spec, 25; rng = rng_a)
        states_kwarg = simulate_batch(bp, 25; rng = rng_b)

        @test length(states_spec) == 25
        @test [s.cumulative_cases for s in states_spec] ==
              [s.cumulative_cases for s in states_kwarg]
    end

    @testset "loglikelihood(data, spec) ≡ loglikelihood(data, model; kwargs)" begin
        bp = BranchingProcess(Poisson(0.5))
        spec_no_obs = ModelSpec(process = bp)
        data = ChainSizes([1, 2, 1, 3, 1, 5, 2])

        @test loglikelihood(data, spec_no_obs) ≈ loglikelihood(data, Poisson(0.5))

        # OffspringCounts: pulls offspring from the spec's process.
        oc = OffspringCounts([0, 1, 2, 0, 3, 1, 0])
        @test loglikelihood(oc, spec_no_obs) ≈ loglikelihood(oc, Poisson(0.5))
    end

    @testset "Observation slot folds into Observed wrapper" begin
        bp = BranchingProcess(Poisson(0.5))
        obs = PerCaseObservation(; detection_prob = 0.7)
        spec = ModelSpec(process = bp, observation = obs)
        data = ChainSizes([1, 2, 1, 3, 1])

        @test loglikelihood(data, spec) ≈ loglikelihood(data, Observed(bp, obs))
    end
end
