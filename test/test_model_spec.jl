@testset "ModelSpec" begin
    # A ModelSpec composes the modelling layers (progression, interventions,
    # attributes, observation) around a pure transmission process. The process
    # carries none of them; the spec is the single place they are attached.

    @testset "wraps a process faithfully" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        s1 = simulate(bp; max_cases = 500, rng = StableRNG(1))
        s2 = simulate(ModelSpec(bp); max_cases = 500, rng = StableRNG(1))
        @test s2.cumulative_cases == s1.cumulative_cases
    end

    @testset "layers on the spec are applied" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        attr = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))

        spec = ModelSpec(bp; interventions = [iso], attributes = attr)
        @test EpiBranch.interventions(spec) == [iso]
        @test EpiBranch.attributes(spec) === attr

        # The isolation layer takes effect: some cases are isolated.
        state = simulate(spec; max_cases = 500, rng = StableRNG(7))
        @test count(is_isolated, state.individuals) > 0
    end

    @testset "keywords set the layers; default to none" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        # A bare process carries no layers, so the spec defaults to none.
        @test isempty(EpiBranch.interventions(ModelSpec(bp)))
        # ...and a keyword attaches one.
        @test length(EpiBranch.interventions(ModelSpec(bp; interventions = [iso]))) == 1
    end
end
