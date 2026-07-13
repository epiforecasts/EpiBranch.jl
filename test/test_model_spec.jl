@testset "ModelSpec" begin
    # A ModelSpec wrapping a process, with its forcing layers on the spec,
    # must run identically to the same process carrying those layers itself —
    # same seed, same outbreak. This is the round trip that lets the process
    # stay a pure kernel while the spec composes the forcings around it.

    @testset "wraps a process faithfully" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        s1 = simulate(bp; max_cases = 500, rng = StableRNG(1))
        s2 = simulate(ModelSpec(bp); max_cases = 500, rng = StableRNG(1))
        @test s2.cumulative_cases == s1.cumulative_cases
    end

    @testset "forcings on the spec match forcings on the process" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        attr = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

        embedded = BranchingProcess(Poisson(2.0), Exponential(5.0);
            interventions = [iso], attributes = attr)
        spec = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso], attributes = attr)

        s1 = simulate(embedded; max_cases = 500, rng = StableRNG(7))
        s2 = simulate(spec; max_cases = 500, rng = StableRNG(7))
        @test s2.cumulative_cases == s1.cumulative_cases
    end

    @testset "keywords default to the wrapped process" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0); interventions = [iso])
        # ModelSpec(bp) inherits the process's own layers unchanged.
        @test length(EpiBranch.interventions(ModelSpec(bp))) == 1
        # ...and a keyword overrides that layer.
        @test isempty(EpiBranch.interventions(ModelSpec(bp; interventions = [])))
    end
end
