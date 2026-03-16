@testset "Density-dependent model" begin
    @testset "Finite population limits outbreak" begin
        rng = StableRNG(42)
        model = DensityDependent(Poisson(3.0), Exponential(5.0), 100)
        state = simulate(model; sim_opts=SimOpts(max_cases=200), rng=rng)

        # Should not exceed population size
        @test state.cumulative_cases <= 100
    end

    @testset "Large population behaves like BP" begin
        # With very large population, density-dependent should approximate standard BP
        rng1 = StableRNG(42)
        model_dd = DensityDependent(Poisson(0.5), Exponential(5.0), 1_000_000)
        state_dd = simulate(model_dd; sim_opts=SimOpts(), rng=rng1)

        rng2 = StableRNG(42)
        model_bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state_bp = simulate(model_bp; sim_opts=SimOpts(), rng=rng2)

        # Both should go extinct for subcritical R
        @test state_dd.extinct
        @test state_bp.extinct
    end

    @testset "Interventions work with density-dependent" begin
        rng = StableRNG(42)
        model = DensityDependent(Poisson(3.0), Exponential(5.0), 500)
        iso = Isolation(delay=Exponential(1.0))

        state = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        # Should have some isolated individuals
        n_isolated = count(ind -> ind.isolated, state.individuals)
        @test n_isolated > 0
    end

    @testset "Generation tracking works" begin
        rng = StableRNG(55)
        model = DensityDependent(Poisson(2.0), Exponential(5.0), 200)
        state = simulate(model; sim_opts=SimOpts(max_generations=3), rng=rng)

        max_gen = maximum(ind.generation for ind in state.individuals)
        @test max_gen <= 3
    end

    @testset "Infection times increase" begin
        rng = StableRNG(77)
        model = DensityDependent(Poisson(2.0), Exponential(5.0), 500)
        state = simulate(model; sim_opts=SimOpts(max_cases=50), rng=rng)

        for ind in state.individuals
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time > parent.infection_time
            end
        end
    end
end
