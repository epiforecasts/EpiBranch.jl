@testset "Finite population (density-dependent)" begin
    @testset "Finite population limits outbreak" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0); population_size = 100)
        state = simulate(model; sim_opts = SimOpts(max_cases = 200), rng = rng)
        @test state.cumulative_cases <= 100
    end

    @testset "Large population behaves like infinite BP" begin
        rng1 = StableRNG(42)
        model_finite = BranchingProcess(Poisson(0.5), Exponential(5.0);
            population_size = 1_000_000)
        state_finite = simulate(model_finite; sim_opts = SimOpts(), rng = rng1)

        rng2 = StableRNG(42)
        model_inf = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state_inf = simulate(model_inf; sim_opts = SimOpts(), rng = rng2)

        @test state_finite.extinct
        @test state_inf.extinct
    end

    @testset "Interventions work with finite population" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0); population_size = 500)
        iso = Isolation(delay = Exponential(1.0))
        init_fn = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

        state = simulate(model;
            interventions = [iso], attributes = init_fn,
            sim_opts = SimOpts(max_cases = 200), rng = rng)

        @test count(is_isolated, state.individuals) > 0
    end

    @testset "Generation tracking works" begin
        rng = StableRNG(55)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0); population_size = 200)
        state = simulate(model; sim_opts = SimOpts(max_generations = 3), rng = rng)
        max_gen = maximum(ind.generation for ind in state.individuals)
        @test max_gen <= 3
    end

    @testset "Infection times increase" begin
        rng = StableRNG(77)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0); population_size = 500)
        state = simulate(model; sim_opts = SimOpts(max_cases = 50), rng = rng)

        for ind in filter(is_infected, state.individuals)
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time > parent.infection_time
            end
        end
    end
end
