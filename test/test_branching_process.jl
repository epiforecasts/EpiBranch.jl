@testset "Branching process" begin
    @testset "Subcritical outbreak goes extinct" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(), rng = rng)
        @test state.extinct
        @test state.cumulative_cases < 100
    end

    @testset "Supercritical outbreak grows" begin
        rng = StableRNG(123)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(max_cases = 500), rng = rng)
        @test state.cumulative_cases >= 500
    end

    @testset "Multiple index cases" begin
        rng = StableRNG(99)
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(n_initial = 5), rng = rng)
        index_chains = [ind.chain_id for ind in state.individuals if ind.parent_id == 0]
        @test length(unique(index_chains)) == 5
    end

    @testset "Generation tracking" begin
        rng = StableRNG(77)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(max_generations = 3), rng = rng)
        max_gen = maximum(ind.generation for ind in state.individuals)
        @test max_gen <= 3
    end

    @testset "Infection times increase across generations" begin
        rng = StableRNG(55)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(max_cases = 100), rng = rng)

        for ind in filter(is_infected, state.individuals)
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time > parent.infection_time
            end
        end
    end

    @testset "Incubation period sets onset times" begin
        rng = StableRNG(33)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        init_fn = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        state = simulate(model;
            attributes = init_fn,
            sim_opts = SimOpts(max_cases = 50),
            rng = rng)

        for ind in filter(is_infected, state.individuals)
            @test !isnan(onset_time(ind))
            @test onset_time(ind) >= ind.infection_time
        end
    end

    @testset "incubation_period accessor" begin
        ind = Individual(id = 1, infection_time = 2.0)
        @test isnan(incubation_period(ind))   # onset not set yet
        ind.state[:onset_time] = 5.5
        @test incubation_period(ind) ≈ 3.5
    end

    @testset "Generation time can depend on individual state" begin
        # An attributes function stores a per-individual latent and sets
        # onset from it; the generation_time function reads the same
        # latent, so onset and generation time come from one draw rather
        # than two independent ones.
        rng = StableRNG(7)
        init_fn = function (rng, ind)
            scale = 3.0 + rand(rng)
            ind.state[:gt_scale] = scale
            ind.state[:onset_time] = ind.infection_time + scale
        end
        gt_fn = ind -> Exponential(ind.state[:gt_scale])
        model = BranchingProcess(Poisson(1.2), gt_fn)
        state = simulate(model; attributes = init_fn,
            sim_opts = SimOpts(max_cases = 50), rng = rng)

        @test state.cumulative_cases > 0
        for ind in filter(is_infected, state.individuals)
            @test incubation_period(ind) ≈ ind.state[:gt_scale]
        end
    end

    @testset "Truncated generation time enforces minimum" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), truncated(Exponential(5.0), 3.0, Inf))
        state = simulate(model;
            condition = 20:500, sim_opts = SimOpts(max_cases = 500), rng = rng)

        for ind in filter(is_infected, state.individuals)
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time - parent.infection_time >= 3.0 - 1e-10
            end
        end
    end
end
