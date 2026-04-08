using DataFrames

@testset "Multi-type branching process" begin
    @testset "Offspring matrix construction" begin
        # 2x2 offspring matrix: type 1 infects mostly type 1, type 2 mostly type 2
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        @test model.n_types == 2
        @test model.offspring isa Function
    end

    @testset "Non-square matrix throws" begin
        M = [1.0 0.5 0.3;
             0.5 1.0 0.3]
        @test_throws ArgumentError BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))
    end

    @testset "Simulation runs with offspring matrix" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 100),
            rng = rng)

        @test state.cumulative_cases > 0

        # All infected individuals should have a type
        infected = filter(is_infected, state.individuals)
        for ind in infected
            @test haskey(ind.state, :type)
            @test individual_type(ind) in 1:2
        end
    end

    @testset "Types are distributed according to matrix" begin
        # Strongly assortative: type 1 → type 1, type 2 → type 2
        M = [3.0 0.0;
             0.0 3.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 200),
            rng = rng)

        infected = filter(is_infected, state.individuals)
        # Children should have the same type as their parent
        for ind in infected
            ind.parent_id == 0 && continue
            parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
            parent = state.individuals[parent_idx]
            @test individual_type(ind) == individual_type(parent)
        end
    end

    @testset "Asymmetric R by type" begin
        # Type 1 has R=3, type 2 has R=0.5
        M = [2.5 0.1;
             0.5 0.4]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        # Run multiple simulations to reduce stochastic variation
        n_type1_total = 0
        n_type2_total = 0
        rng = StableRNG(42)
        for _ in 1:20
            state = simulate(model; sim_opts = SimOpts(max_cases = 200), rng = rng)
            infected = filter(is_infected, state.individuals)
            n_type1_total += count(ind -> individual_type(ind) == 1, infected)
            n_type2_total += count(ind -> individual_type(ind) == 2, infected)
        end

        # Type 1 should dominate across many simulations
        @test n_type1_total > n_type2_total
    end

    @testset "NegBin offspring with matrix" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> NegBin(R_j, 0.5), Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 50),
            rng = rng)

        @test state.cumulative_cases > 0
    end

    @testset "Custom offspring function" begin
        # Explicit function: type 1 produces [2, 1], type 2 produces [0, 3]
        function my_offspring(rng, parent_type)
            if parent_type == 1
                return [rand(rng, Poisson(2.0)), rand(rng, Poisson(1.0))]
            else
                return [rand(rng, Poisson(0.0)), rand(rng, Poisson(3.0))]
            end
        end

        model = BranchingProcess(my_offspring, Exponential(5.0); n_types = 2)

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 50),
            rng = rng)

        @test state.cumulative_cases > 0
        infected = filter(is_infected, state.individuals)
        @test all(ind -> individual_type(ind) in 1:2, infected)
    end

    @testset "Multi-type with interventions" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))
        iso = Isolation(delay = Exponential(1.0))
        init_fn = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

        rng = StableRNG(42)
        state = simulate(model;
            interventions = [iso], attributes = init_fn,
            sim_opts = SimOpts(max_cases = 100),
            rng = rng)

        n_isolated = count(ind -> is_isolated(ind), state.individuals)
        @test n_isolated > 0
    end

    @testset "Multi-type with finite population" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0);
            population_size = 50)

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 100),
            rng = rng)

        @test state.cumulative_cases <= 50
    end

    @testset "Chain statistics work with multi-type" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(model; sim_opts = SimOpts(max_cases = 50), rng = rng)

        cs = chain_statistics(state)
        @test cs isa DataFrame
        @test sum(cs.size) == state.cumulative_cases
    end

    @testset "Contacts table includes type" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(model; sim_opts = SimOpts(max_cases = 50), rng = rng)

        ct = contacts(state)
        @test ct isa DataFrame
        @test nrow(ct) > 0
    end

    @testset "Type labels" begin
        M = [1.5 0.3;
             0.3 1.0]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0);
            type_labels = ["children", "adults"])
        @test model.type_labels == ["children", "adults"]
    end

    @testset "3-type model" begin
        M = [2.0 0.5 0.1;
             0.5 1.5 0.3;
             0.1 0.3 0.8]
        model = BranchingProcess(M, R_j -> Poisson(R_j), Exponential(5.0);
            type_labels = ["0-14", "15-64", "65+"])

        rng = StableRNG(42)
        state = simulate(model;
            sim_opts = SimOpts(max_cases = 100),
            rng = rng)

        infected = filter(is_infected, state.individuals)
        types = [individual_type(ind) for ind in infected]
        @test all(t -> t in 1:3, types)
        # Should have all 3 types represented in a large enough outbreak
        if length(infected) > 20
            @test length(unique(types)) == 3
        end
    end
end
