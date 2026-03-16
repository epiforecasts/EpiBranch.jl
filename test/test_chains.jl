using DataFrames

@testset "Chain statistics" begin
    @testset "Single chain" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        state = simulate(model; sim_opts=SimOpts(n_initial=1), rng=rng)

        df = chain_statistics(state)
        @test df isa DataFrame
        @test nrow(df) == 1
        @test df.chain_id[1] == 1
        @test df.size[1] == state.cumulative_cases
    end

    @testset "Multiple chains" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; sim_opts=SimOpts(n_initial=5), rng=rng)

        df = chain_statistics(state)
        @test nrow(df) == 5
        @test sum(df.size) == state.cumulative_cases
    end

    @testset "Chain length matches max generation" begin
        rng = StableRNG(77)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; sim_opts=SimOpts(max_cases=50), rng=rng)

        df = chain_statistics(state)
        max_gen = maximum(ind.generation for ind in state.individuals)
        @test maximum(df.length) == max_gen
    end

    @testset "Batch chain statistics" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        states = simulate_batch(model, 10; rng=rng)

        df = chain_statistics(states)
        @test "sim_id" in names(df)
        @test "chain_id" in names(df)
        @test "size" in names(df)
        @test "length" in names(df)
        @test length(unique(df.sim_id)) == 10
    end
end
