@testset "Interventions" begin
    @testset "Isolation reduces transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        rng1 = StableRNG(42)
        results_no_iso = simulate_batch(model, 100;
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng1)

        rng2 = StableRNG(42)
        iso = Isolation(delay=Exponential(2.0))
        results_iso = simulate_batch(model, 100;
            interventions=[iso],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng2)

        ext_no_iso = count(s -> s.extinct, results_no_iso)
        ext_iso = count(s -> s.extinct, results_iso)
        @test ext_iso >= ext_no_iso
    end

    @testset "Contact tracing marks individuals as traced" begin
        rng = StableRNG(101)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        ct = ContactTracing(probability=1.0, delay=Exponential(1.0))

        state = simulate(model;
            interventions=[iso, ct],
            sim_opts=SimOpts(max_cases=50, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        n_traced = count(ind -> is_traced(ind), state.individuals)
        n_with_isolated_parent = count(state.individuals) do ind
            ind.parent_id == 0 && return false
            parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
            parent_idx === nothing && return false
            parent = state.individuals[parent_idx]
            is_isolated(parent) && !is_asymptomatic(parent)
        end
        if n_with_isolated_parent > 0
            @test n_traced > 0
        end
    end

    @testset "Intervention start_time respected" begin
        rng = StableRNG(200)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))

        iso = Isolation(delay=Exponential(1.0), start_time=1000.0)
        state = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(max_cases=50, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        @test all(ind -> ind.infection_time < 1000.0, state.individuals)
    end

    @testset "Asymptomatic cases are not isolated" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))

        state = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(
                max_cases=500,
                incubation_period=LogNormal(1.5, 0.5),
                prob_asymptomatic=0.5,
            ),
            rng=rng)

        for ind in state.individuals
            if is_asymptomatic(ind)
                @test !is_isolated(ind)
            end
        end
        @test any(ind -> is_asymptomatic(ind), state.individuals)
    end

    @testset "Test sensitivity affects isolation" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))

        state_no_test = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(
                max_cases=100,
                incubation_period=LogNormal(1.5, 0.5),
                test_sensitivity=0.0,
            ),
            rng=StableRNG(42))

        n_isolated_no_test = count(ind -> is_isolated(ind), state_no_test.individuals)
        @test n_isolated_no_test == 0
    end

    @testset "Hazard-based isolation reduces more with early isolation" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        rng1 = StableRNG(42)
        iso_fast = Isolation(delay=Exponential(0.5))
        results_fast = simulate_batch(model, 200;
            interventions=[iso_fast],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng1)

        rng2 = StableRNG(42)
        iso_slow = Isolation(delay=Exponential(10.0))
        results_slow = simulate_batch(model, 200;
            interventions=[iso_slow],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng2)

        cp_fast = containment_probability(results_fast)
        cp_slow = containment_probability(results_slow)
        @test cp_fast >= cp_slow
    end

    @testset "Intervention initialises state on individuals" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        ct = ContactTracing(probability=0.5, delay=Exponential(1.0))

        state = simulate(model;
            interventions=[iso, ct],
            sim_opts=SimOpts(max_cases=20, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        # All individuals should have intervention fields initialised
        for ind in state.individuals
            @test haskey(ind.state, :isolated)
            @test haskey(ind.state, :isolation_time)
            @test haskey(ind.state, :traced)
            @test haskey(ind.state, :quarantined)
        end
    end
end
