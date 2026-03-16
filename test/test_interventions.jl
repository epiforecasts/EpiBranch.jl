@testset "Interventions" begin
    @testset "Isolation reduces transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        # Without isolation
        rng1 = StableRNG(42)
        results_no_iso = simulate_batch(model, 100;
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng1)

        # With isolation
        rng2 = StableRNG(42)
        iso = Isolation(delay=Exponential(2.0))
        results_iso = simulate_batch(model, 100;
            interventions=[iso],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng2)

        # More outbreaks should go extinct with isolation
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

        # At least some individuals should be traced (those with isolated parents)
        n_traced = count(ind -> ind.traced, state.individuals)
        # With probability=1.0, all children of isolated symptomatic parents should be traced
        n_with_isolated_parent = count(state.individuals) do ind
            ind.parent_id == 0 && return false
            parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
            parent_idx === nothing && return false
            parent = state.individuals[parent_idx]
            parent.isolated && !parent.asymptomatic
        end
        if n_with_isolated_parent > 0
            @test n_traced > 0
        end
    end

    @testset "Intervention start_time respected" begin
        rng = StableRNG(200)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))

        # Isolation starts at time 1000 — should have no effect on early outbreak
        iso = Isolation(delay=Exponential(1.0), start_time=1000.0)
        state = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(max_cases=50, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        # All individuals should be before time 1000, so isolation should not have triggered
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

        # Asymptomatic individuals should never be isolated
        for ind in state.individuals
            if ind.asymptomatic
                @test !ind.isolated
            end
        end
        # Some individuals should be asymptomatic (with 50% prob and many cases)
        @test any(ind -> ind.asymptomatic, state.individuals)
    end

    @testset "Test sensitivity affects isolation" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))

        # With perfect test sensitivity
        state_perfect = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(
                max_cases=100,
                incubation_period=LogNormal(1.5, 0.5),
                test_sensitivity=1.0,
            ),
            rng=StableRNG(42))

        # With zero test sensitivity
        state_no_test = simulate(model;
            interventions=[iso],
            sim_opts=SimOpts(
                max_cases=100,
                incubation_period=LogNormal(1.5, 0.5),
                test_sensitivity=0.0,
            ),
            rng=StableRNG(42))

        # No one should be isolated with zero test sensitivity
        n_isolated_no_test = count(ind -> ind.isolated, state_no_test.individuals)
        @test n_isolated_no_test == 0
    end

    @testset "Hazard-based isolation reduces more with early isolation" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        # Fast isolation (small delay)
        rng1 = StableRNG(42)
        iso_fast = Isolation(delay=Exponential(0.5))
        results_fast = simulate_batch(model, 200;
            interventions=[iso_fast],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng1)

        # Slow isolation (large delay)
        rng2 = StableRNG(42)
        iso_slow = Isolation(delay=Exponential(10.0))
        results_slow = simulate_batch(model, 200;
            interventions=[iso_slow],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng2)

        # Faster isolation should lead to more containment
        cp_fast = containment_probability(results_fast)
        cp_slow = containment_probability(results_slow)
        @test cp_fast >= cp_slow
    end
end
