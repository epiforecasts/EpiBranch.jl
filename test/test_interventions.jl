@testset "Interventions" begin
    clinical = clinical_presentation(
        incubation_period=LogNormal(1.5, 0.5),
        prob_asymptomatic=0.0,
    )

    @testset "Isolation reduces transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        rng1 = StableRNG(42)
        results_no_iso = simulate_batch(model, 100;
            attributes=clinical, sim_opts=SimOpts(max_cases=200), rng=rng1)

        rng2 = StableRNG(42)
        iso = Isolation(delay=Exponential(2.0))
        results_iso = simulate_batch(model, 100;
            interventions=[iso], attributes=clinical,
            sim_opts=SimOpts(max_cases=200), rng=rng2)

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
            interventions=[iso, ct], attributes=clinical,
            sim_opts=SimOpts(max_cases=50), rng=rng)

        n_traced = count(is_traced, state.individuals)
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
            interventions=[iso], attributes=clinical,
            sim_opts=SimOpts(max_cases=50), rng=rng)

        @test all(ind -> ind.infection_time < 1000.0, state.individuals)
    end

    @testset "Asymptomatic cases are not isolated" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        clinical_asymp = clinical_presentation(
            incubation_period=LogNormal(1.5, 0.5),
            prob_asymptomatic=0.5,
        )

        state = simulate(model;
            interventions=[iso], attributes=clinical_asymp,
            sim_opts=SimOpts(max_cases=500), rng=rng)

        for ind in state.individuals
            if is_asymptomatic(ind)
                @test !is_isolated(ind)
            end
        end
        @test any(is_asymptomatic, state.individuals)
    end

    @testset "Test sensitivity affects isolation" begin
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        clinical_no_test = compose(
            clinical_presentation(incubation_period=LogNormal(1.5, 0.5)),
            testing(sensitivity=0.0),
        )

        state = simulate(model;
            interventions=[iso], attributes=clinical_no_test,
            sim_opts=SimOpts(max_cases=100), rng=StableRNG(42))

        n_isolated = count(is_isolated, state.individuals)
        @test n_isolated == 0
    end

    @testset "Hazard-based isolation reduces more with early isolation" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        rng1 = StableRNG(42)
        iso_fast = Isolation(delay=Exponential(0.5))
        results_fast = simulate_batch(model, 200;
            interventions=[iso_fast], attributes=clinical,
            sim_opts=SimOpts(max_cases=200), rng=rng1)

        rng2 = StableRNG(42)
        iso_slow = Isolation(delay=Exponential(10.0))
        results_slow = simulate_batch(model, 200;
            interventions=[iso_slow], attributes=clinical,
            sim_opts=SimOpts(max_cases=200), rng=rng2)

        @test containment_probability(results_fast) >= containment_probability(results_slow)
    end

    @testset "Intervention initialises state on individuals" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        ct = ContactTracing(probability=0.5, delay=Exponential(1.0))

        state = simulate(model;
            interventions=[iso, ct], attributes=clinical,
            sim_opts=SimOpts(max_cases=20), rng=rng)

        for ind in state.individuals
            @test haskey(ind.state, :isolated)
            @test haskey(ind.state, :traced)
        end
    end

    @testset "Missing init gives helpful error" begin
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))

        @test_throws ErrorException simulate(model;
            interventions=[iso], sim_opts=SimOpts(max_cases=10), rng=StableRNG(42))
    end

    @testset "compose works" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        init_fn = compose(
            clinical_presentation(incubation_period=LogNormal(1.5, 0.5)),
            demographics(age_distribution=Normal(40, 15)),
        )

        state = simulate(model;
            attributes=init_fn, sim_opts=SimOpts(max_cases=50), rng=rng)

        ind = state.individuals[1]
        @test haskey(ind.state, :onset_time)
        @test haskey(ind.state, :age)
        @test haskey(ind.state, :sex)
    end
end
