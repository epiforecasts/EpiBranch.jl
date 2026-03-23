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
        iso = Isolation(delay=Exponential(1.0), test_sensitivity=0.0)

        state = simulate(model;
            interventions=[iso], attributes=clinical,
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

    @testset "Ring vaccination" begin
        @testset "Leaky mode reduces transmission" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Isolation(delay=Exponential(1.0))
            ct = ContactTracing(probability=1.0, delay=Exponential(0.5))

            rng1 = StableRNG(42)
            rv = RingVaccination(efficacy=0.9, mode=:leaky)
            results_vacc = simulate_batch(model, 100;
                interventions=[iso, ct, rv], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng1)

            rng2 = StableRNG(42)
            results_no_vacc = simulate_batch(model, 100;
                interventions=[iso, ct], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng2)

            @test containment_probability(results_vacc) >= containment_probability(results_no_vacc)
        end

        @testset "All-or-nothing mode" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Isolation(delay=Exponential(1.0))
            ct = ContactTracing(probability=1.0, delay=Exponential(0.5))
            rv = RingVaccination(efficacy=0.8, mode=:all_or_nothing)

            state = simulate(model;
                interventions=[iso, ct, rv], attributes=clinical,
                sim_opts=SimOpts(max_cases=100), rng=StableRNG(42))

            n_vaccinated = count(is_vaccinated, state.individuals)
            @test n_vaccinated > 0

            for ind in state.individuals
                @test haskey(ind.state, :vaccinated)
                @test haskey(ind.state, :vaccination_time)
            end
        end

        @testset "Delay to immunity" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Isolation(delay=Exponential(1.0))
            ct = ContactTracing(probability=1.0, delay=Exponential(0.5))

            rng1 = StableRNG(42)
            rv_instant = RingVaccination(efficacy=0.9, delay_to_immunity=0.0)
            results_instant = simulate_batch(model, 100;
                interventions=[iso, ct, rv_instant], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng1)

            rng2 = StableRNG(42)
            rv_delayed = RingVaccination(efficacy=0.9, delay_to_immunity=14.0)
            results_delayed = simulate_batch(model, 100;
                interventions=[iso, ct, rv_delayed], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng2)

            # Instant immunity should contain better than delayed
            @test containment_probability(results_instant) >= containment_probability(results_delayed)
        end
    end

    @testset "Contact tracing without quarantine" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay=Exponential(1.0))
        ct = ContactTracing(probability=1.0, delay=Exponential(1.0),
            quarantine_on_trace=false)

        state = simulate(model;
            interventions=[iso, ct], attributes=clinical,
            sim_opts=SimOpts(max_cases=50), rng=rng)

        n_traced = count(is_traced, state.individuals)
        if n_traced > 0
            # Traced contacts should not be quarantined
            traced = filter(is_traced, state.individuals)
            @test !any(is_quarantined, traced)
        end
    end

    @testset "Scheduled interventions" begin
        @testset "start_time delays activation" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))

            # Compare scheduled (late start) vs always-on — scheduled should contain less
            rng1 = StableRNG(42)
            iso_late = Scheduled(Isolation(delay=Exponential(1.0)); start_time=20.0)
            results_late = simulate_batch(model, 100;
                interventions=[iso_late], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng1)

            rng2 = StableRNG(42)
            iso_always = Isolation(delay=Exponential(1.0))
            results_always = simulate_batch(model, 100;
                interventions=[iso_always], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng2)

            @test containment_probability(results_always) >= containment_probability(results_late)

            # Fields should still be initialised on all individuals
            state = simulate(model;
                interventions=[iso_late], attributes=clinical,
                sim_opts=SimOpts(max_cases=50), rng=StableRNG(99))
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
            end
        end

        @testset "start_after_cases delays activation" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Scheduled(Isolation(delay=Exponential(0.5)); start_after_cases=20)

            rng1 = StableRNG(42)
            results_scheduled = simulate_batch(model, 100;
                interventions=[iso], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng1)

            # Compare with always-on isolation — scheduled should contain less
            rng2 = StableRNG(42)
            iso_always = Isolation(delay=Exponential(0.5))
            results_always = simulate_batch(model, 100;
                interventions=[iso_always], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=rng2)

            @test containment_probability(results_always) >= containment_probability(results_scheduled)
        end

        @testset "custom predicate" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Scheduled(Isolation(delay=Exponential(1.0)),
                state -> state.current_generation >= 3)

            state = simulate(model;
                interventions=[iso], attributes=clinical,
                sim_opts=SimOpts(max_cases=100), rng=StableRNG(42))

            # Verify the intervention ran without errors and fields exist
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
            end
        end

        @testset "end_time deactivates" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            # Active only in a short window
            iso = Scheduled(Isolation(delay=Exponential(0.5));
                start_time=5.0, end_time=10.0)

            state = simulate(model;
                interventions=[iso], attributes=clinical,
                sim_opts=SimOpts(max_cases=100), rng=StableRNG(42))

            # Late individuals should not be isolated
            late = filter(i -> is_infected(i) && i.infection_time > 15.0, state.individuals)
            if !isempty(late)
                @test !any(is_isolated, late)
            end
        end

        @testset "mixed Scheduled and always-on" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Isolation(delay=Exponential(1.0))
            ct = Scheduled(ContactTracing(probability=0.5, delay=Exponential(1.0));
                start_after_cases=10)

            state = simulate(model;
                interventions=[iso, ct], attributes=clinical,
                sim_opts=SimOpts(max_cases=50), rng=StableRNG(42))

            # All individuals should have both isolation and tracing fields
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
                @test haskey(ind.state, :traced)
            end
        end

        @testset "filters on action time not infection time" begin
            model = BranchingProcess(Poisson(3.0), Exponential(5.0))
            iso = Scheduled(Isolation(delay=Exponential(0.1)); start_time=15.0)

            state = simulate(model;
                interventions=[iso], attributes=clinical,
                sim_opts=SimOpts(max_cases=200), rng=StableRNG(42))

            # No individual should be isolated with isolation_time < 15.0
            for ind in state.individuals
                if is_isolated(ind)
                    @test isolation_time(ind) >= 15.0
                end
            end
        end

        @testset "requires at least one condition" begin
            iso = Isolation(delay=Exponential(1.0))
            @test_throws ErrorException Scheduled(iso)
        end
    end
end
