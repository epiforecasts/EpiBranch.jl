@testset "Interventions" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Isolation reduces transmission" begin
        rng1 = StableRNG(42)
        results_no_iso = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0); attributes = clinical),
            100; max_cases = 200, rng = rng1)

        rng2 = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
        results_iso = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0);
                interventions = [iso], attributes = clinical),
            100; max_cases = 200, rng = rng2)

        ext_no_iso = count(s -> s.extinct, results_no_iso)
        ext_iso = count(s -> s.extinct, results_iso)
        @test ext_iso >= ext_no_iso
    end

    @testset "Contact tracing marks individuals as traced" begin
        rng = StableRNG(101)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(1.0))

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso, ct], attributes = clinical);
            max_cases = 50, rng = rng)

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

    @testset "Scheduled start_time respected" begin
        rng = StableRNG(200)
        iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(1.0)); start_time = 1000.0)

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso], attributes = clinical);
            max_cases = 50, rng = rng)

        @test all(ind -> ind.infection_time < 1000.0, state.individuals)
    end

    @testset "Asymptomatic cases are not isolated" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        clinical_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5),
            prob_asymptomatic = 0.5
        )

        state = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0);
                interventions = [iso], attributes = clinical_asymp);
            max_cases = 500, rng = rng)

        for ind in state.individuals
            if is_asymptomatic(ind)
                @test !is_isolated(ind)
            end
        end
        @test any(is_asymptomatic, state.individuals)
    end

    @testset "Test sensitivity affects isolation" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), test_sensitivity = 0.0)

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso], attributes = clinical);
            max_cases = 100, rng = StableRNG(42))

        n_isolated = count(is_isolated, state.individuals)
        @test n_isolated == 0
    end

    @testset "Hazard-based isolation reduces more with early isolation" begin
        rng1 = StableRNG(42)
        iso_fast = Isolation(onset_to_isolation_delay = Exponential(0.5))
        results_fast = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0);
                interventions = [iso_fast], attributes = clinical),
            200; max_cases = 200, rng = rng1)

        rng2 = StableRNG(42)
        iso_slow = Isolation(onset_to_isolation_delay = Exponential(10.0))
        results_slow = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0);
                interventions = [iso_slow], attributes = clinical),
            200; max_cases = 200, rng = rng2)

        @test containment_probability(results_fast) >= containment_probability(results_slow)
    end

    @testset "Intervention initialises state on individuals" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(1.0))

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso, ct], attributes = clinical);
            max_cases = 20, rng = rng)

        for ind in state.individuals
            @test haskey(ind.state, :isolated)
            @test haskey(ind.state, :traced)
        end
    end

    @testset "Missing init gives helpful error" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))

        @test_throws ErrorException simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0); interventions = [iso]);
            max_cases = 10, rng = StableRNG(42))
    end

    @testset "compose works" begin
        rng = StableRNG(42)
        init_fn = compose(
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Normal(40, 15))
        )

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0); attributes = init_fn);
            max_cases = 50, rng = rng)

        ind = state.individuals[1]
        @test haskey(ind.state, :onset_time)
        @test haskey(ind.state, :age)
        @test haskey(ind.state, :sex)
    end

    @testset "Ring vaccination" begin
        @testset "Leaky mode reduces transmission" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))

            rng1 = StableRNG(42)
            rv = RingVaccination(efficacy = 0.9, mode = LeakyMode())
            results_vacc = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            results_no_vacc = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct], attributes = clinical),
                100; max_cases = 200, rng = rng2)

            @test containment_probability(results_vacc) >=
                  containment_probability(results_no_vacc)
        end

        @testset "All-or-nothing mode" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            rv = RingVaccination(efficacy = 0.8, mode = AllOrNothingMode())

            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))

            n_vaccinated = count(is_vaccinated, state.individuals)
            @test n_vaccinated > 0

            for ind in state.individuals
                @test haskey(ind.state, :vaccinated)
                @test haskey(ind.state, :vaccination_time)
            end
        end

        @testset "Delay to immunity" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))

            rng1 = StableRNG(42)
            rv_instant = RingVaccination(efficacy = 0.9, delay_to_immunity = 0.0)
            results_instant = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_instant], attributes = clinical),
                200; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            rv_delayed = RingVaccination(efficacy = 0.9, delay_to_immunity = 14.0)
            results_delayed = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_delayed], attributes = clinical),
                200; max_cases = 200, rng = rng2)

            # Instant immunity should contain at least as well as delayed
            @test containment_probability(results_instant) >=
                  containment_probability(results_delayed) - 0.05
        end

        @testset "Coverage thins vaccinations" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))

            # Coverage = 0 means nobody gets vaccinated, even though traced.
            rv_zero = RingVaccination(efficacy = 0.9, coverage = 0.0)
            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_zero], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))
            @test count(is_vaccinated, state.individuals) == 0
            @test count(is_traced, state.individuals) > 0

            # Coverage = 1 reproduces the previous behaviour: every eligible
            # traced contact is vaccinated.
            rv_full = RingVaccination(efficacy = 0.9, coverage = 1.0)
            state_full = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_full], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))
            n_vacc_full = count(is_vaccinated, state_full.individuals)
            @test n_vacc_full > 0

            # Partial coverage gives strictly fewer vaccinations than full
            # coverage (over enough simulations).
            rv_partial = RingVaccination(efficacy = 0.9, coverage = 0.3)
            n_vacc_partial = sum(
                count(is_vaccinated, s.individuals)
            for s in simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_partial], attributes = clinical),
                30; max_cases = 100, rng = StableRNG(7)))
            n_vacc_full_batch = sum(
                count(is_vaccinated, s.individuals)
            for s in simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_full], attributes = clinical),
                30; max_cases = 100, rng = StableRNG(7)))
            @test n_vacc_partial < n_vacc_full_batch
        end

        @testset "Coverage accepts a function" begin
            # Age-conditional coverage: 50+ always vaccinated, under-50 never.
            attrs = compose(clinical,
                demographics(age_distribution = Uniform(0, 90)))
            iso = Isolation(onset_to_isolation_delay = Exponential(0.5))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            rv = RingVaccination(efficacy = 0.9,
                coverage = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0)
            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv], attributes = attrs);
                max_cases = 200, rng = StableRNG(101))
            for ind in state.individuals
                is_vaccinated(ind) && @test ind.state[:age] >= 50
            end
        end

        @testset "Eligibility window skips late vaccinations" begin
            # With a long isolation delay, only some traced contacts are
            # within a tight window. A short window should produce strictly
            # fewer vaccinations than an infinite one.
            iso = Isolation(onset_to_isolation_delay = Exponential(5.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(1.0))

            rv_inf = RingVaccination(efficacy = 0.9, eligibility_window = Inf)
            n_inf = sum(
                count(is_vaccinated, s.individuals)
            for s in simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_inf], attributes = clinical),
                50; max_cases = 100, rng = StableRNG(3)))

            rv_narrow = RingVaccination(efficacy = 0.9, eligibility_window = 1.0)
            states_narrow = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_narrow], attributes = clinical),
                50; max_cases = 100, rng = StableRNG(3))
            n_narrow = sum(
                count(is_vaccinated, s.individuals) for s in states_narrow)
            @test n_narrow < n_inf

            # Every vaccinated contact must satisfy the window.
            for state in states_narrow
                for ind in state.individuals
                    if is_vaccinated(ind)
                        @test isolation_time(ind) - ind.infection_time <= 1.0
                    end
                end
            end
        end

        @testset "Onward efficacy blocks next-generation transmission" begin
            # With onward_efficacy = 1.0 and delay_to_immunity = 0.0,
            # any infected child of a vaccinated parent must have been
            # infected strictly before the parent's vaccination time
            # (i.e. before the parent was even traced/isolated) — once
            # the parent's immunity is in place the onward risk is
            # certain to block.
            iso = Isolation(onset_to_isolation_delay = Exponential(0.5))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            rv = RingVaccination(efficacy = 0.0, onward_efficacy = 1.0,
                delay_to_immunity = 0.0)

            saw_blocked_chain = false
            for seed in 1:5
                state = simulate(
                    BranchingProcess(Poisson(3.0), Exponential(5.0);
                        interventions = [iso, ct, rv], attributes = clinical);
                    max_cases = 500, rng = StableRNG(seed))
                by_id = Dict(ind.id => ind for ind in state.individuals)
                for child in state.individuals
                    child.parent_id == 0 && continue
                    is_infected(child) || continue
                    parent = by_id[child.parent_id]
                    if is_vaccinated(parent)
                        saw_blocked_chain = true
                        @test child.infection_time < parent.state[:vaccination_time]
                    end
                end
            end
            @test saw_blocked_chain  # otherwise the test is vacuous
        end

        @testset "Onward efficacy default is no-op" begin
            # onward_efficacy = 0.0 (the default) returns no onward risk,
            # so a parent's vaccination state cannot affect their onward
            # transmission. Compare against a deterministic baseline:
            # with only the susceptibility risk in play, the simulation
            # should be bit-identical to the previous behaviour for the
            # same seed.
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            rv = RingVaccination(efficacy = 0.9)  # default onward_efficacy

            rng1 = StableRNG(42)
            results_default = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            # Explicit onward_efficacy = 0.0 should reproduce the same
            # outcome with the same seed (no extra rng draws).
            rv_explicit = RingVaccination(efficacy = 0.9, onward_efficacy = 0.0)
            rng2 = StableRNG(42)
            results_explicit = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct, rv_explicit], attributes = clinical),
                100;
                max_cases = 200,
                rng = rng2)

            @test [s.cumulative_cases for s in results_default] ==
                  [s.cumulative_cases for s in results_explicit]
        end
    end

    @testset "Contact tracing without quarantine" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            quarantine_on_trace = false)

        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso, ct], attributes = clinical);
            max_cases = 50, rng = rng)

        n_traced = count(is_traced, state.individuals)
        if n_traced > 0
            # Traced contacts should not be quarantined
            traced = filter(is_traced, state.individuals)
            @test !any(is_quarantined, traced)
        end
    end

    @testset "Traced test-negative contacts isolate via tracing pathway" begin
        # Regression test: with test_sensitivity < 1 and FlagOnly tracing,
        # test-negative contacts must still be isolated via the tracing
        # pathway. Previously a `is_test_positive || return` gate in
        # Isolation::resolve_individual discarded traced_isolation_time for
        # test-negative contacts, so 1 − test_sensitivity of cases never
        # isolated even when traced.
        #
        # Filter to cases that ran through resolve_individual at least once
        # (those that became parents — they have non-empty
        # secondary_case_ids — or those who are themselves index cases).
        # Final-generation contacts created in the last step never resolve
        # because the engine stops before they would be active.
        rng = StableRNG(20260601)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), test_sensitivity = 0.4)
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            quarantine_on_trace = false)

        state = simulate(
            BranchingProcess(Poisson(3.0), Exponential(5.0);
                interventions = [iso, ct], attributes = clinical);
            max_cases = 500, rng = rng)

        resolved = filter(
            ind -> !isempty(ind.secondary_case_ids) ||
                   ind.parent_id == 0,
            state.individuals
        )
        traced_symptomatic = filter(
            ind -> is_traced(ind) && !is_asymptomatic(ind),
            resolved
        )
        @test !isempty(traced_symptomatic)
        @test all(is_isolated, traced_symptomatic)

        traced_test_negative = filter(
            ind -> !is_test_positive(ind), traced_symptomatic
        )
        @test !isempty(traced_test_negative)
        @test all(is_isolated, traced_test_negative)
    end

    @testset "Scheduled interventions" begin
        @testset "start_time delays activation" begin
            # Compare scheduled (late start) vs always-on — scheduled should contain less
            rng1 = StableRNG(42)
            iso_late = Scheduled(Isolation(onset_to_isolation_delay = Exponential(1.0)); start_time = 20.0)
            results_late = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso_late], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            iso_always = Isolation(onset_to_isolation_delay = Exponential(1.0))
            results_always = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso_always], attributes = clinical),
                100; max_cases = 200, rng = rng2)

            @test containment_probability(results_always) >=
                  containment_probability(results_late)

            # Fields should still be initialised on all individuals
            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso_late], attributes = clinical);
                max_cases = 50, rng = StableRNG(99))
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
            end
        end

        @testset "start_after_cases delays activation" begin
            iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(0.5)); start_after_cases = 20)

            rng1 = StableRNG(42)
            results_scheduled = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            # Compare with always-on isolation — scheduled should contain less
            rng2 = StableRNG(42)
            iso_always = Isolation(onset_to_isolation_delay = Exponential(0.5))
            results_always = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso_always], attributes = clinical),
                100; max_cases = 200, rng = rng2)

            @test containment_probability(results_always) >=
                  containment_probability(results_scheduled)
        end

        @testset "custom predicate" begin
            iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(1.0)),
                state -> state.current_generation >= 3)

            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))

            # Verify the intervention ran without errors and fields exist
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
            end
        end

        @testset "end_time deactivates" begin
            # Active only in a short window
            iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(0.5));
                start_time = 5.0, end_time = 10.0)

            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))

            # Late individuals should not be isolated
            late = filter(i -> is_infected(i) && i.infection_time > 15.0, state.individuals)
            if !isempty(late)
                @test !any(is_isolated, late)
            end
        end

        @testset "mixed Scheduled and always-on" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = Scheduled(
                ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(1.0));
                start_after_cases = 10)

            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso, ct], attributes = clinical);
                max_cases = 50, rng = StableRNG(42))

            # All individuals should have both isolation and tracing fields
            for ind in state.individuals
                @test haskey(ind.state, :isolated)
                @test haskey(ind.state, :traced)
            end
        end

        @testset "filters on action time not infection time" begin
            iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(0.1)); start_time = 15.0)

            state = simulate(
                BranchingProcess(Poisson(3.0), Exponential(5.0);
                    interventions = [iso], attributes = clinical);
                max_cases = 200, rng = StableRNG(42))

            # No individual should be isolated with isolation_time < 15.0
            for ind in state.individuals
                if is_isolated(ind)
                    @test isolation_time(ind) >= 15.0
                end
            end
        end

        @testset "requires at least one condition" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            @test_throws ErrorException Scheduled(iso)
        end
    end
end
