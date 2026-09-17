# A minimal intervention that overrides nothing, used to check the protocol's
# defaults are inert.
struct _NoTraceIntervention <: AbstractIntervention end

@testset "Interventions" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Isolation reduces transmission" begin
        rng1 = StableRNG(42)
        results_no_iso = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); attributes = clinical),
            100; max_cases = 200, rng = rng1)

        rng2 = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
        results_iso = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
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
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = clinical);
            max_cases = 50, rng = rng)

        @test all(ind -> ind.infection_time < 1000.0, state.individuals)
    end

    @testset "reset!(Isolation) leaves another intervention's isolation intact" begin
        # `:isolated`/`:isolation_time` are shared: ContactTracing's Quarantine
        # writes them directly. A Scheduled(Isolation) resetting a pre-start
        # isolation must not un-quarantine a contact Isolation never touched.
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))

        # Isolation set by another intervention (no provenance marker).
        traced = Individual(id = 1)
        set_isolated!(traced, 3.0)
        @test is_isolated(traced)
        EpiBranch.reset!(iso, traced)
        @test is_isolated(traced)               # preserved
        @test isolation_time(traced) == 3.0

        # An isolation Isolation itself set (marked) is still reset.
        own = Individual(id = 2)
        set_isolated!(own, 3.0)
        own.state[:isolated_by_isolation] = true
        EpiBranch.reset!(iso, own)
        @test !is_isolated(own)
    end

    @testset "Continuous-time tracing hooks" begin
        # The hooks the Sellke models call. Exercised here directly because the
        # only processes that drive them live in the companion packages, so the
        # core suite would otherwise never touch this code.
        ct = ContactTracing(TraceEveryone(), 1.0, Exponential(0.5))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))

        # Only interventions that trace declare themselves, so the race can skip
        # gathering contacts entirely when nothing needs them.
        @test EpiBranch.traces_contacts(ct)
        @test !EpiBranch.traces_contacts(iso)
        @test EpiBranch.traces_contacts(Scheduled(ct; start_time = 5.0))
        @test !EpiBranch.traces_contacts(Scheduled(iso; start_time = 5.0))

        # Models declare whether they can name a case's contacts at all.
        @test !EpiBranch.supplies_contacts(BranchingProcess(Poisson(1.0)))

        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))
        function pair()
            infector = Individual(id = 1)
            infector.state[:infected] = true
            infector.state[:onset_time] = 1.0
            set_isolated!(infector, 2.0)
            contact = Individual(id = 2)
            EpiBranch.initialise_individual!(ct, contact, state)
            return infector, contact
        end

        # The bare intervention traces the contacts it is handed.
        infector, contact = pair()
        EpiBranch.trace_contacts!(ct, state, infector, [contact])
        @test is_traced(contact)
        @test is_quarantined(contact)
        @test contact.state[:traced_by] == infector.id

        # A case never traces itself, even when handed to itself.
        solo, _ = pair()
        EpiBranch.trace_contacts!(ct, state, solo, [solo])
        @test !is_traced(solo)

        # Scheduled delegates when active and stays out of the way when not.
        infector, contact = pair()
        EpiBranch.trace_contacts!(Scheduled(ct; start_time = 0.0), state, infector,
            [contact])
        @test is_traced(contact)

        infector, contact = pair()
        state.max_infection_time = 0.0
        EpiBranch.trace_contacts!(Scheduled(ct; start_time = 100.0), state, infector,
            [contact])
        @test !is_traced(contact)

        # A quarantined contact reports its quarantine as the time it leaves
        # onward transmission; an untraced one contributes no removal.
        traced = Individual(id = 3)
        traced.state[:quarantined] = true
        set_isolated!(traced, 4.0)
        @test EpiBranch.infectious_removal_time(ct, traced) == 4.0
        @test EpiBranch.infectious_removal_time(ct, Individual(id = 4)) == Inf

        # A contact on a route that opens later is traced no earlier than the
        # route opens, directly or through Scheduled; an earlier bound leaves
        # the case's own trace time alone.
        infector, contact = pair()
        EpiBranch.trace_contacts!(ct, state, infector, [contact], [50.0])
        @test is_quarantined(contact)
        # the trace delay runs from when the contact could first be reached
        @test isolation_time(contact) > 50.0
        infector, contact = pair()
        EpiBranch.trace_contacts!(Scheduled(ct; start_time = 0.0), state, infector,
            [contact], [50.0])
        @test isolation_time(contact) >= 50.0
        infector, early = pair()
        EpiBranch.trace_contacts!(ct, state, infector, [early], [-Inf])
        @test isolation_time(early) < 50.0

        # Defaults are inert, so an intervention that does not trace costs
        # nothing on the continuous-time path, with or without trace bounds.
        @test !EpiBranch.traces_contacts(_NoTraceIntervention())
        @test EpiBranch.trace_contacts!(
            _NoTraceIntervention(), state, Individual(id = 5), Individual[]) === nothing
        @test EpiBranch.trace_contacts!(
            _NoTraceIntervention(), state, Individual(id = 5), Individual[],
            Float64[]) === nothing
    end

    @testset "Isolation keeps the earliest pathway when already isolated" begin
        # A quarantine written by ContactTracing leaves `:isolated` set before
        # Isolation resolves the individual. That is the ordering the
        # continuous-time models produce, where a contact is traced when its
        # infector settles rather than when it settles itself. Isolation must
        # treat the standing quarantine as a competing pathway: if the
        # individual would have self-reported earlier, the earlier time wins.
        # Otherwise tracing *delays* isolation instead of advancing it.
        iso = Isolation(onset_to_isolation_delay = Exponential(1e-9))
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        # Quarantined late, but onset was early: the self-reported time wins.
        late = Individual(id = 1)
        late.state[:onset_time] = 2.0
        late.state[:test_positive] = true
        set_isolated!(late, 50.0)
        EpiBranch.resolve_individual!(iso, late, state)
        @test isolation_time(late) ≈ 2.0 atol=1e-6
        @test get(late.state, :isolated_by_isolation, false)

        # Quarantined early: the quarantine stands and is left untouched.
        early = Individual(id = 2)
        early.state[:onset_time] = 40.0
        early.state[:test_positive] = true
        set_isolated!(early, 1.0)
        EpiBranch.resolve_individual!(iso, early, state)
        @test isolation_time(early) == 1.0
        @test !get(early.state, :isolated_by_isolation, false)

        # A test-negative quarantined contact has no self-reporting pathway,
        # so the quarantine stands.
        negative = Individual(id = 3)
        negative.state[:onset_time] = 2.0
        negative.state[:test_positive] = false
        set_isolated!(negative, 50.0)
        EpiBranch.resolve_individual!(iso, negative, state)
        @test isolation_time(negative) == 50.0
    end

    @testset "Asymptomatic cases are not isolated" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        clinical_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5),
            prob_asymptomatic = 0.5
        )

        state = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = clinical);
            max_cases = 100, rng = StableRNG(42))

        n_isolated = count(is_isolated, state.individuals)
        @test n_isolated == 0
    end

    @testset "Hazard-based isolation reduces more with early isolation" begin
        rng1 = StableRNG(42)
        iso_fast = Isolation(onset_to_isolation_delay = Exponential(0.5))
        results_fast = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso_fast], attributes = clinical),
            200; max_cases = 200, rng = rng1)

        rng2 = StableRNG(42)
        iso_slow = Isolation(onset_to_isolation_delay = Exponential(10.0))
        results_slow = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso_slow], attributes = clinical),
            200; max_cases = 200, rng = rng2)

        @test containment_probability(results_fast) >= containment_probability(results_slow)
    end

    @testset "Intervention initialises state on individuals" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(1.0))

        state = simulate(
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
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
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0)); interventions = [iso]);
            max_cases = 10, rng = StableRNG(42))
    end

    @testset "attribute list works" begin
        rng = StableRNG(42)
        init_fn = [
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Normal(40, 15))
        ]

        state = simulate(
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0)); attributes = init_fn);
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            results_no_vacc = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_instant], attributes = clinical),
                200; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            rv_delayed = RingVaccination(efficacy = 0.9, delay_to_immunity = 14.0)
            results_delayed = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_zero], attributes = clinical);
                max_cases = 100, rng = StableRNG(42))
            @test count(is_vaccinated, state.individuals) == 0
            @test count(is_traced, state.individuals) > 0

            # Coverage = 1 reproduces the previous behaviour: every eligible
            # traced contact is vaccinated.
            rv_full = RingVaccination(efficacy = 0.9, coverage = 1.0)
            state_full = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_partial], attributes = clinical),
                30; max_cases = 100, rng = StableRNG(7)))
            n_vacc_full_batch = sum(
                count(is_vaccinated, s.individuals)
            for s in simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_full], attributes = clinical),
                30; max_cases = 100, rng = StableRNG(7)))
            @test n_vacc_partial < n_vacc_full_batch
        end

        @testset "Coverage accepts a function" begin
            # Age-conditional coverage: 50+ always vaccinated, under-50 never.
            attrs = [clinical,
                demographics(age_distribution = Uniform(0, 90))]
            iso = Isolation(onset_to_isolation_delay = Exponential(0.5))
            ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            rv = RingVaccination(efficacy = 0.9,
                coverage = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0)
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_inf], attributes = clinical),
                50; max_cases = 100, rng = StableRNG(3)))

            rv_narrow = RingVaccination(efficacy = 0.9, eligibility_window = 1.0)
            states_narrow = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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

        @testset "An unbounded eligibility window admits a contact not yet exposed" begin
            # A never-infected individual's infection time is NaN, which fails
            # any comparison against a finite window.
            unexposed = Individual(id = 1, infection_time = NaN)
            rng = StableRNG(1)
            @test EpiBranch._within_eligibility_window(Inf, unexposed, 3.0, rng)
            @test EpiBranch._within_eligibility_window(
                (rng, ind) -> Inf, unexposed, 3.0, rng)
            @test !EpiBranch._within_eligibility_window(21.0, unexposed, 3.0, rng)
        end

        @testset "Doses are timed at the trace, whatever the trace action" begin
            # A ring member is vaccinated when the tracing team reaches
            # them, so `:vaccination_time` is the trace time.
            iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
            for quarantine in (true, false)
                ct = ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Exponential(1.0),
                    quarantine_on_trace = quarantine)
                rv = RingVaccination(efficacy = 0.9)
                state = simulate(
                    ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                        interventions = [iso, ct, rv], attributes = clinical);
                    condition = 50:200, max_cases = 200, rng = StableRNG(11))
                n_checked = 0
                for ind in state.individuals
                    is_vaccinated(ind) || continue
                    n_checked += 1
                    @test ind.state[:vaccination_time] == ind.state[:trace_time]
                end
                @test n_checked > 0  # otherwise the test is vacuous
            end
        end

        @testset "Asymptomatic traced contacts are vaccinated" begin
            # Without a quarantine, tracing records an isolation time only
            # for contacts with a known onset, and an asymptomatic ring
            # member still gets a dose.
            clinical_asymp = clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5),
                prob_asymptomatic = 0.3)
            iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
            ct = ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(1.0),
                quarantine_on_trace = false)
            rv = RingVaccination(efficacy = 0.9)

            state = simulate(
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical_asymp);
                condition = 50:200, max_cases = 200, rng = StableRNG(1))
            traced_asymp = filter(
                ind -> is_traced(ind) && get(ind.state, :asymptomatic, false),
                state.individuals)
            @test !isempty(traced_asymp)  # otherwise the test is vacuous
            @test all(is_vaccinated, traced_asymp)
        end

        @testset "Second dose" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
            ct = ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(1.0))
            prime = RingVaccination(efficacy = 0.6, delay_to_immunity = 21.0,
                dose_label = :prime)
            process = BranchingProcess(Poisson(2.0), Exponential(5.0))

            @testset "Given dose_delay days after the prime" begin
                boost = RingVaccination(efficacy = 0.5, dose_delay = 28.0,
                    delay_to_immunity = 14.0, requires_dose = :prime,
                    dose_label = :boost)
                state = simulate(
                    ModelSpec(process; interventions = [iso, ct, prime, boost],
                        attributes = clinical);
                    condition = 50:200, max_cases = 200, rng = StableRNG(5))
                n_boosted = 0
                for ind in state.individuals
                    get(ind.state, :vaccinated_boost, false) || continue
                    n_boosted += 1
                    @test ind.state[:vaccination_time_boost] ==
                          ind.state[:vaccination_time_prime] + 28.0
                end
                @test n_boosted > 0  # otherwise the test is vacuous
            end

            @testset "requires_dose gates the boost on the prime" begin
                # Nobody is primed, so nobody can be boosted.
                prime_none = RingVaccination(efficacy = 0.6, coverage = 0.0,
                    dose_label = :prime)
                boost = RingVaccination(efficacy = 0.5, requires_dose = :prime,
                    dose_label = :boost)
                state = simulate(
                    ModelSpec(process;
                        interventions = [iso, ct, prime_none, boost],
                        attributes = clinical);
                    condition = 50:200, max_cases = 200, rng = StableRNG(5))
                @test count(is_traced, state.individuals) > 0
                @test !any(i -> get(i.state, :vaccinated_boost, false),
                    state.individuals)
            end

            @testset "Boost coverage thins the boosted among the primed" begin
                # Full coverage boosts everyone primed; partial coverage
                # boosts a strict subset. Each run is checked on its own,
                # because the coverage draws shift the rng stream between runs.
                full = RingVaccination(efficacy = 0.5, requires_dose = :prime,
                    dose_label = :boost)
                partial = RingVaccination(efficacy = 0.5, coverage = 0.5,
                    requires_dose = :prime, dose_label = :boost)
                for (boost, boosts_everyone) in ((full, true), (partial, false))
                    states = simulate(
                        ModelSpec(process; interventions = [iso, ct, prime, boost],
                            attributes = clinical),
                        30; max_cases = 100, rng = StableRNG(9))
                    primed = sum(count(i -> i.state[:vaccinated_prime], s.individuals)
                    for s in states)
                    boosted = sum(count(i -> i.state[:vaccinated_boost], s.individuals)
                    for s in states)
                    @test primed > 0  # otherwise the test is vacuous
                    for s in states, ind in s.individuals

                        ind.state[:vaccinated_boost] && @test ind.state[:vaccinated_prime]
                    end
                    boosts_everyone ? (@test boosted == primed) :
                    (@test 0 < boosted < primed)
                end
            end

            @testset "A dose is not given before the dose it requires" begin
                # A mass prime is recorded as soon as its eligibility time is
                # drawn, and on day 60 that time lies after most boosts fall
                # due. Day 20 checks that boosts after the prime still happen.
                n_boosted = 0
                n_early = 0
                for eligibility_time in (60.0, 20.0)
                    mass_prime = MassVaccination(efficacy = 0.6,
                        eligibility_time = eligibility_time, dose_label = :prime)
                    boost = RingVaccination(efficacy = 0.5, dose_delay = 7.0,
                        requires_dose = :prime, dose_label = :boost)
                    states = simulate(
                        ModelSpec(process; interventions = [iso, ct, mass_prime, boost],
                            attributes = clinical),
                        20; max_cases = 300, rng = StableRNG(3))
                    for s in states, ind in s.individuals

                        get(ind.state, :vaccinated_boost, false) || continue
                        n_boosted += 1
                        n_early += ind.state[:vaccination_time_boost] <
                                   ind.state[:vaccination_time_prime]
                    end
                end
                @test n_boosted > 0  # otherwise the test is vacuous
                @test n_early == 0
            end

            @testset "A dose scheduled before the one it requires is rejected" begin
                # List order is right, but the boost arrives at the trace while
                # the prime does not arrive until 28 days later.
                late_prime = RingVaccination(efficacy = 0.6, dose_delay = 28.0,
                    dose_label = :prime)
                early_boost = RingVaccination(efficacy = 0.5, dose_delay = 0.0,
                    requires_dose = :prime, dose_label = :boost)
                @test_throws ArgumentError ModelSpec(process;
                    interventions = [iso, ct, late_prime, early_boost],
                    attributes = clinical)
                # Same instant is allowed: both doses are given at the trace.
                same_instant = RingVaccination(efficacy = 0.5,
                    requires_dose = :prime, dose_label = :boost)
                @test ModelSpec(process;
                    interventions = [iso, ct, prime, same_instant],
                    attributes = clinical) isa ModelSpec
            end

            @testset "A dose listed before the one it requires is rejected" begin
                boost = RingVaccination(efficacy = 0.5, requires_dose = :prime,
                    dose_label = :boost)
                @test_throws ArgumentError ModelSpec(process;
                    interventions = [iso, ct, boost, prime], attributes = clinical)
                # Wrapping in `Scheduled` must not hide the requirement.
                @test_throws ArgumentError ModelSpec(process;
                    interventions = [iso, ct, Scheduled(boost; start_time = 10.0),
                        prime], attributes = clinical)
                @test ModelSpec(process;
                    interventions = [iso, ct, prime, boost],
                    attributes = clinical) isa ModelSpec
            end
        end

        @testset "Post-exposure efficacy" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
            ct = ContactTracing(probability = 0.7,
                isolation_to_trace_delay = Exponential(1.0),
                quarantine_on_trace = false)
            process = BranchingProcess(Poisson(3.0), Exponential(5.0))
            scen(iv, attrs = clinical) = ModelSpec(process;
                interventions = iv, attributes = attrs)
            post_only(; kwargs...) = RingVaccination(efficacy = 0.0,
                post_exposure_efficacy = 0.9; kwargs...)

            # An outbreak with fixed delays. The index case, infected at 0, has
            # onset at 5 and is isolated at 5.5, when both its contacts (infected
            # at 1 or 5) are traced and vaccinated with immediate immunity. Their
            # onsets (6 or 10) come after that, so a certain dose aborts both at
            # 5.5. A contact infected at 1 has contacts of its own at 2, before
            # the abort, or at 6, after it. Only the index tests positive, so a
            # contact could be isolated only through its trace.
            function aborted_outbreak(eligibility = SymptomaticParent())
                model = ModelSpec(
                    BranchingProcess(Dirac(2),
                        DiscreteNonParametric([1.0, 5.0], [0.5, 0.5]));
                    interventions = [
                        Isolation(onset_to_isolation_delay = Dirac(0.5),
                            test_sensitivity = (rng, ind) -> ind.parent_id == 0 ? 1.0 : 0.0),
                        ContactTracing(eligibility, ConstantRate(1.0),
                            ConstantDelay(Dirac(0.0)), FlagOnly()),
                        RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0)],
                    attributes = clinical_presentation(incubation_period = Dirac(5.0)),
                    progression = [Recovery(delay = Dirac(3.0))])
                # This seed gives the index one contact at each time, and the
                # contact infected at 1 one contact at each of 2 and 6.
                state = simulate(model; max_generations = 2, max_cases = nothing,
                    rng = StableRNG(4))
                cases = filter(is_infected, state.individuals)
                return state, filter(i -> i.generation == 1, cases),
                filter(i -> i.generation == 2, cases)
            end

            @testset "An aborted infection keeps its earlier transmissions" begin
                state, contacts, grandcontacts = aborted_outbreak()
                @test length(contacts) == 2
                @test all(c -> c.state[:infection_aborted_time] == 5.5, contacts)
                early = filter(c -> c.infection_time == 1.0, contacts)
                @test !isempty(early)
                # Transmission before immunity stands; none follows it.
                @test !isempty(grandcontacts)
                @test all(g -> g.infection_time < 5.5, grandcontacts)
                @test any(state.individuals) do ind
                    ind.generation == 2 && !is_infected(ind) &&
                        ind.infection_time > 5.5
                end
            end

            @testset "An aborted infection has no onset or outcome" begin
                state, contacts, _ = aborted_outbreak()
                index = state.individuals[1]
                @test onset_time(index) == 5.0
                @test index.state[:outcome] == :recovered
                for c in contacts
                    @test isnan(onset_time(c))
                    @test !haskey(c.state, :outcome)
                    # Traced before the onset it would have had, but never
                    # isolated because it has no onset.
                    @test is_traced(c)
                    @test !is_isolated(c)
                end
                df = linelist(state)
                aborted_rows = filter(r -> r.generation == 1, df)
                @test all(ismissing, aborted_rows.date_onset)
                @test all(!ismissing, aborted_rows.date_infection_aborted)
                @test size(df, 1) == length(filter(is_infected, state.individuals))
            end

            @testset "An aborted infection's clinical course ends at the abort" begin
                # Infected at 1 and aborted at 4: whatever a transition is timed
                # from, it stands if it takes effect before 4 and is undone
                # otherwise, along with everything timed from it.
                progression = [
                    Transition(:early, from = :infection, delay = 1.0),
                    Transition(:worse, from = :early, delay = 1.0),
                    Transition(:hospitalised, from = :infection, delay = 6.0),
                    Transition(:died, from = :hospitalised, delay = 1.0,
                        terminal = true),
                    Reporting(delay = 0.5, from = :early_time),
                    Death(delay = 5.0, probability = 1.0,
                        from = ind -> ind.infection_time),
                    Recovery(delay = 10.0, from = ind -> ind.infection_time)]
                function course(aborted)
                    state = EpiBranch.new_state(process, progression,
                        EpiBranch.NoAttributes(), StableRNG(1))
                    ind = Individual(id = 1, infection_time = 1.0)
                    aborted && (ind.state[:infection_aborted_time] = 4.0)
                    EpiBranch.resolve_transitions!(state, ind)
                    return ind.state
                end

                st = course(true)
                @test st[:early] && st[:early_time] == 2.0
                @test st[:worse] && st[:worse_time] == 3.0
                @test st[:reported] && st[:reporting_time] == 2.5
                @test !st[:hospitalised] && st[:hospitalised_time] == Inf
                @test !st[:died] && st[:died_time] == Inf
                @test st[:death_candidate_time] == Inf
                @test st[:recovery_candidate_time] == Inf
                @test !haskey(st, :outcome) && !haskey(st, :outcome_time)

                control = course(false)
                @test control[:hospitalised] && control[:died]
                @test control[:outcome] == :died && control[:outcome_time] == 6.0
            end

            @testset "No transition takes effect after an abort in simulation" begin
                progression = [
                    Transition(:hospitalised, from = :infection, delay = Gamma(4, 2),
                        probability = 0.2),
                    Transition(:died, from = :hospitalised, delay = Gamma(2, 3),
                        probability = 0.5, terminal = true),
                    Death(delay = LogNormal(2.0, 0.4), probability = 0.1)]
                spec = ModelSpec(process; progression,
                    interventions = [iso, ct, post_only()], attributes = clinical)
                results = simulate(spec, 30; max_cases = 300, rng = StableRNG(2))
                aborted = 0
                late = 0
                miscounted = 0
                for s in results
                    for ind in filter(is_infected, s.individuals)
                        t = get(ind.state, :infection_aborted_time, nothing)
                        t === nothing && continue
                        aborted += 1
                        for key in (:hospitalised_time, :died_time,
                            :death_candidate_time, :outcome_time)
                            get(ind.state, key, Inf) < Inf &&
                                ind.state[key] >= t && (late += 1)
                        end
                    end
                    # Aborted cases stay in the line list and the chain sizes.
                    n_cases = count(is_infected, s.individuals)
                    size(linelist(s), 1) == n_cases &&
                    sum(chain_statistics(s).size) == n_cases || (miscounted += 1)
                end
                @test aborted > 0
                @test late == 0
                @test miscounted == 0
            end

            @testset "An aborted infection seeds no onset-triggered ring" begin
                state, _, grandcontacts = aborted_outbreak(OnSymptomOnset())
                @test !isempty(grandcontacts)
                @test !any(is_traced, grandcontacts)
                @test !any(ind -> isnan(isolation_time(ind)), state.individuals)
            end

            @testset "Protects contacts a pre-exposure dose cannot reach" begin
                # `efficacy` needs immunity before the exposure, which a dose
                # given at the trace never achieves here; `post_exposure_efficacy`
                # needs immunity before onset, which it often achieves.
                base = simulate(scen([iso, ct]), 400; max_cases = 200,
                    rng = StableRNG(42))
                pre = simulate(scen([iso, ct, RingVaccination(efficacy = 0.9)]),
                    400; max_cases = 200, rng = StableRNG(42))
                post = simulate(scen([iso, ct, post_only()]),
                    400; max_cases = 200, rng = StableRNG(42))

                @test containment_probability(pre) == containment_probability(base)
                @test containment_probability(post) > containment_probability(base)
            end

            @testset "Aborting protects less than blocking all later transmission" begin
                # An abort removes only what a contact would transmit after its
                # immunity, and only for contacts whose immunity arrives before
                # their onset, so it cannot do much more than blocking all of a
                # vaccinated contact's later transmission.
                containment(rv) = mean(containment_probability(
                                           simulate(scen([iso, ct, rv]), 200;
                                           max_cases = 200, rng = StableRNG(seed)))
                for seed in 1:5)
                onward = containment(RingVaccination(efficacy = 0.0,
                    onward_efficacy = 0.9))
                @test containment(post_only()) < onward + 0.05
            end

            @testset "A dose that cannot abort leaves the run untouched" begin
                # Incubation periods here average about 5 days, so immunity 100
                # days after the trace never arrives before an onset, and the
                # run should match one without the parameter draw for draw.
                slow(post) = RingVaccination(efficacy = 0.0,
                    post_exposure_efficacy = post, delay_to_immunity = 100.0)
                fingerprint(states) = [(ind.id, ind.infection_time,
                                           sort!(collect(ind.state); by = first))
                                       for s in states for ind in s.individuals]
                base = simulate(scen([iso, ct, slow(0.0)]), 100; max_cases = 200,
                    rng = StableRNG(3))
                results = simulate(scen([iso, ct, slow(1.0)]), 100; max_cases = 200,
                    rng = StableRNG(3))
                @test isequal(fingerprint(results), fingerprint(base))
            end

            @testset "The abort races immunity against onset" begin
                rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0,
                    delay_to_immunity = 2.0)
                function contact_with(incubation)
                    c = Individual(id = 1, infection_time = 10.0)
                    c.state[:incubation_period] = incubation
                    c.state[:onset_time] = 10.0 + incubation
                    return c
                end
                rng = StableRNG(1)

                # Immunity at 14 beats an onset at 16.
                c = contact_with(6.0)
                EpiBranch._abort_infection!(rv, c, 12.0, rng)
                @test c.state[:infection_aborted_time] == 14.0
                @test isnan(onset_time(c))

                # Onset at 13 comes before immunity at 14.
                c = contact_with(3.0)
                EpiBranch._abort_infection!(rv, c, 12.0, rng)
                @test !haskey(c.state, :infection_aborted_time)
                @test onset_time(c) == 13.0

                # Immunity at 9 precedes the exposure: that is a blocked
                # infection, left to the contact-side risk.
                c = contact_with(6.0)
                EpiBranch._abort_infection!(rv, c, 7.0, rng)
                @test !haskey(c.state, :infection_aborted_time)
                c.state[:vaccinated] = true
                c.state[:vaccination_time] = 7.0
                risk = EpiBranch._contact_risk(rv, c)
                @test risk.event_time == 9.0
                @test risk.block_probability == 1.0

                # No onset to race: asymptomatic contacts are never aborted.
                c = contact_with(NaN)
                EpiBranch._abort_infection!(rv, c, 12.0, rng)
                @test !haskey(c.state, :infection_aborted_time)

                # Only a dose that can matter draws from the rng.
                partial = RingVaccination(efficacy = 0.0,
                    post_exposure_efficacy = 0.5, delay_to_immunity = 2.0)
                r1, r2 = StableRNG(9), StableRNG(9)
                EpiBranch._abort_infection!(partial, contact_with(3.0), 12.0, r1)
                @test rand(r1) == rand(r2)
            end

            @testset "Setting both efficacies warns" begin
                both = RingVaccination(efficacy = 0.9, post_exposure_efficacy = 0.9)
                @test_logs (:warn, r"both `efficacy` and `post_exposure_efficacy`") ModelSpec(
                    process; interventions = [iso, ct, both], attributes = clinical)
                # Either alone is silent.
                @test_logs ModelSpec(process;
                    interventions = [iso, ct, post_only()], attributes = clinical)
                @test_logs ModelSpec(process;
                    interventions = [iso, ct, RingVaccination(efficacy = 0.9)],
                    attributes = clinical)
            end

            @testset "Every combination of risks is returned" begin
                function risks(rv; contact_dosed = true, parent_dosed = true)
                    parent = Individual(id = 1, infection_time = 0.0)
                    contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
                    for (ind, dosed) in ((contact, contact_dosed), (parent, parent_dosed))
                        ind.state[:vaccinated] = dosed
                        ind.state[:vaccination_time] = dosed ? 2.0 : Inf
                        ind.state[:vaccine_efficacy] = rv.efficacy
                    end
                    r = EpiBranch.competing_risk(rv, parent, contact, nothing)
                    r === nothing ? 0 : (r isa EpiBranch.Risk ? 1 : length(r))
                end

                susceptibility_only = RingVaccination(efficacy = 0.5)
                post = post_only()
                onward_only = RingVaccination(efficacy = 0.0, onward_efficacy = 0.5)
                post_onward = RingVaccination(efficacy = 0.0,
                    post_exposure_efficacy = 0.5, onward_efficacy = 0.5)

                @test risks(susceptibility_only) == 1
                @test risks(post) == 1
                @test risks(post; contact_dosed = false) == 0
                @test risks(onward_only; contact_dosed = false) == 1
                @test risks(RingVaccination(efficacy = 0.5, onward_efficacy = 0.5)) == 2
                @test risks(post_onward) == 2
                @test risks(post_onward; contact_dosed = false) == 1
                # Efficacy and post-exposure efficacy share one contact-side
                # block at the same immunity time.
                both = RingVaccination(efficacy = 0.5, post_exposure_efficacy = 0.5)
                @test risks(both) == 1
                @test risks(post_onward; contact_dosed = false, parent_dosed = false) == 0
            end

            @testset "The engine ends an aborted infection" begin
                parent = Individual(id = 1, infection_time = 0.0)
                contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
                @test EpiBranch.competing_risk(EpiBranch.AbortedInfection(),
                    parent, contact, nothing) === nothing
                parent.state[:infection_aborted_time] = 4.0
                risk = EpiBranch.competing_risk(EpiBranch.AbortedInfection(),
                    parent, contact, nothing)
                @test risk.event_time == 4.0
                @test risk.block_probability == 1.0
            end

            @testset "An aborted infection stays ended after a scheduled dose stops" begin
                # The abort is stored on the case, so the end of its
                # transmission does not depend on whether the ring vaccination
                # that recorded it is still active.
                scheduled = Scheduled(post_only(); end_time = 15.0)
                results = simulate(scen([iso, ct, scheduled]), 100;
                    max_cases = 300, rng = StableRNG(1))
                aborted = 0
                after_abort = 0
                for s in results
                    for ind in filter(is_infected, s.individuals)
                        t = get(ind.state, :infection_aborted_time, nothing)
                        t === nothing && continue
                        aborted += 1
                        after_abort += count(ind.secondary_case_ids) do id
                            child = s.individuals[id]
                            is_infected(child) && child.infection_time >= t
                        end
                    end
                end
                @test aborted > 0
                @test after_abort == 0
            end

            @testset "Requires an incubation period" begin
                @test :incubation_period in EpiBranch.required_fields(post_only())
                @test :incubation_period ∉
                      EpiBranch.required_fields(RingVaccination(efficacy = 0.9))
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
                    ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            # Explicit onward_efficacy = 0.0 should reproduce the same
            # outcome with the same seed (no extra rng draws).
            rv_explicit = RingVaccination(efficacy = 0.9, onward_efficacy = 0.0)
            rng2 = StableRNG(42)
            results_explicit = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv_explicit], attributes = clinical),
                100;
                max_cases = 200,
                rng = rng2)

            @test [s.cumulative_cases for s in results_default] ==
                  [s.cumulative_cases for s in results_explicit]
        end

        @testset "Severity efficacy" begin
            @testset "Recorded alongside the other per-dose state" begin
                rv = RingVaccination(efficacy = 0.0, severity_efficacy = 0.4,
                    delay_to_immunity = 5.0)
                contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
                EpiBranch._record_vaccination!(rv, contact, 3.0, StableRNG(1))
                @test contact.state[:severity_efficacy] == 0.4
                @test immunity_time(contact) == 8.0
                @test severity_efficacy(contact) == 0.4
            end

            @testset "Unvaccinated individuals carry no severity protection" begin
                ind = Individual(id = 1, infection_time = 0.0)
                @test severity_efficacy(ind) == 0.0
                @test immunity_time(ind) == Inf
            end

            @testset "Lowers deaths without changing case counts" begin
                # A vaccine with efficacy = 0.0 leaves transmission untouched;
                # severity_efficacy = 1.0 fully protects anyone whose immunity
                # has developed by their own onset from the (otherwise
                # certain) death drawn below.
                iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
                ct = ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Exponential(0.5))
                progression = [Death(delay = 0.0,
                    probability = (rng, ind) -> immunity_time(ind) <= onset_time(ind) ?
                                                1.0 - severity_efficacy(ind) : 1.0)]
                scen(rv) = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical,
                    progression = progression)
                died(ind) = get(ind.state, :outcome, nothing) === :died

                rv_protected = RingVaccination(efficacy = 0.0, severity_efficacy = 1.0,
                    delay_to_immunity = 0.0)
                results_protected = simulate(scen(rv_protected), 50;
                    max_cases = 200, rng = StableRNG(42))

                rv_unprotected = RingVaccination(efficacy = 0.0, severity_efficacy = 0.0,
                    delay_to_immunity = 0.0)
                results_unprotected = simulate(scen(rv_unprotected), 50;
                    max_cases = 200, rng = StableRNG(42))

                # Case counts are bit-identical: severity_efficacy does not
                # touch transmission.
                @test [s.cumulative_cases for s in results_protected] ==
                      [s.cumulative_cases for s in results_unprotected]

                n_vaccinated = sum(count(is_vaccinated, s.individuals)
                for s in results_protected)
                n_died_protected = sum(count(died, s.individuals)
                for s in results_protected)
                n_died_unprotected = sum(count(died, s.individuals)
                for s in results_unprotected)

                @test n_vaccinated > 0  # otherwise the test is vacuous
                @test n_died_protected < n_died_unprotected
                # Anyone whose immunity arrived before their own onset is
                # fully protected (severity_efficacy = 1.0).
                @test all(results_protected) do s
                    all(s.individuals) do ind
                        immunity_time(ind) > onset_time(ind) || !died(ind)
                    end
                end
            end

            @testset "Immunity arriving after the outcome confers no protection" begin
                # delay_to_immunity is long enough that immunity never
                # develops before onset, so severity_efficacy must leave
                # every death exactly as if the dose were never given.
                iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
                ct = ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Exponential(0.5))
                progression = [Death(delay = 0.0,
                    probability = (rng, ind) -> immunity_time(ind) <= onset_time(ind) ?
                                                1.0 - severity_efficacy(ind) : 1.0)]
                scen(rv) = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical,
                    progression = progression)
                died(ind) = get(ind.state, :outcome, nothing) === :died

                rv_late = RingVaccination(efficacy = 0.0, severity_efficacy = 1.0,
                    delay_to_immunity = 1e6)
                results_late = simulate(scen(rv_late), 50;
                    max_cases = 200, rng = StableRNG(42))

                rv_none = RingVaccination(efficacy = 0.0, severity_efficacy = 0.0,
                    delay_to_immunity = 1e6)
                results_none = simulate(scen(rv_none), 50;
                    max_cases = 200, rng = StableRNG(42))

                n_vaccinated = sum(count(is_vaccinated, s.individuals)
                for s in results_late)
                n_died_late = sum(count(died, s.individuals) for s in results_late)
                n_died_none = sum(count(died, s.individuals) for s in results_none)

                @test n_vaccinated > 0  # otherwise the test is vacuous
                @test n_died_late == n_died_none
            end

            @testset "Composes with a per-individual base probability" begin
                # severity_efficacy multiplies whatever base probability the
                # Death transition's `probability` computes, so it composes
                # with age-conditional CFR the same way `efficacy` composes
                # with any other per-individual heterogeneity.
                ind_high = Individual(id = 1, infection_time = 0.0,
                    state = Dict{Symbol, Any}(:onset_time => 5.0, :age => 85))
                ind_low = Individual(id = 2, infection_time = 0.0,
                    state = Dict{Symbol, Any}(:onset_time => 5.0, :age => 20))
                rv = RingVaccination(efficacy = 0.0, severity_efficacy = 0.5,
                    delay_to_immunity = 0.0)
                for ind in (ind_high, ind_low)
                    EpiBranch._record_vaccination!(rv, ind, 0.0, StableRNG(1))
                end
                base(ind) = ind.state[:age] >= 80 ? 0.3 : 0.02
                cfr(ind) = immunity_time(ind) <= onset_time(ind) ?
                           base(ind) * (1 - severity_efficacy(ind)) : base(ind)
                @test cfr(ind_high) ≈ 0.15
                @test cfr(ind_low) ≈ 0.01
            end
        end
    end

    @testset "Contact tracing without quarantine" begin
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            quarantine_on_trace = false)

        state = simulate(
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso, ct], attributes = clinical);
            max_cases = 50, rng = rng)

        n_traced = count(is_traced, state.individuals)
        if n_traced > 0
            # Traced contacts should not be quarantined
            traced = filter(is_traced, state.individuals)
            @test !any(is_quarantined, traced)
        end
    end

    @testset "is_vaccinated respects dose_label namespacing" begin
        ind = Individual(id = 1, state = Dict{Symbol, Any}(:vaccinated_boost => true))
        @test is_vaccinated(ind; dose_label = :boost)
        @test !is_vaccinated(ind)                       # default label is unset
        @test !is_vaccinated(ind; dose_label = :prime)
    end

    @testset "RingVaccination fires under FlagOnly tracing" begin
        # FlagOnly writes :traced_isolation_time, not :isolation_time. Ring
        # vaccination keys on the trace-driven isolation time, so it must still
        # fire (previously it silently no-op'd when tracing only flagged).
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5), quarantine_on_trace = false)
        rv = RingVaccination(efficacy = 0.8)
        state = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso, ct, rv], attributes = clinical);
            max_cases = 300, rng = StableRNG(3))
        # Contacts were flagged (not quarantined) and then ring-vaccinated.
        @test any(ind -> is_traced(ind) && !is_quarantined(ind), state.individuals)
        @test count(is_vaccinated, state.individuals) > 0
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
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso_late], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            rng2 = StableRNG(42)
            iso_always = Isolation(onset_to_isolation_delay = Exponential(1.0))
            results_always = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso_always], attributes = clinical),
                100; max_cases = 200, rng = rng2)

            @test containment_probability(results_always) >=
                  containment_probability(results_late)

            # Fields should still be initialised on all individuals
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso], attributes = clinical),
                100; max_cases = 200, rng = rng1)

            # Compare with always-on isolation — scheduled should contain less
            rng2 = StableRNG(42)
            iso_always = Isolation(onset_to_isolation_delay = Exponential(0.5))
            results_always = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso_always], attributes = clinical),
                100; max_cases = 200, rng = rng2)

            @test containment_probability(results_always) >=
                  containment_probability(results_scheduled)
        end

        @testset "custom predicate" begin
            iso = Scheduled(Isolation(onset_to_isolation_delay = Exponential(1.0)),
                state -> state.current_generation >= 3)

            state = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
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
