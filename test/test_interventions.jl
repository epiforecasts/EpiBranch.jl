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

        @testset "Doses are timed at the trace, whatever the trace action" begin
            # A ring member is vaccinated when the tracing team reaches
            # them, so `:vaccination_time` is the trace time regardless of
            # what the trace action wrote on their isolation state.
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
            # for contacts with a known onset. Vaccination must not inherit
            # that restriction: an asymptomatic ring member gets a dose.
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
                # boosts a strict subset. Compared within each run, since
                # the coverage draws move the rng stream between them.
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
                # Same instant is allowed: both doses fire at the trace.
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

            @testset "Protects contacts a pre-exposure dose cannot reach" begin
                # `efficacy` asks for immunity before the exposure, which a dose
                # given at the trace never achieves here; `post_exposure_efficacy`
                # asks for immunity before onset, which it often does.
                base = simulate(scen([iso, ct]), 400; max_cases = 200,
                    rng = StableRNG(42))
                pre = simulate(scen([iso, ct, RingVaccination(efficacy = 0.9)]),
                    400; max_cases = 200, rng = StableRNG(42))
                post = simulate(
                    scen([
                        iso, ct, RingVaccination(efficacy = 0.0,
                            post_exposure_efficacy = 0.9)]),
                    400; max_cases = 200, rng = StableRNG(42))

                @test containment_probability(pre) == containment_probability(base)
                @test containment_probability(post) > containment_probability(base)
            end

            @testset "Immunity arriving after onset protects nobody" begin
                # Incubation periods here average about 5 days, so immunity 100
                # days after the trace can never beat an onset.
                slow = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0,
                    delay_to_immunity = 100.0)
                base = simulate(scen([iso, ct]), 200; max_cases = 200,
                    rng = StableRNG(3))
                results = simulate(scen([iso, ct, slow]), 200; max_cases = 200,
                    rng = StableRNG(3))
                @test containment_probability(results) ==
                      containment_probability(base)
            end

            @testset "The risk is immunity racing the contact's onset" begin
                rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.9,
                    delay_to_immunity = 2.0)
                contact = Individual(id = 1, infection_time = 10.0)
                contact.state[:vaccinated] = true
                contact.state[:vaccination_time] = 12.0
                contact.state[:incubation_period] = 6.0

                # Immunity at 14 beats an onset at 16, so the risk's event time
                # falls at or before the exposure the engine is resolving.
                risk = EpiBranch._post_exposure_risk(rv, contact)
                @test risk.event_time <= contact.infection_time
                @test risk.block_probability == 0.9

                # A shorter incubation puts onset at 13, before immunity at 14.
                contact.state[:incubation_period] = 3.0
                @test EpiBranch._post_exposure_risk(rv, contact).event_time >
                      contact.infection_time

                # No onset to race: asymptomatic contacts fall back to needing
                # immunity before the exposure, which at 14 > 10 fails here.
                contact.state[:incubation_period] = NaN
                asymp_risk = EpiBranch._post_exposure_risk(rv, contact)
                @test asymp_risk.event_time == 14.0
                @test asymp_risk.event_time > contact.infection_time

                # Unvaccinated contacts are untouched.
                unvaccinated = Individual(id = 2, infection_time = 10.0)
                unvaccinated.state[:vaccinated] = false
                @test EpiBranch._post_exposure_risk(rv, unvaccinated) === nothing
            end

            @testset "Setting both efficacies warns" begin
                both = RingVaccination(efficacy = 0.9, post_exposure_efficacy = 0.9)
                @test_logs (:warn, r"both `efficacy` and `post_exposure_efficacy`") ModelSpec(
                    process; interventions = [iso, ct, both], attributes = clinical)
                # Either alone is silent.
                @test_logs ModelSpec(process;
                    interventions = [iso, ct,
                        RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.9)],
                    attributes = clinical)
                @test_logs ModelSpec(process;
                    interventions = [iso, ct, RingVaccination(efficacy = 0.9)],
                    attributes = clinical)
            end

            @testset "Every combination of risks is returned" begin
                # The branch ladder in `competing_risk` is written out for
                # inference, so each shape needs exercising — including the
                # three-risk case, which no simulation test reaches.
                function risks(rv; contact_dosed = true, parent_dosed = true,
                        incubation = 6.0)
                    parent = Individual(id = 1, infection_time = 0.0)
                    contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
                    for (ind, dosed) in ((contact, contact_dosed), (parent, parent_dosed))
                        ind.state[:vaccinated] = dosed
                        ind.state[:vaccination_time] = dosed ? 2.0 : Inf
                        ind.state[:vaccine_efficacy] = rv.efficacy
                        ind.state[:incubation_period] = incubation
                    end
                    r = EpiBranch.competing_risk(rv, parent, contact, nothing)
                    r === nothing ? 0 : (r isa EpiBranch.Risk ? 1 : length(r))
                end

                susceptibility_only = RingVaccination(efficacy = 0.5)
                post_only = RingVaccination(efficacy = 0.0,
                    post_exposure_efficacy = 0.5)
                onward_only = RingVaccination(efficacy = 0.0, onward_efficacy = 0.5)
                all_three = RingVaccination(efficacy = 0.5,
                    post_exposure_efficacy = 0.5, onward_efficacy = 0.5)

                @test risks(susceptibility_only; parent_dosed = false) == 1
                @test risks(post_only; parent_dosed = false) == 1
                @test risks(onward_only; contact_dosed = false) == 1
                @test risks(RingVaccination(efficacy = 0.5, onward_efficacy = 0.5)) == 2
                @test risks(RingVaccination(efficacy = 0.0,
                    post_exposure_efficacy = 0.5, onward_efficacy = 0.5)) == 2
                @test risks(RingVaccination(efficacy = 0.5,
                        post_exposure_efficacy = 0.5);
                    parent_dosed = false) == 2
                @test risks(all_three) == 3
                # Nobody dosed: no risk at all.
                @test risks(all_three; contact_dosed = false, parent_dosed = false) == 0
                # An asymptomatic contact still gets the post-exposure risk, at
                # the stricter pre-exposure event time.
                @test risks(post_only; parent_dosed = false, incubation = NaN) == 1
            end

            @testset "Requires an incubation period" begin
                rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.9)
                @test :incubation_period in EpiBranch.required_fields(rv)
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
