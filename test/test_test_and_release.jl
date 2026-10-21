using Distributions
using StableRNGs

@testset "TestAndRelease" begin
    @testset "A non-finite duration is rejected" begin
        @test_throws ArgumentError TestAndRelease(duration = Inf, test_day = 3.0)
    end

    @testset "A negative test shortens the quarantine once the contact's infection settles" begin
        rng = StableRNG(1)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        @test isolation_release_time(contact) == 19.0 # the backstop: 5.0 + 14.0

        # The race settles the contact's infection well after the test day
        # (8.0 = trace time 5.0 + test_day 3.0): not yet infected then, so the
        # test, once granted, reads as negative.
        contact.state[EpiBranch.TEST_RELEASE_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_release_time(contact) == 8.0
        @test isolation_time(contact) == 5.0
    end

    @testset "A contact already infected by the test day keeps the full duration" begin
        rng = StableRNG(2)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        contact.state[EpiBranch.TEST_RELEASE_GRANTED_KEY] = true
        contact.infection_time = 6.0 # infected before the test day, 8.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_release_time(contact) == 19.0
    end

    @testset "An imperfect test can still miss a true infection" begin
        rng = StableRNG(3)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 0.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        contact.state[EpiBranch.TEST_RELEASE_GRANTED_KEY] = true
        contact.infection_time = 6.0 # infected, but sensitivity 0 never detects it
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_release_time(contact) == 8.0
    end

    @testset "A test that is never granted leaves the backstop standing" begin
        rng = StableRNG(4)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng) # not granted
        @test isolation_release_time(contact) == 19.0
    end

    @testset "A contact already settled when the test is granted resolves immediately" begin
        rng = StableRNG(5)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        # Already settled (infected, or found to have escaped infection
        # altogether) before being traced, as a contact tracing can reach
        # through a later infector than the one that actually infected it.
        contact = Individual(id = 1, infection_time = 100.0)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        contact.state[:trace_time] = 5.0
        contact.state[:_settled] = true

        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(6)
        )
        EpiBranch.apply_actions!(ct, state, [contact])
        @test contact.state[EpiBranch.TEST_RELEASE_GRANTED_KEY]
        @test isolation_release_time(contact) == 8.0
    end

    @testset "CapacityConstrained rations the test, not the quarantine itself" begin
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)
        cc = CapacityConstrained(ct; budget_per_period = 0.0)

        contact = Individual(id = 1, infection_time = 100.0)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, StableRNG(7))
        contact.state[:trace_time] = 5.0

        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(8)
        )
        EpiBranch.apply_actions!(cc, state, [contact])
        @test !get(contact.state, EpiBranch.TEST_RELEASE_GRANTED_KEY, false)
        @test isolation_release_time(contact) == 19.0
    end

    @testset "A quarantine merged with another trace is left as it stands" begin
        # A second trace of the same contact appends its own stretch under
        # `QUARANTINE_STRETCHES_KEY`, which can merge with the first; the
        # merged entry no longer matches what the first trace recorded, so
        # the shorten is skipped rather than guessed at.
        rng = StableRNG(9)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng) # [5, 19)
        EpiBranch.apply_trace!(t, contact, nothing, 10.0, rng) # merges into [5, 24)
        @test isolation_release_time(contact) == 24.0

        contact.state[EpiBranch.TEST_RELEASE_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_release_time(contact) == 24.0
    end

    @testset "End to end: every granted test's outcome matches the contact's true infection time" begin
        # On a branching process a traced contact is always a real secondary
        # case (tracing cannot reach one before it exists), so every test
        # here is read against a contact truly infected by the test day — the
        # scenario the issue reports getting wrong, by reading the infection
        # before the race had settled it, is covered by the unit tests above
        # instead, where a contact not yet infected can be built directly.
        # This exercises the dispatch through the full engine: that a
        # contact's test is granted, deferred until settlement where it is
        # not yet known, and resolved to the true outcome either way.
        iso = Isolation(onset_to_isolation_delay = Dirac(1.0), duration = Inf)
        ct = ContactTracing(
            OnSymptomOnset(), 1.0, Dirac(0.5),
            TestAndRelease(duration = 10.0, test_day = 1.0, sensitivity = 1.0)
        )
        clinical = clinical_presentation(
            incubation_period = Dirac(1.0), prob_asymptomatic = 0.0
        )
        model = ModelSpec(
            BranchingProcess(Poisson(4.0), Exponential(1.0));
            attributes = clinical, interventions = [iso, ct]
        )
        state = simulate(model; max_cases = 300, rng = StableRNG(11))

        # Only a resolved test has a decided outcome to check; one granted but
        # never resolved belongs to a contact the run stopped before settling,
        # which is left at its backstop duration (see `TestAndRelease`).
        tested = filter(
            ind -> get(ind.state, EpiBranch.TEST_RELEASE_RESOLVED_KEY, false),
            state.individuals
        )
        @test !isempty(tested)
        for ind in tested
            day = ind.state[EpiBranch.TEST_RELEASE_DAY_KEY]
            @test ind.infection_time <= day
            @test isolation_release_time(ind) > day
        end
    end
end
