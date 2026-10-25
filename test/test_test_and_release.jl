using Distributions
using StableRNGs

# A `TraceAction` built entirely on the seams `ContactTracing` forwards to its
# action (`EpiBranch.intervention_actions`, `EpiBranch.on_infection_settled!`)
# and on `EpiBranch.shorten_quarantine!`, kept here as a fixture rather than in
# `src/`: a worked example of what those seams let an extension do, checked
# against the same expectations a built-in version would have to meet. See
# "A trace action that proposes its own actions" in the extending guide for
# the same policy explained step by step.
struct TestAndRelease <: EpiBranch.TraceAction
    duration::Float64
    test_day::Float64
    sensitivity::Float64

    function TestAndRelease(duration::Float64, test_day::Float64, sensitivity::Float64)
        isfinite(duration) || throw(
            ArgumentError(
                "`TestAndRelease`'s `duration` must be finite, got $duration"
            )
        )
        return new(duration, test_day, sensitivity)
    end
end
function TestAndRelease(; duration, test_day, sensitivity = 1.0)
    return TestAndRelease(Float64(duration), Float64(test_day), Float64(sensitivity))
end

const TAR_DAY_KEY = :test_release_day
const TAR_BACKSTOP_KEY = :test_release_backstop
const TAR_GRANTED_KEY = :test_release_granted
const TAR_RESOLVED_KEY = :test_release_resolved

# A second trace of an already-quarantined contact extends the same
# quarantine (see `Quarantine`), so only the first trace has a single,
# unmerged stretch of its own for a negative test to shorten later.
function EpiBranch.apply_trace!(t::TestAndRelease, contact, state, trace_time, rng)
    first_trace = !get(contact.state, :_quarantined_by_tracing, false)::Bool
    EpiBranch.apply_trace!(Quarantine(duration = t.duration), contact, state, trace_time, rng)
    (isfinite(trace_time) && first_trace) || return nothing
    release = only(EpiBranch.removal_stretches(contact, EpiBranch.QUARANTINE_STRETCHES_KEY))[2]
    contact.state[TAR_BACKSTOP_KEY] = (trace_time, release)
    contact.state[TAR_DAY_KEY] = trace_time + t.test_day
    return nothing
end

# Read `ind.infection_time` only once it is final: called either from
# `intervention_actions` below, for a contact already settled when its test is
# granted, or from `on_infection_settled!`, under its own guarantee that the
# time is now fixed. Idempotent, so it is harmless if both end up calling it.
function _resolve_test!(t::TestAndRelease, ind, rng)
    get(ind.state, TAR_RESOLVED_KEY, false) && return nothing
    ind.state[TAR_RESOLVED_KEY] = true
    day = ind.state[TAR_DAY_KEY]
    positive = ind.infection_time <= day && rand(rng) < t.sensitivity
    positive && return nothing
    start, backstop = ind.state[TAR_BACKSTOP_KEY]
    EpiBranch.shorten_quarantine!(ind, start, backstop, day)
    return nothing
end

# The test itself is proposed as an `InterventionAction` dated at the test
# day, so `CapacityConstrained` can ration it like a dose. Granting it records
# that the contact had a test, not its outcome.
function EpiBranch.intervention_actions(t::TestAndRelease, state, candidates)
    actions = EpiBranch.InterventionAction[]
    for ind in candidates
        haskey(ind.state, TAR_DAY_KEY) || continue
        get(ind.state, TAR_GRANTED_KEY, false) && continue
        day = ind.state[TAR_DAY_KEY]
        effect! = (person, at, st) -> begin
            person.state[TAR_GRANTED_KEY] = true
            EpiBranch.is_settled(st, person) && _resolve_test!(t, person, st.rng)
        end
        push!(actions, EpiBranch.InterventionAction(ind, day, effect!))
    end
    return actions
end

# A contact's infection time cannot be read until the race has settled it; for
# a contact not yet settled when its test was granted, this is where its
# outcome is finally decided.
function EpiBranch.on_infection_settled!(t::TestAndRelease, ind, state, rng)
    get(ind.state, TAR_GRANTED_KEY, false) || return nothing
    _resolve_test!(t, ind, rng)
    return nothing
end

EpiBranch.capacity_key(::TestAndRelease) = TAR_GRANTED_KEY
EpiBranch.capacity_time_key(::TestAndRelease) = TAR_DAY_KEY
EpiBranch.removal_gap_host_times(::TestAndRelease) = (EpiBranch.QUARANTINE_STRETCHES_KEY,)
EpiBranch.removal_gap_host_times_withdrawn(::TestAndRelease) = (EpiBranch.REMOVAL_STRETCHES_KEY,)

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
        contact.state[TAR_GRANTED_KEY] = true
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
        contact.state[TAR_GRANTED_KEY] = true
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
        contact.state[TAR_GRANTED_KEY] = true
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
        @test contact.state[TAR_GRANTED_KEY]
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
        @test !get(contact.state, TAR_GRANTED_KEY, false)
        @test isolation_release_time(contact) == 19.0
    end

    @testset "CapacityConstrained(ContactTracing(...)) still quarantines when it rations the test away" begin
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)
        cc = CapacityConstrained(ct; budget_per_period = 0.0)

        infector = Individual(id = 1)
        EpiBranch.set_isolated!(infector, 5.0; release_time = Inf)
        contact = Individual(id = 2)

        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(14)
        )
        EpiBranch.trace_contacts!(cc, state, infector, [contact])
        @test is_quarantined(contact)
        @test isolation_release_time(contact) == 19.0 # the backstop: 5.0 + 14.0
    end

    @testset "A negative test shortens a quarantine combined with a standing isolation" begin
        # `apply_trace!` folds the quarantine into a standing isolation
        # already in force at the trace (its `_isolation_in_force` branch),
        # so the shortened release has to be redone against that standing
        # isolation rather than written in directly.
        rng = StableRNG(10)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.set_isolated!(contact, 2.0; release_time = 10.0)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        @test isolation_release_time(contact) == 19.0 # the backstop: 5.0 + 14.0

        # The test day (8.0) comes back negative, below the standing
        # isolation's own release (10.0), which keeps the contact removed
        # regardless of the negative test.
        contact.state[TAR_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_time(contact) == 2.0
        @test isolation_release_time(contact) == 10.0
        @test EpiBranch.removal_stretches(contact) == [(2.0, 10.0)]
    end

    @testset "Shortening a quarantine combined with a later standing isolation keeps both" begin
        # The standing isolation starts after the quarantine, so once the
        # negative test shortens the quarantine's own release ahead of it,
        # the two fall apart again: the standing isolation must keep its own
        # release rather than being folded into the quarantine's, or erased
        # by it.
        rng = StableRNG(11)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.set_isolated!(contact, 10.0; release_time = 25.0)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        @test isolation_release_time(contact) == 25.0 # combined with the standing isolation

        contact.state[TAR_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_time(contact) == 10.0
        @test isolation_release_time(contact) == 25.0
        @test EpiBranch.removal_stretches(contact) == [(5.0, 8.0), (10.0, 25.0)]
    end

    @testset "Shortening a quarantine combined with a nested standing isolation keeps it" begin
        # The standing isolation falls entirely inside the quarantine's own
        # span, so the merged entry happens to equal the quarantine's own
        # `(start, old_release)` exactly; shortening it must still leave the
        # standing isolation's own release in place.
        rng = StableRNG(12)
        t = TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.set_isolated!(contact, 10.0; release_time = 15.0)
        EpiBranch.apply_trace!(t, contact, nothing, 5.0, rng)
        @test isolation_release_time(contact) == 19.0 # the backstop: 5.0 + 14.0

        contact.state[TAR_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_time(contact) == 10.0
        @test isolation_release_time(contact) == 15.0
        @test EpiBranch.removal_stretches(contact) == [(5.0, 8.0), (10.0, 15.0)]
    end

    @testset "Shortening a quarantine bridging two standing isolations is left as it stands" begin
        # `apply_trace!` folds an unrelated, already-lapsed isolation on the
        # far side of the standing one into the same combined stretch. That
        # combined stretch no longer matches what shortening the quarantine
        # alone would rebuild, so the shorten is skipped rather than
        # silently misreported.
        rng = StableRNG(13)
        t = TestAndRelease(duration = 9.0, test_day = 3.0, sensitivity = 1.0)
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), t)

        contact = Individual(id = 1, infection_time = NaN)
        EpiBranch.set_isolated!(contact, 0.0; release_time = 2.0)
        EpiBranch.set_isolated!(contact, 8.0; release_time = 10.0)
        EpiBranch.apply_trace!(t, contact, nothing, 2.0, rng)
        @test isolation_release_time(contact) == 11.0 # combined: 2.0 + 9.0
        @test EpiBranch.removal_stretches(contact) == [(0.0, 11.0)] # bridges both

        contact.state[TAR_GRANTED_KEY] = true
        contact.infection_time = 100.0
        EpiBranch.on_infection_settled!(ct, contact, nothing, rng)
        @test isolation_time(contact) == 2.0
        @test isolation_release_time(contact) == 11.0
        @test EpiBranch.removal_stretches(contact) == [(0.0, 11.0)]
        # The quarantine's own record must be left standing too: shortening
        # it alone, while the shared history above stays unmatched, would
        # leave the two keys disagreeing on when the contact was released.
        @test EpiBranch.removal_stretches(contact, EpiBranch.QUARANTINE_STRETCHES_KEY) ==
            [(2.0, 11.0)]
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

        contact.state[TAR_GRANTED_KEY] = true
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
        # which is left at its backstop duration.
        tested = filter(
            ind -> get(ind.state, TAR_RESOLVED_KEY, false),
            state.individuals
        )
        @test !isempty(tested)
        for ind in tested
            day = ind.state[TAR_DAY_KEY]
            @test ind.infection_time <= day
            @test isolation_release_time(ind) > day
        end
    end
end
