using EpiBranch: InterventionAction

function _held_contact(iv, id, trace_time)
    ind = Individual(
        id = id, state = Dict{Symbol, Any}(:traced => true, :trace_time => trace_time)
    )
    EpiBranch.initialise_individual!(iv, ind, nothing)
    return ind
end

# A stay that ends by returning a further action, as the extending guide
# describes for a bed: the place is taken when the visit is admitted and given
# back when that action runs.
struct _Bed2 <: EpiBranch.AbstractIntervention
    stay::Float64
end
EpiBranch.capacity_key(::_Bed2) = :bed
EpiBranch.capacity_time_key(::_Bed2) = :bed_time
function EpiBranch.intervention_actions(b::_Bed2, state, candidates)
    return [
        InterventionAction(
            ind, 1.0 + ind.id,
            function (person, at, st)
                person.state[:bed] = true
                person.state[:bed_time] = at
                return [
                    InterventionAction(
                        person, at + b.stay,
                        (p, t, s) -> EpiBranch.release_capacity!(s, b, p, t)
                    ),
                ]
            end
        ) for ind in candidates if !get(ind.state, :bed, false)
    ]
end

struct _Ward <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_Ward) = :ward_bed
EpiBranch.resource_key(::_Ward, ind, state) = ind.state[:ward]
EpiBranch.intervention_actions(::_Ward, state, candidates) = [
    InterventionAction(ind, 1.0, (p, t, s) -> (p.state[:ward_bed] = true))
        for ind in candidates
]

@testset "DeferRefused under CapacityConstrained" begin
    function run_policy(policy)
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, on_refusal = policy)
        state = empty_state()
        contacts = [_held_contact(rv, i, 0.5 * i) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 0.5
        EpiBranch.apply_actions!(cc, state, contacts)
        return cc, state, contacts
    end

    @testset "The default drops what is refused" begin
        cc, state, contacts = run_policy(DropRefused())
        @test count(is_vaccinated, contacts) == 2
        @test CapacityConstrained(RingVaccination(efficacy = 0.9); budget_per_period = 1.0).on_refusal isa DropRefused
    end

    @testset "Deferral books refused people into later periods" begin
        cc, state, contacts = run_policy(DeferRefused())
        @test count(is_vaccinated, contacts) == 5
        days = sort([contacts[i].state[EpiBranch._vaccination_time_key(:default)] for i in 1:5])
        # Two doses per five-day period, the rest moved to period starts.
        @test count(<(5.0), days) == 2
        @test count(d -> 5.0 <= d < 10.0, days) == 2
        @test count(d -> 10.0 <= d < 15.0, days) == 1
        @test capacity_usage(cc, state, nothing).used <= capacity_usage(cc, state).available
    end

    @testset "A refused person is dropped once max_periods are tried" begin
        cc, state, contacts = run_policy(DeferRefused(max_periods = 1))
        @test count(is_vaccinated, contacts) == 4
    end

    @testset "Deferral needs a finite period and actions it can move" begin
        rv = RingVaccination(efficacy = 0.9)
        @test_throws ArgumentError CapacityConstrained(
            rv; budget_per_period = 1.0, on_refusal = DeferRefused()
        )
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf)
        @test_throws ArgumentError CapacityConstrained(
            iso; budget_per_period = 1.0, period = 1.0, on_refusal = DeferRefused()
        )
        @test_throws ArgumentError DeferRefused(max_periods = 0)
    end
end

@testset "HeldCapacity" begin
    @testset "Caps how many hold a place at once" begin
        rv = RingVaccination(efficacy = 0.9)
        hc = HeldCapacity(rv; capacity = 2, hold = 3.0)
        state = empty_state()
        contacts = [_held_contact(rv, i, Float64(i)) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(hc, state, contacts)

        # Stays [1, 4) and [2, 5) fill both places; the person at day 3 is
        # turned away, and the ones at days 4 and 5 find a place free.
        taken = [c.id for c in contacts if is_vaccinated(c)]
        @test taken == [1, 2, 4, 5]
        @test held_usage(hc, state, 2.5).held == 2
        @test held_usage(hc, state, 4.5).held == 2
        @test all(t -> held_usage(hc, state, t).held <= 2, 0.0:0.25:10.0)
        @test held_usage(hc, state, 100.0).held == 0
        @test held_usage(hc, state, 2.5).capacity == 2
    end

    @testset "The cap holds whatever order actions are admitted in" begin
        rv = RingVaccination(efficacy = 0.9)
        hc = HeldCapacity(rv; capacity = 1, hold = 2.0)
        state = empty_state()
        late = _held_contact(rv, 1, 3.0)
        early = _held_contact(rv, 2, 2.0)
        append!(state.individuals, [late, early])
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(hc, state, [late])
        EpiBranch.apply_actions!(hc, state, [early])
        # [3, 5) is taken first, so [2, 4) overlaps it and is refused.
        @test is_vaccinated(late)
        @test !is_vaccinated(early)
    end

    @testset "DeferRefused waits for the first place to free up" begin
        rv = RingVaccination(efficacy = 0.9)
        hc = HeldCapacity(rv; capacity = 1, hold = 4.0, on_refusal = DeferRefused())
        state = empty_state()
        contacts = [_held_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state.individuals, contacts)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(hc, state, contacts)
        days = [c.state[EpiBranch._vaccination_time_key(:default)] for c in contacts]
        @test days == [1.0, 5.0, 9.0]
        @test all(d -> held_usage(hc, state, d).held == 1, days)
    end

    @testset "An action's own further action gives the place back" begin
        bed = _Bed2(2.0)
        hc = HeldCapacity(bed; capacity = 1)
        state = empty_state()
        people = [Individual(id = i) for i in 1:2]
        append!(state.individuals, people)
        state.max_infection_time = 0.0

        EpiBranch.apply_actions!(hc, state, people[1:1])
        @test held_usage(hc, state, 2.5).held == 1
        # The stay ends at day 4 (admitted for day 2, kept two days).
        @test held_usage(hc, state, 4.5).held == 0

        # Held until released: the second person is turned away while it runs.
        hc2 = HeldCapacity(_Bed2(100.0); capacity = 1)
        s2 = empty_state()
        p2 = [Individual(id = i) for i in 1:2]
        append!(s2.individuals, p2)
        s2.max_infection_time = 0.0
        EpiBranch.apply_actions!(hc2, s2, p2)
        @test count(p -> get(p.state, :bed, false), p2) == 1
    end

    @testset "release_capacity! shortens or withdraws a stay" begin
        rv = RingVaccination(efficacy = 0.9)
        hc = HeldCapacity(rv; capacity = 1)
        state = empty_state()
        a = _held_contact(rv, 1, 1.0)
        b = _held_contact(rv, 2, 6.0)
        append!(state.individuals, [a, b])
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(hc, state, [a])
        EpiBranch.apply_actions!(hc, state, [b])
        @test !is_vaccinated(b)

        EpiBranch.release_capacity!(state, rv, a, 5.0)
        @test held_usage(hc, state, 4.0).held == 1
        @test held_usage(hc, state, 5.0).held == 0
        # A person holding nothing, or released later than their stay ends, is left alone.
        EpiBranch.release_capacity!(state, rv, a, 9.0)
        EpiBranch.release_capacity!(state, rv, b, 9.0)
        @test held_usage(hc, state, 5.0).held == 0
        EpiBranch.release_capacity!(state, rv, a, 0.5)
        @test held_usage(hc, state, 1.0).held == 0
    end

    @testset "A resource_key gives each group its own places" begin
        hc = HeldCapacity(_Ward(); capacity = 1)
        state = empty_state()
        people = [
            Individual(id = 1, state = Dict{Symbol, Any}(:ward => :a)),
            Individual(id = 2, state = Dict{Symbol, Any}(:ward => :a)),
            Individual(id = 3, state = Dict{Symbol, Any}(:ward => :b)),
        ]
        append!(state.individuals, people)
        state.max_infection_time = 0.0
        EpiBranch.apply_actions!(hc, state, people)
        @test [get(p.state, :ward_bed, false) for p in people] == [true, false, true]
        @test held_usage(hc, state, 1.0, :a).held == 1
        @test held_usage(hc, state, 1.0, :b).held == 1
    end

    @testset "Caps how many are quarantined at once" begin
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = 4.0))
        hc = HeldCapacity(ct; capacity = 1, hold = 4.0)
        state = empty_state()
        infector = Individual(id = 1)
        set_isolated!(infector, 0.0; release_time = Inf)
        contacts = [Individual(id = i) for i in 2:4]
        append!(state.individuals, contacts)
        EpiBranch.trace_contacts!(hc, state, infector, contacts, [1.0, 2.0, 5.0])
        @test all(is_traced, contacts)
        @test [is_quarantined(c) for c in contacts] == [true, false, true]

        deferring = HeldCapacity(ct; capacity = 1, hold = 4.0, on_refusal = DeferRefused())
        state = empty_state()
        contacts = [Individual(id = i) for i in 2:4]
        append!(state.individuals, contacts)
        EpiBranch.trace_contacts!(deferring, state, infector, contacts, [1.0, 2.0, 3.0])
        @test all(is_quarantined, contacts)
        @test [isolation_time(c) for c in contacts] == [1.0, 5.0, 9.0]
    end

    @testset "Caps how many are isolated at once" begin
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0), duration = 5.0)
        hc = HeldCapacity(iso; capacity = 1)
        state = empty_state()
        people = [
            Individual(
                id = i, state = Dict{Symbol, Any}(
                    :test_positive => true, :onset_time => t, :asymptomatic => false
                )
            ) for (i, t) in enumerate([1.0, 3.0, 7.0])
        ]
        append!(state.individuals, people)
        for p in people
            EpiBranch.resolve_individual!(hc, p, state)
        end
        # [1, 6) holds the one place, so the case at day 3 is turned away and
        # the case at day 7 finds it free again.
        @test [is_isolated(p) for p in people] == [true, false, true]
        @test held_usage(hc, state, 3.0).held == 1
        @test held_usage(hc, state, 6.5).held == 0
    end

    @testset "A simulation never exceeds the cap" begin
        cap = 2
        model = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(3.0));
            interventions = [
                HeldCapacity(
                    Isolation(onset_to_isolation_delay = Exponential(1.0), duration = 6.0);
                    capacity = cap
                ),
            ],
            attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        )
        for seed in 1:3
            state = simulate(model; max_cases = 60, rng = StableRNG(seed))
            iso = [
                (isolation_time(c), isolation_release_time(c))
                    for c in state.individuals if isfinite(isolation_time(c))
            ]
            @test !isempty(iso)
            for (s, _) in iso
                @test count(((a, b),) -> a <= s < b, iso) <= cap
            end
        end
    end

    @testset "A capacity schedule is checked across the whole stay" begin
        rv = RingVaccination(efficacy = 0.9)
        function run_schedule(capacity, days, hold; policy = DropRefused())
            hc = HeldCapacity(rv; capacity, hold, on_refusal = policy)
            state = empty_state()
            contacts = [_held_contact(rv, i, d) for (i, d) in enumerate(days)]
            append!(state.individuals, contacts)
            state.max_infection_time = 1.0
            EpiBranch.apply_actions!(hc, state, contacts)
            return hc, state, contacts
        end
        key = EpiBranch._vaccination_time_key(:default)

        # Capacity drops to 1 at day 5. The second stay would span the drop
        # with the first still held, so it is refused although a place is free
        # at its start.
        hc, state, contacts = run_schedule(Steps([5.0], [2.0, 1.0]), [1.0, 2.0, 3.0], 6.0)
        @test [c.id for c in contacts if is_vaccinated(c)] == [1]
        @test held_usage(hc, state, 1.5).capacity == 2
        @test held_usage(hc, state, 5.5).capacity == 1

        # Stays that end before the drop are not affected by it, and nothing
        # is admitted on top of a place held at the lower capacity.
        hc, state, contacts = run_schedule(Steps([5.0], [2.0, 1.0]), [1.0, 2.0, 6.0, 7.0], 2.0)
        @test [c.id for c in contacts if is_vaccinated(c)] == [1, 2, 3]
        @test held_usage(hc, state, 2.5).held == 2
        @test all(t -> held_usage(hc, state, t).held <= held_usage(hc, state, t).capacity, 0.0:0.25:12.0)

        # Places that open after a refusal let a deferred action in.
        hc, state, contacts = run_schedule(
            Steps([10.0], [1.0, 3.0]), [1.0, 2.0, 3.0], 20.0; policy = DeferRefused()
        )
        @test [c.state[key] for c in contacts] == [1.0, 10.0, 10.0]

        # DeferRefused still waits for a rise that comes after every stay has
        # ended, or when there are no stays at all.
        hc, state, contacts = run_schedule(
            Steps([10.0], [0.0, 3.0]), [1.0], 2.0; policy = DeferRefused()
        )
        @test [c.state[key] for c in contacts] == [10.0]
        hc, state, contacts = run_schedule(
            Steps([5.0, 10.0], [1.0, 0.0, 3.0]), [1.0, 6.0], 2.0; policy = DeferRefused()
        )
        @test [c.state[key] for c in contacts] == [1.0, 10.0]

        # A rise whose stay would span a later fall is skipped for the next rise.
        hc, state, contacts = run_schedule(
            Steps([10.0, 12.0, 20.0], [0.0, 1.0, 0.0, 3.0]), [1.0], 5.0; policy = DeferRefused()
        )
        @test [c.state[key] for c in contacts] == [20.0]

        @test_throws ArgumentError HeldCapacity(rv; capacity = Seasonal(amplitude = 0.5, peak_day = 0.0))
        @test_throws ArgumentError HeldCapacity(rv; capacity = t -> 2.0)
        @test_throws ArgumentError HeldCapacity(rv; capacity = "two")
    end

    @testset "Constructor validates its arguments" begin
        rv = RingVaccination(efficacy = 0.9)
        @test_throws ArgumentError HeldCapacity(rv; capacity = -1)
        @test_throws ArgumentError HeldCapacity(rv; capacity = 1, hold = -1.0) |>
            x -> EpiBranch.apply_actions!(x, empty_state(), [_held_contact(rv, 1, 1.0)])
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf)
        @test_throws ArgumentError HeldCapacity(iso; capacity = 1, on_refusal = DeferRefused())
    end
end
