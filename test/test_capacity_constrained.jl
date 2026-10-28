# Build an already-traced contact, ready to be considered for a dose:
# `initialise_individual!` sets the vaccination flags to their unset
# defaults, exactly as the engine does for a real contact.
# An intervention whose effect, not its `claim!`, uses up capacity, as the
# extending guide's clinic appointment does; a race runs that effect later.
struct _ClinicVisit <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_ClinicVisit) = :attended
EpiBranch.capacity_time_key(::_ClinicVisit) = :appointment_time
function _attend!(person, time, state)
    person.state[:attended] = true
    person.state[:appointment_time] = time
    return nothing
end
function EpiBranch.intervention_actions(::_ClinicVisit, state, candidates)
    return [
        EpiBranch.InterventionAction(ind, state.max_infection_time + 1.0, _attend!)
            for ind in candidates if !get(ind.state, :attended, false)
    ]
end

# A trace action of its own, with a resource to ration, as the extending
# guide describes one.
struct _Bed <: EpiBranch.TraceAction end
function EpiBranch.apply_trace!(::_Bed, contact, state, trace_time, rng)
    contact.state[:bedded] = true
    contact.state[:bed_time] = trace_time
    return nothing
end
EpiBranch.capacity_key(::_Bed) = :bedded
EpiBranch.capacity_time_key(::_Bed) = :bed_time

function _traced_contact(iv, id, trace_time)
    ind = Individual(
        id = id, state = Dict{Symbol, Any}(
            :traced => true, :trace_time => trace_time
        )
    )
    EpiBranch.initialise_individual!(iv, ind, nothing)
    return ind
end

@testset "CapacityConstrained" begin
    @testset "Rations doses within a period, earliest trace time first" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, t) for (i, t) in enumerate([3.0, 1.0, 2.0, 4.0])]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc, state, contacts)

        vaccinated_times = sort([c.state[:trace_time] for c in contacts if is_vaccinated(c)])
        @test vaccinated_times == [1.0, 2.0]
        @test count(is_vaccinated, contacts) == 2
    end

    @testset "A call within budget is unaffected" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 10.0, period = 5.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 3
    end

    @testset "Budget replenishes with carry_over = true" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = true)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # First period: 1 dose used out of an allowance of 2, 1 left unused.
        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        # Second period (t >= 5): the unused dose from the first period carries
        # over, so 3 (1 leftover + 2 new) are available for the 3 new contacts.
        second = [_traced_contact(rv, i, Float64(i)) for i in 6:8]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_actions!(cc, state, second)
        @test count(is_vaccinated, second) == 3
    end

    @testset "Unused budget is lost with carry_over = false" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = false)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        # Second period: only this period's own allowance of 2 is available,
        # the unused dose from the first period does not carry over.
        second = [_traced_contact(rv, i, Float64(i)) for i in 6:8]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_actions!(cc, state, second)
        @test count(is_vaccinated, second) == 2
    end

    @testset "A custom priority function is honoured" begin
        rv = RingVaccination(efficacy = 0.9)
        # Reverse of the default: latest trace time served first.
        cc = CapacityConstrained(
            rv; budget_per_period = 1.0,
            priority = (ind, state) -> -ind.state[:trace_time]
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc, state, contacts)
        @test only(filter(is_vaccinated, contacts)).state[:trace_time] == 3.0
    end

    @testset "A dose withdrawn for missing its window refunds its capacity" begin
        # `_capacity_usage` rescans ring vaccination's own usage key
        # (`:vaccinated`) on every call, rather than keeping a running total,
        # so withdrawing a dose through `on_infection_settled!` is itself the
        # refund: the next call sees one fewer dose used.
        rv = RingVaccination(efficacy = 0.9, eligibility_window = 2.0)
        cc = CapacityConstrained(rv; budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # A finite window still admits a dose for a pending, not yet exposed
        # contact (`infection_time = NaN`), as on a network or household race.
        first = _traced_contact(rv, 1, 5.0)
        first.infection_time = NaN
        append!(state.individuals, [first])
        state.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc, state, [first])
        @test is_vaccinated(first)

        second = _traced_contact(rv, 2, 6.0)
        second.infection_time = NaN
        append!(state.individuals, [second])
        state.max_infection_time = 6.0
        EpiBranch.apply_actions!(cc, state, [second])
        @test !is_vaccinated(second)

        # The race now learns `first`'s real infection time: its dose, given
        # at day 5, falls outside the 2-day window and is withdrawn.
        first.infection_time = 0.0
        EpiBranch.on_infection_settled!(cc, first, state, state.rng)
        @test !is_vaccinated(first)

        EpiBranch.apply_actions!(cc, state, [second])
        @test is_vaccinated(second)
    end

    @testset "capacity_usage reports doses used against doses available" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:4]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0
        EpiBranch.apply_actions!(cc, state, contacts)

        usage = capacity_usage(cc, state)
        @test usage.used == 2
        @test usage.available == 2.0
    end

    @testset "Other hooks delegate unchanged" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 10.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contact = _traced_contact(rv, 1, 1.0)
        append!(state.individuals, [contact])
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(cc, state, [contact])

        parent = Individual(id = 2)
        risk = EpiBranch.competing_risk(cc, parent, contact, state)
        @test risk !== nothing
        @test EpiBranch.required_fields(cc) == EpiBranch.required_fields(rv)
        @test EpiBranch.is_active(cc, state) == EpiBranch.is_active(rv, state)
        @test EpiBranch.intervention_time(cc, contact) ==
            EpiBranch.intervention_time(rv, contact)
        @test EpiBranch.infectious_removal_time(cc, contact) ==
            EpiBranch.infectious_removal_time(rv, contact)
        @test EpiBranch.keep_active(cc, state, [contact], [true]) ==
            EpiBranch.keep_active(rv, state, [contact], [true])
        @test EpiBranch._unwrap_scheduled(cc) === rv
        @test EpiBranch.resolve_individual!(cc, contact, state) ===
            EpiBranch.resolve_individual!(rv, contact, state)
        @test EpiBranch.reset!(cc, contact) === EpiBranch.reset!(rv, contact)
    end

    @testset "Rations isolation through intervention_time and reset!, like Scheduled" begin
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf)
        cc = CapacityConstrained(iso; budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        earlier = Individual(
            id = 1,
            state = Dict{Symbol, Any}(
                :test_positive => true, :onset_time => 1.0, :asymptomatic => false
            )
        )
        later = Individual(
            id = 2,
            state = Dict{Symbol, Any}(
                :test_positive => true, :onset_time => 2.0, :asymptomatic => false
            )
        )
        append!(state.individuals, [earlier, later])

        EpiBranch.resolve_individual!(cc, earlier, state)
        EpiBranch.resolve_individual!(cc, later, state)

        @test is_isolated(earlier)
        @test !is_isolated(later)
        @test isolation_time(later) == Inf
        @test capacity_usage(cc, state).used == 1
    end

    @testset "A resource_key splits isolation's budget per group" begin
        struct _DistrictIsolation <: EpiBranch.InterventionWrapper
            intervention::Isolation
        end
        EpiBranch.resource_key(::_DistrictIsolation, individual, state) =
            individual.state[:district]

        iso = Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf)
        cc = CapacityConstrained(_DistrictIsolation(iso); budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        a = Individual(
            id = 1,
            state = Dict{Symbol, Any}(
                :district => :A, :test_positive => true, :onset_time => 1.0,
                :asymptomatic => false
            )
        )
        b = Individual(
            id = 2,
            state = Dict{Symbol, Any}(
                :district => :B, :test_positive => true, :onset_time => 1.0,
                :asymptomatic => false
            )
        )
        append!(state.individuals, [a, b])

        EpiBranch.resolve_individual!(cc, b, state)
        EpiBranch.resolve_individual!(cc, a, state)

        @test is_isolated(a)
        @test is_isolated(b)
        @test capacity_usage(cc, state, :A).used == 1
        @test capacity_usage(cc, state, :B).used == 1
    end

    @testset "Rations a quarantine, leaving the trace itself unrationed" begin
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf))
        cc = CapacityConstrained(ct; budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        infector = Individual(id = 1)
        set_isolated!(infector, 0.0; release_time = Inf)
        earlier = Individual(id = 2)
        later = Individual(id = 3)
        append!(state.individuals, [earlier, later])

        @test EpiBranch.traces_contacts(cc)
        EpiBranch.trace_contacts!(cc, state, infector, [earlier, later], [1.0, 2.0])

        # Both are found, and the ring could grow past either; only the
        # earlier one's own place in the budget is actually quarantined.
        @test is_traced(earlier) && is_traced(later)
        @test is_quarantined(earlier)
        @test !is_quarantined(later)
    end

    @testset "A quarantine over budget leaves the contact traced but not removed" begin
        model = ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [
                Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
                CapacityConstrained(
                    ContactTracing(
                        probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
                        action = Quarantine(duration = Inf)
                    );
                    budget_per_period = 1.0
                ),
            ],
            attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        )
        state = simulate(model; max_cases = 20, rng = StableRNG(2))

        traced = filter(is_traced, state.individuals)
        @test !isempty(traced)
        @test count(is_quarantined, traced) < length(traced)
    end

    @testset "Group actions ration actual members" begin
        gv = GroupVaccination(efficacy = 0.9)
        @test EpiBranch.capacity_key(gv) == :vaccinated
        for wrap in (
                v -> CapacityConstrained(v; budget_per_period = 1.0),
                v -> Scheduled(CapacityConstrained(v; budget_per_period = 1.0); start_time = 1.0),
                v -> CapacityConstrained(Scheduled(v; start_time = 1.0); budget_per_period = 1.0),
            )
            iv = wrap(gv)
            state = EpiBranch.new_state(
                BranchingProcess(Poisson(1.0)),
                EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
            )
            for i in 1:3
                ind = Individual(
                    id = i,
                    state = Dict{Symbol, Any}(
                        :group => :A,
                        :test_positive => i == 1, :isolation_time => 2.0
                    )
                )
                EpiBranch.initialise_individual!(gv, ind, state)
                push!(state.individuals, ind)
            end
            EpiBranch.apply_actions!(iv, state, [state.individuals[1]])
            @test count(is_vaccinated, state.individuals) == 1
        end
    end

    @testset "Severity efficacy is recorded only for contacts within the budget" begin
        rv = RingVaccination(efficacy = 0.9, severity_efficacy = 0.5)
        cc = CapacityConstrained(rv; budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        earlier = _traced_contact(rv, 1, 1.0)
        later = _traced_contact(rv, 2, 2.0)
        append!(state.individuals, [earlier, later])
        state.max_infection_time = 2.0
        EpiBranch.apply_actions!(cc, state, [later, earlier])

        @test severity_efficacy(earlier) == 0.5
        @test severity_efficacy(later) == 0.0
    end

    @testset "Capacity keys are read through wrappers and required of others" begin
        mv = MassVaccination(efficacy = 0.9, eligibility_time = 1.0)
        rv = RingVaccination(efficacy = 0.9, dose_label = :prime)
        @test EpiBranch.capacity_key(mv) ==
            EpiBranch.capacity_key(RingVaccination(efficacy = 0.9))
        @test EpiBranch.capacity_time_key(mv) isa Symbol
        @test EpiBranch.capacity_time_key(Scheduled(rv; start_time = 1.0)) ==
            EpiBranch.capacity_time_key(rv)

        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        @test EpiBranch.capacity_key(iso) == :_isolated_by_isolation
        @test EpiBranch.capacity_time_key(iso) == :isolation_time

        quarantining = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = Inf)
        )
        @test EpiBranch.capacity_key(quarantining) == :_quarantined_by_tracing
        @test EpiBranch.capacity_time_key(quarantining) == :trace_time
        @test EpiBranch.capacity_key(Scheduled(quarantining; start_time = 1.0)) ==
            EpiBranch.capacity_key(quarantining)

        # `FlagOnly` has no removal to ration, so it has no capacity key of
        # its own.
        flagging = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            action = FlagOnly()
        )
        @test_throws ArgumentError EpiBranch.capacity_key(flagging)
        @test_throws ArgumentError EpiBranch.capacity_time_key(flagging)
    end

    @testset "A dose's own date decides its period, not the call's clock" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(
            rv; budget_per_period = 1.0, period = 1.0,
            carry_over = false
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # All five doses fall due on day 3, though the call that admits them
        # runs on day 6.2.
        contacts = [_traced_contact(rv, i, 3.0 + i / 10) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 6.2
        EpiBranch.apply_actions!(cc, state, contacts)

        @test count(is_vaccinated, contacts) == 1

        # The admitted dose used day 3's budget, not day 6's.
        state.max_infection_time = 3.1
        @test capacity_usage(cc, state).used == 1
        state.max_infection_time = 6.2
        @test capacity_usage(cc, state).used == 0
    end

    @testset "A day's budget binds across calls admitted on different clocks" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(
            rv; budget_per_period = 1.0, period = 1.0,
            carry_over = false
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # Two different cases are settled on widely separated days, but both
        # propose a dose dated to day 10: the budget for that single day must
        # still hold, however far apart the calls that propose it are.
        first = [_traced_contact(rv, 1, 10.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        second = [_traced_contact(rv, 2, 10.5)]
        append!(state.individuals, second)
        state.max_infection_time = 50.0
        EpiBranch.apply_actions!(cc, state, second)
        @test count(is_vaccinated, second) == 0
    end

    @testset "A fractional budget is not rounded up" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(
            rv; budget_per_period = 2.5, period = 1.0,
            carry_over = false
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, 5.0) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc, state, contacts)

        @test count(is_vaccinated, contacts) == 2
    end

    @testset "Rationing across many calls does not depend on a bystander population" begin
        # Usage is now kept as a running count rather than rescanning
        # `state.individuals` on every call, so a large population nobody
        # ever proposes a dose for must not change what gets rationed.
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        append!(state.individuals, [Individual(id = i) for i in 1:500])

        for day in 1:20
            contact = _traced_contact(rv, 500 + day, Float64(day))
            append!(state.individuals, [contact])
            state.max_infection_time = Float64(day)
            EpiBranch.apply_actions!(cc, state, [contact])
        end

        dosed = filter(is_vaccinated, state.individuals)
        @test length(dosed) == 20
        @test capacity_usage(cc, state).used == 20
    end

    @testset "Two wrappers sharing a dose label share one running budget" begin
        # Both default to dose_label :default, so per the docstring they
        # share one budget: an admission through either must be visible to
        # the other's own running count, not just to a fresh rescan.
        rv_a = RingVaccination(efficacy = 0.9)
        rv_b = RingVaccination(efficacy = 0.5)
        cc_a = CapacityConstrained(rv_a; budget_per_period = 2.0)
        cc_b = CapacityConstrained(rv_b; budget_per_period = 2.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        first = _traced_contact(rv_a, 1, 1.0)
        second = _traced_contact(rv_b, 2, 2.0)
        third = _traced_contact(rv_a, 3, 3.0)
        append!(state.individuals, [first, second, third])
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc_a, state, [first])
        EpiBranch.apply_actions!(cc_b, state, [second])
        EpiBranch.apply_actions!(cc_a, state, [third])

        @test count(is_vaccinated, [first, second, third]) == 2
        @test !is_vaccinated(third)
    end

    @testset "A dose revised to an earlier trigger is charged to its new day" begin
        # A contact reached again from an earlier trace has its dose moved
        # earlier (`may_revise`); the running count must move it too, or
        # the old day stays charged and the new one is not.
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 1.0, carry_over = false)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        revised = _traced_contact(rv, 1, 5.0)
        append!(state.individuals, [revised])
        state.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc, state, [revised])
        @test revised.state[:vaccination_time] == 5.0

        revised.state[:trace_time] = 3.0
        EpiBranch.apply_actions!(cc, state, [revised])
        @test revised.state[:vaccination_time] == 3.0

        same_day = _traced_contact(rv, 2, 3.0)
        freed_day = _traced_contact(rv, 3, 5.0)
        append!(state.individuals, [same_day, freed_day])
        EpiBranch.apply_actions!(cc, state, [same_day])
        EpiBranch.apply_actions!(cc, state, [freed_day])
        @test !is_vaccinated(same_day)
        @test is_vaccinated(freed_day)
    end

    @testset "A dose recorded directly counts against the running budget" begin
        # `record_dose!` is public for a custom `trace_contacts!` to call
        # outside the action protocol; a dose it records must still count
        # once the wrapper's running count has been seeded.
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        first = _traced_contact(rv, 1, 1.0)
        second = _traced_contact(rv, 2, 2.0)
        third = _traced_contact(rv, 3, 3.0)
        append!(state.individuals, [first, second, third])
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc, state, [first])
        EpiBranch.record_dose!(rv, second, 2.0, state)
        EpiBranch.apply_actions!(cc, state, [third])

        @test is_vaccinated(first)
        @test is_vaccinated(second)
        @test !is_vaccinated(third)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "A dose from an unwrapped intervention on the same label counts" begin
        # The docstring promises that doses given by another intervention
        # under the label count against the budget; that other intervention
        # need not be wrapped itself.
        rv = RingVaccination(efficacy = 0.9)
        other = RingVaccination(efficacy = 0.5)
        cc = CapacityConstrained(rv; budget_per_period = 2.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        first = _traced_contact(rv, 1, 1.0)
        second = _traced_contact(other, 2, 2.0)
        third = _traced_contact(rv, 3, 3.0)
        append!(state.individuals, [first, second, third])
        state.max_infection_time = 3.0

        EpiBranch.apply_actions!(cc, state, [first])
        EpiBranch.apply_actions!(other, state, [second])
        EpiBranch.apply_actions!(cc, state, [third])

        @test is_vaccinated(first)
        @test is_vaccinated(second)
        @test !is_vaccinated(third)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "A prior dose on a person created mid-run counts" begin
        # An attribute builder can give a person a dose from before the run;
        # one created after the running count was seeded must still draw on
        # the budget, as `capacity_usage` reports them doing.
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        first = _traced_contact(rv, 1, 1.0)
        append!(state.individuals, [first])
        state.max_infection_time = 3.0
        EpiBranch.apply_actions!(cc, state, [first])

        prior = Individual(
            id = 2, state = Dict{Symbol, Any}(
                :vaccinated => true, :vaccination_time => -10.0,
                :vaccine_efficacy => 0.9
            )
        )
        EpiBranch.initialise_individual!(rv, prior, state)
        append!(state.individuals, [prior])

        third = _traced_contact(rv, 3, 3.0)
        append!(state.individuals, [third])
        EpiBranch.apply_actions!(cc, state, [third])
        @test !is_vaccinated(third)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "An effect a race runs later files its own use" begin
        # Inside a race an admitted action waits on the event queue, so the
        # flag its effect sets is still unset when it is admitted; the use
        # must be filed when the effect runs, as the next admission relies
        # on it.
        cc = CapacityConstrained(_ClinicVisit(); budget_per_period = 2.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        visitors = [Individual(id = i) for i in 1:3]
        append!(state.individuals, visitors)
        state.max_infection_time = 1.0

        deferred = Any[]
        EpiBranch._with_action_sink(state, (action, admitter) -> push!(deferred, (action, admitter))) do
            EpiBranch.apply_actions!(cc, state, visitors[1:2])
        end
        @test length(deferred) == 2
        @test !any(ind -> get(ind.state, :attended, false), visitors)
        for (action, admitter) in deferred
            EpiBranch._fire_action!(admitter, state, action)
        end
        @test count(ind -> get(ind.state, :attended, false), visitors) == 2

        state.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc, state, visitors[3:3])
        @test !get(visitors[3].state, :attended, false)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "A quarantine by an unwrapped tracer counts against the budget" begin
        # Backward tracing shares the quarantine key with the rationed
        # forward tracing, so a source it quarantines uses up the budget.
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf))
        cc = CapacityConstrained(ct; budget_per_period = 2.0)
        st = SourceTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf))
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        infector = Individual(id = 1)
        set_isolated!(infector, 0.0; release_time = Inf)
        forward = Individual(id = 2)
        source = Individual(id = 3)
        case = Individual(id = 4, parent_id = 3, infection_time = 1.0)
        set_isolated!(case, 2.0; release_time = Inf)
        third = Individual(id = 5)
        append!(state.individuals, [infector, forward, source, case, third])
        state.max_infection_time = 3.0

        EpiBranch.trace_contacts!(cc, state, infector, [forward], [1.0])
        EpiBranch.trace_contacts!(st, state, case, [source])
        @test is_quarantined(source)
        @test capacity_usage(cc, state).used == 2

        EpiBranch.trace_contacts!(cc, state, infector, [third], [3.0])
        @test !is_quarantined(third)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "A custom trace action by an unwrapped tracer counts too" begin
        ct = ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), _Bed())
        cc = CapacityConstrained(ct; budget_per_period = 2.0)
        st = SourceTracing(TraceEveryone(), 1.0, Dirac(0.0), _Bed())
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        infector = Individual(id = 1)
        set_isolated!(infector, 0.0; release_time = Inf)
        forward = Individual(id = 2)
        source = Individual(id = 3)
        case = Individual(id = 4, parent_id = 3, infection_time = 1.0)
        set_isolated!(case, 2.0; release_time = Inf)
        third = Individual(id = 5)
        append!(state.individuals, [infector, forward, source, case, third])
        state.max_infection_time = 3.0

        EpiBranch.trace_contacts!(cc, state, infector, [forward], [1.0])
        EpiBranch.trace_contacts!(st, state, case, [source])
        @test get(source.state, :bedded, false)
        @test capacity_usage(cc, state).used == 2

        EpiBranch.trace_contacts!(cc, state, infector, [third], [3.0])
        @test !get(third.state, :bedded, false)
        @test capacity_usage(cc, state).used == 2
    end

    @testset "Constructor validates its arguments" begin
        rv = RingVaccination(efficacy = 0.9)
        @test_throws ArgumentError CapacityConstrained(rv; budget_per_period = -1.0)
        @test_throws ArgumentError CapacityConstrained(rv; budget_per_period = 1.0, period = 0.0)
    end

    @testset "Rejected candidates do not use up the budget" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # The two untraced candidates rank first (earliest trace_time) but
        # RingVaccination skips an untraced contact without using a dose, so
        # the budget must still stretch to the two traced candidates behind
        # them instead of being exhausted on the first two by count alone.
        untraced = [
            Individual(
                id = i, state = Dict{Symbol, Any}(
                    :traced => false, :trace_time => Float64(i)
                )
            ) for i in 1:2
        ]
        traced = [_traced_contact(rv, i, Float64(i)) for i in 3:4]
        for ind in untraced
            EpiBranch.initialise_individual!(rv, ind, nothing)
        end
        contacts = vcat(untraced, traced)
        append!(state.individuals, contacts)
        state.max_infection_time = 4.0

        EpiBranch.apply_actions!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 2
        @test all(is_vaccinated, traced)
    end

    @testset "An already-dosed re-exposed contact is processed with no budget left" begin
        rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0)
        cc = CapacityConstrained(rv; budget_per_period = 0.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contact = Individual(
            id = 1,
            infection_time = 0.0,
            state = Dict{Symbol, Any}(
                :traced => true, :trace_time => 0.0, :incubation_period => 10.0,
                :vaccinated => true, :vaccination_time => 1.0
            )
        )
        append!(state.individuals, [contact])
        state.max_infection_time = 1.0

        EpiBranch.apply_actions!(cc, state, [contact])
        @test contact.state[:infection_aborted_time] == 1.0
    end

    @testset "An unlimited budget does not error" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = Inf)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0

        EpiBranch.apply_actions!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 5
    end

    @testset "priority is evaluated once per candidate" begin
        rv = RingVaccination(efficacy = 0.9)
        calls = Ref(0)
        cc = CapacityConstrained(
            rv; budget_per_period = 2.0,
            priority = (ind, state) -> (calls[] += 1; ind.state[:trace_time])
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0

        EpiBranch.apply_actions!(cc, state, contacts)
        @test calls[] == 5
    end

    @testset "capacity_usage is scoped to the period with carry_over = false" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = false)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_actions!(cc, state, first)

        second = [_traced_contact(rv, i, Float64(i)) for i in 6:7]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_actions!(cc, state, second)

        usage = capacity_usage(cc, state)
        @test usage.available == 2.0
        @test usage.used == 2

        # A dose admitted in a later period does not use an earlier period's
        # budget when admission returns to an earlier clock.
        state.max_infection_time = 3.0
        @test capacity_usage(cc, state).used == 1
    end

    @testset "carry_over = false binds across calls in the same period" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(
            rv; budget_per_period = 1.0, period = 5.0,
            carry_over = false
        )
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )

        # Both doses are dated day 3, inside period [0, 5), though the calls
        # that admit them run on days 12 and 13, well into a later period.
        first = [_traced_contact(rv, 1, 3.1)]
        append!(state.individuals, first)
        state.max_infection_time = 12.0
        EpiBranch.apply_actions!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        second = [_traced_contact(rv, 2, 3.2)]
        append!(state.individuals, second)
        state.max_infection_time = 13.0
        EpiBranch.apply_actions!(cc, state, second)
        @test count(is_vaccinated, second) == 0

        state.max_infection_time = 3.0
        usage = capacity_usage(cc, state)
        @test usage.used == 1
        @test usage.available == 1.0
    end

    @testset "Composes with Scheduled in either order" begin
        rv = RingVaccination(efficacy = 0.9)

        cc1 = CapacityConstrained(Scheduled(rv; start_time = 10.0); budget_per_period = 5.0)
        state1 = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        contacts1 = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state1.individuals, contacts1)
        state1.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc1, state1, contacts1)
        @test count(is_vaccinated, contacts1) == 0

        cc2 = Scheduled(CapacityConstrained(rv; budget_per_period = 5.0); start_time = 10.0)
        state2 = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        contacts2 = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state2.individuals, contacts2)
        state2.max_infection_time = 5.0
        EpiBranch.apply_actions!(cc2, state2, contacts2)
        @test count(is_vaccinated, contacts2) == 0
    end

    @testset "resource_key defaults to nothing, one budget for everyone" begin
        rv = RingVaccination(efficacy = 0.9)
        ind = _traced_contact(rv, 1, 1.0)
        @test EpiBranch.resource_key(rv, ind, nothing) === nothing
    end

    @testset "resource_key splits the budget per group" begin
        struct _GroupedAppointment <: EpiBranch.AbstractIntervention end
        function EpiBranch.intervention_actions(::_GroupedAppointment, state, candidates)
            return [
                EpiBranch.InterventionAction(
                    ind, ind.state[:time],
                    (person, at, st) -> person.state[:attended] = true
                ) for ind in candidates
            ]
        end
        EpiBranch.capacity_key(::_GroupedAppointment) = :attended
        EpiBranch.capacity_time_key(::_GroupedAppointment) = :time
        EpiBranch.resource_key(::_GroupedAppointment, individual, state) =
            individual.state[:district]

        cc = CapacityConstrained(_GroupedAppointment(); budget_per_period = 1.0)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1)
        )
        a1 = Individual(id = 1, state = Dict{Symbol, Any}(:district => :A, :time => 1.0))
        a2 = Individual(id = 2, state = Dict{Symbol, Any}(:district => :A, :time => 2.0))
        b1 = Individual(id = 3, state = Dict{Symbol, Any}(:district => :B, :time => 1.0))
        append!(state.individuals, [a1, a2, b1])
        state.max_infection_time = 2.0

        EpiBranch.apply_actions!(cc, state, [a1, a2, b1])
        @test get(a1.state, :attended, false)
        @test !get(a2.state, :attended, false)
        @test get(b1.state, :attended, false)

        @test capacity_usage(cc, state, :A).used == 1
        @test capacity_usage(cc, state, :B).used == 1
        @test capacity_usage(cc, state, :C).used == 0
    end

    @testset "Fewer doses under a binding capacity constraint than without one" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf)
        ct = ContactTracing(
            probability = 0.9, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = Inf)
        )
        rv = RingVaccination(efficacy = 0.8)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 2.0)

        base(interventions) = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(5.0));
            interventions, attributes = clinical
        )

        state_uncapped = simulate(
            base([iso, ct, rv]);
            condition = 50:300, max_cases = 300, rng = MersenneTwister(1)
        )
        state_capped = simulate(
            base([iso, ct, cc]);
            condition = 50:300, max_cases = 300, rng = MersenneTwister(1)
        )

        @test count(is_vaccinated, state_capped.individuals) <
            count(is_vaccinated, state_uncapped.individuals)
    end
end

@testset "Carried-over capacity holds at every period, whatever order doses come in" begin
    # Cases settle in infection order, so a dose dated earlier is often
    # admitted after later ones; it must still leave every later day within
    # its cumulative allowance.
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf)
    ct = ContactTracing(
        probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
        action = Quarantine(duration = Inf)
    )
    rv = CapacityConstrained(
        RingVaccination(efficacy = 0.9); budget_per_period = 5.0, period = 1.0
    )
    @test rv.carry_over
    over = 0
    for seed in 1:20
        s = simulate(
            ModelSpec(
                BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso, ct, rv], attributes = clinical
            );
            max_cases = 200, rng = StableRNG(seed)
        )
        times = EpiBranch._dose_times(rv, s)
        for day in 1:(ceil(Int, maximum(times; init = 0.0)) + 1)
            count(<(day), times) <= 5 * day || (over += 1)
        end
    end
    @test over == 0
end
