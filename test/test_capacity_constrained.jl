# Build an already-traced contact, ready to be considered for a dose:
# `initialise_individual!` sets the vaccination flags to their unset
# defaults, exactly as the engine does for a real contact.
function _traced_contact(iv, id, trace_time)
    ind = Individual(id = id, state = Dict{Symbol, Any}(
        :traced => true, :trace_time => trace_time))
    EpiBranch.initialise_individual!(iv, ind, nothing)
    return ind
end

@testset "CapacityConstrained" begin
    @testset "Rations doses within a period, earliest trace time first" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, t) for (i, t) in enumerate([3.0, 1.0, 2.0, 4.0])]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)

        vaccinated_times = sort([c.state[:trace_time] for c in contacts if is_vaccinated(c)])
        @test vaccinated_times == [1.0, 2.0]
        @test count(is_vaccinated, contacts) == 2
    end

    @testset "A call within budget is unaffected" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 10.0, period = 5.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 3
    end

    @testset "Budget replenishes with carry_over = true" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = true)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        # First period: 1 dose used out of an allowance of 2, 1 left unused.
        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_post_transmission!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        # Second period (t >= 5): the unused dose from the first period carries
        # over, so 3 (1 leftover + 2 new) are available for the 3 new contacts.
        second = [_traced_contact(rv, i, Float64(i)) for i in 6:8]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_post_transmission!(cc, state, second)
        @test count(is_vaccinated, second) == 3
    end

    @testset "Unused budget is lost with carry_over = false" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = false)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_post_transmission!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        # Second period: only this period's own allowance of 2 is available,
        # the unused dose from the first period does not carry over.
        second = [_traced_contact(rv, i, Float64(i)) for i in 6:8]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_post_transmission!(cc, state, second)
        @test count(is_vaccinated, second) == 2
    end

    @testset "A custom priority function is honoured" begin
        rv = RingVaccination(efficacy = 0.9)
        # Reverse of the default: latest trace time served first.
        cc = CapacityConstrained(rv; budget_per_period = 1.0,
            priority = (ind, state) -> -ind.state[:trace_time])
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)
        @test only(filter(is_vaccinated, contacts)).state[:trace_time] == 3.0
    end

    @testset "capacity_usage reports doses used against doses available" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:4]
        append!(state.individuals, contacts)
        state.max_infection_time = 3.0
        EpiBranch.apply_post_transmission!(cc, state, contacts)

        usage = capacity_usage(cc, state)
        @test usage.used == 2
        @test usage.available == 2.0
    end

    @testset "Other hooks delegate unchanged" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 10.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contact = _traced_contact(rv, 1, 1.0)
        append!(state.individuals, [contact])
        state.max_infection_time = 1.0
        EpiBranch.apply_post_transmission!(cc, state, [contact])

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

    @testset "Continuous-time tracing is not passed through unrationed" begin
        ct = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(1.0))
        cc = CapacityConstrained(ct; budget_per_period = 1.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))
        infector = Individual(id = 1)
        contact = Individual(id = 2)

        @test EpiBranch.traces_contacts(ct)
        @test !EpiBranch.traces_contacts(cc)
        @test EpiBranch.trace_contacts!(cc, state, infector, [contact]) === nothing
        @test EpiBranch.trace_contacts!(cc, state, infector, [contact], [-Inf]) === nothing
        @test !haskey(contact.state, :traced)
    end

    @testset "Group actions ration actual members" begin
        gv = GroupVaccination(efficacy = 0.9)
        @test EpiBranch.capacity_key(gv) == :vaccinated
        for wrap in (v -> CapacityConstrained(v; budget_per_period = 1.0),
            v -> Scheduled(CapacityConstrained(v; budget_per_period = 1.0); start_time = 1.0),
            v -> CapacityConstrained(Scheduled(v; start_time = 1.0); budget_per_period = 1.0))
            iv = wrap(gv)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0)),
                EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))
            for i in 1:3
                ind = Individual(id = i,
                    state = Dict{Symbol, Any}(:group => :A,
                        :test_positive => i == 1, :isolation_time => 2.0))
                EpiBranch.initialise_individual!(gv, ind, state)
                push!(state.individuals, ind)
            end
            EpiBranch.apply_post_transmission!(iv, state, [state.individuals[1]])
            @test count(is_vaccinated, state.individuals) == 1
        end
    end

    @testset "Severity efficacy is recorded only for contacts within the budget" begin
        rv = RingVaccination(efficacy = 0.9, severity_efficacy = 0.5)
        cc = CapacityConstrained(rv; budget_per_period = 1.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        earlier = _traced_contact(rv, 1, 1.0)
        later = _traced_contact(rv, 2, 2.0)
        append!(state.individuals, [earlier, later])
        state.max_infection_time = 2.0
        EpiBranch.apply_post_transmission!(cc, state, [later, earlier])

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

        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        @test_throws ArgumentError EpiBranch.capacity_key(iso)
        @test_throws ArgumentError EpiBranch.capacity_time_key(iso)
    end

    @testset "A dose dated before the period began still uses the call's budget" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 1.0,
            carry_over = false)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, 3.0 + i / 10) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 6.2
        EpiBranch.apply_post_transmission!(cc, state, contacts)

        @test count(is_vaccinated, contacts) == 1
    end

    @testset "A fractional budget is not rounded up" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.5, period = 1.0,
            carry_over = false)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, 5.0) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0
        EpiBranch.apply_post_transmission!(cc, state, contacts)

        @test count(is_vaccinated, contacts) == 2
    end

    @testset "Constructor validates its arguments" begin
        rv = RingVaccination(efficacy = 0.9)
        @test_throws ArgumentError CapacityConstrained(rv; budget_per_period = -1.0)
        @test_throws ArgumentError CapacityConstrained(rv; budget_per_period = 1.0, period = 0.0)
    end

    @testset "Rejected candidates do not use up the budget" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        # The two untraced candidates rank first (earliest trace_time) but
        # RingVaccination skips an untraced contact without using a dose, so
        # the budget must still stretch to the two traced candidates behind
        # them instead of being exhausted on the first two by count alone.
        untraced = [Individual(id = i, state = Dict{Symbol, Any}(
                        :traced => false, :trace_time => Float64(i))) for i in 1:2]
        traced = [_traced_contact(rv, i, Float64(i)) for i in 3:4]
        for ind in untraced
            EpiBranch.initialise_individual!(rv, ind, nothing)
        end
        contacts = vcat(untraced, traced)
        append!(state.individuals, contacts)
        state.max_infection_time = 4.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 2
        @test all(is_vaccinated, traced)
    end

    @testset "An already-dosed re-exposed contact is processed with no budget left" begin
        rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0)
        cc = CapacityConstrained(rv; budget_per_period = 0.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contact = Individual(id = 1,
            infection_time = 0.0,
            state = Dict{Symbol, Any}(
                :traced => true, :trace_time => 0.0, :incubation_period => 10.0,
                :vaccinated => true, :vaccination_time => 1.0))
        append!(state.individuals, [contact])
        state.max_infection_time = 1.0

        EpiBranch.apply_post_transmission!(cc, state, [contact])
        @test contact.state[:infection_aborted_time] == 1.0
    end

    @testset "An unlimited budget does not error" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = Inf)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)
        @test count(is_vaccinated, contacts) == 5
    end

    @testset "priority is evaluated once per candidate" begin
        rv = RingVaccination(efficacy = 0.9)
        calls = Ref(0)
        cc = CapacityConstrained(rv; budget_per_period = 2.0,
            priority = (ind, state) -> (calls[] += 1; ind.state[:trace_time]))
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        contacts = [_traced_contact(rv, i, Float64(i)) for i in 1:5]
        append!(state.individuals, contacts)
        state.max_infection_time = 5.0

        EpiBranch.apply_post_transmission!(cc, state, contacts)
        @test calls[] == 5
    end

    @testset "capacity_usage is scoped to the period with carry_over = false" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 2.0, period = 5.0, carry_over = false)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        first = [_traced_contact(rv, 1, 1.0)]
        append!(state.individuals, first)
        state.max_infection_time = 1.0
        EpiBranch.apply_post_transmission!(cc, state, first)

        second = [_traced_contact(rv, i, Float64(i)) for i in 6:7]
        append!(state.individuals, second)
        state.max_infection_time = 6.0
        EpiBranch.apply_post_transmission!(cc, state, second)

        usage = capacity_usage(cc, state)
        @test usage.available == 2.0
        @test usage.used == 2
    end

    @testset "carry_over = false binds across calls in the same period" begin
        rv = RingVaccination(efficacy = 0.9)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 5.0,
            carry_over = false)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))

        # Both calls are charged to the period [10, 15), while the doses they
        # give are dated from the trace times, well before that period began.
        first = [_traced_contact(rv, 1, 3.1)]
        append!(state.individuals, first)
        state.max_infection_time = 12.0
        EpiBranch.apply_post_transmission!(cc, state, first)
        @test count(is_vaccinated, first) == 1

        second = [_traced_contact(rv, 2, 3.2)]
        append!(state.individuals, second)
        state.max_infection_time = 13.0
        EpiBranch.apply_post_transmission!(cc, state, second)
        @test count(is_vaccinated, second) == 0

        usage = capacity_usage(cc, state)
        @test usage.used == 1
        @test usage.available == 1.0
    end

    @testset "Composes with Scheduled in either order" begin
        rv = RingVaccination(efficacy = 0.9)

        cc1 = CapacityConstrained(Scheduled(rv; start_time = 10.0); budget_per_period = 5.0)
        state1 = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))
        contacts1 = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state1.individuals, contacts1)
        state1.max_infection_time = 5.0
        EpiBranch.apply_post_transmission!(cc1, state1, contacts1)
        @test count(is_vaccinated, contacts1) == 0

        cc2 = Scheduled(CapacityConstrained(rv; budget_per_period = 5.0); start_time = 10.0)
        state2 = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), MersenneTwister(1))
        contacts2 = [_traced_contact(rv, i, Float64(i)) for i in 1:3]
        append!(state2.individuals, contacts2)
        state2.max_infection_time = 5.0
        EpiBranch.apply_post_transmission!(cc2, state2, contacts2)
        @test count(is_vaccinated, contacts2) == 0
    end

    @testset "Fewer doses under a binding capacity constraint than without one" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
        ct = ContactTracing(probability = 0.9, isolation_to_trace_delay = Exponential(1.0))
        rv = RingVaccination(efficacy = 0.8)
        cc = CapacityConstrained(rv; budget_per_period = 1.0, period = 2.0)

        base(interventions) = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(5.0));
            interventions, attributes = clinical)

        state_uncapped = simulate(base([iso, ct, rv]);
            condition = 50:300, max_cases = 300, rng = MersenneTwister(1))
        state_capped = simulate(base([iso, ct, cc]);
            condition = 50:300, max_cases = 300, rng = MersenneTwister(1))

        @test count(is_vaccinated, state_capped.individuals) <
              count(is_vaccinated, state_uncapped.individuals)
    end
end
