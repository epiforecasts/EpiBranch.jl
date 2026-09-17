# Build an Individual carrying the given group and state keys, initialised
# through the vaccination so `:vaccinated`/`:vaccination_time` start unset,
# matching what the engine does for a real contact.
function _group_member(gv, id, group; kwargs...)
    ind = Individual(id = id, state = Dict{Symbol, Any}(:group => group, kwargs...))
    EpiBranch.initialise_individual!(gv, ind, nothing)
    return ind
end

@testset "Group vaccination" begin
    @testset "Reaches every member of a triggered group and nobody outside it" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            dose_delay = 2.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 5.0)
        other_a = _group_member(gv, 2, :A, test_positive = false)
        another_a = _group_member(gv, 3, :A, test_positive = false)
        b1 = _group_member(gv, 4, :B, test_positive = false)
        b2 = _group_member(gv, 5, :B, test_positive = false)
        new_contacts = [confirmed, other_a, another_a, b1, b2]
        append!(state.individuals, new_contacts)

        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        @test is_vaccinated(confirmed)
        @test is_vaccinated(other_a)
        @test is_vaccinated(another_a)
        @test !is_vaccinated(b1)
        @test !is_vaccinated(b2)
    end

    @testset "Vaccination time is the trigger time plus the delay" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            dose_delay = 3.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 6.0)
        member = _group_member(gv, 2, :A, test_positive = false)
        new_contacts = [confirmed, member]
        append!(state.individuals, new_contacts)

        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        # The trigger fires at the confirmed case's isolation time (the
        # default trigger time for a policy that does not override it), so
        # the dose lands 3 days after that.
        @test member.state[:vaccination_time] == 9.0
        @test confirmed.state[:vaccination_time] == 9.0
    end

    @testset "Members created after the trigger are still reached" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            dose_delay = 1.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 4.0)
        first_gen = [confirmed]
        append!(state.individuals, first_gen)
        EpiBranch.apply_post_transmission!(gv, state, first_gen)
        @test is_vaccinated(confirmed)

        # A member of the same group appearing only in a later generation
        # (the trigger already fired) must still be vaccinated, at the same
        # group trigger time.
        latecomer = _group_member(gv, 2, :A, test_positive = false)
        second_gen = [latecomer]
        append!(state.individuals, second_gen)
        EpiBranch.apply_post_transmission!(gv, state, second_gen)

        @test is_vaccinated(latecomer)
        @test latecomer.state[:vaccination_time] == confirmed.state[:vaccination_time]
    end

    @testset "No triggering case leaves the group unvaccinated" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation())
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        members = [_group_member(gv, i, :A, test_positive = false) for i in 1:3]
        append!(state.individuals, members)
        EpiBranch.apply_post_transmission!(gv, state, members)

        @test !any(is_vaccinated, members)
    end

    @testset "Protection blocks only exposures after the trigger plus the delay" begin
        # `competing_risk` is inherited from `AbstractVaccination`'s
        # susceptibility risk, so the event time it reports is what actually
        # gates transmission: an exposure timed before it goes unblocked, one
        # timed after (or at) it is blocked with probability `efficacy`.
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            dose_delay = 2.0, delay_to_immunity = 5.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 10.0)
        member = _group_member(gv, 2, :A, test_positive = false)
        new_contacts = [confirmed, member]
        append!(state.individuals, new_contacts)
        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        # Trigger at 10, dose at 12, immunity at 12 + 5 = 17.
        risk = EpiBranch.competing_risk(gv, confirmed, member, state)
        @test risk.event_time == 17.0
        @test risk.block_probability == member.state[:vaccine_efficacy]
    end

    @testset "Required fields include the group key and the eligibility's own fields" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation())
        fields = EpiBranch.required_fields(gv)
        @test :group in fields
        @test :test_positive in fields

        custom = GroupVaccination(efficacy = 0.9, eligibility = OnSymptomOnset(),
            group_key = :village)
        fields2 = EpiBranch.required_fields(custom)
        @test :village in fields2
        @test :onset_time in fields2
    end

    @testset "Doses scale with group size, ring doses with ring size" begin
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(0.5))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))

        n_group_doses = 0
        n_ring_doses = 0
        for seed in 1:10
            attrs = [clinical, groups(3)]
            gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation())
            state_group = simulate(
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [iso, gv], attributes = attrs);
                max_cases = 60, rng = StableRNG(seed))
            n_group_doses += count(is_vaccinated, state_group.individuals)

            rv = RingVaccination(efficacy = 0.9)
            state_ring = simulate(
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = clinical);
                max_cases = 60, rng = StableRNG(seed))
            n_ring_doses += count(is_vaccinated, state_ring.individuals)
        end

        # Every case triggers its whole (roughly population/3-sized) group,
        # whereas a ring dose only ever reaches directly traced contacts, so
        # the group strategy spends many more doses on the same outbreaks.
        @test n_group_doses > n_ring_doses
    end

    @testset "Fallback composition: ring vaccination first leaves group vaccination to fill the rest" begin
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(0.5))
        ct = ContactTracing(probability = 0.3, isolation_to_trace_delay = Exponential(0.5))
        rv = RingVaccination(efficacy = 0.9)
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation())
        attrs = [clinical, groups(2)]

        n_vaccinated = 0
        for seed in 1:10
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [iso, ct, rv, gv], attributes = attrs);
                max_cases = 60, rng = StableRNG(seed))

            # A dose recorded by the ring is not overwritten by the group pass:
            # every vaccinated individual has a single, well-defined vaccination
            # time, whichever of the two interventions set it.
            for ind in state.individuals
                if is_vaccinated(ind)
                    @test isfinite(ind.state[:vaccination_time])
                end
            end
            n_vaccinated += count(is_vaccinated, state.individuals)
        end
        @test n_vaccinated > 0
    end
end
