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

    @testset "A distributional delay is drawn once per member" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            dose_delay = Uniform(1.0, 4.0), delay_to_immunity = Uniform(5.0, 10.0))
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 6.0)
        members = [_group_member(gv, i, :A, test_positive = false) for i in 2:20]
        new_contacts = [confirmed; members]
        append!(state.individuals, new_contacts)

        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        vacc_times = [m.state[:vaccination_time] for m in new_contacts]
        @test all(t -> 7.0 <= t <= 10.0, vacc_times)
        @test length(unique(vacc_times)) > 1
        for m in new_contacts
            delay = immunity_time(m) - m.state[:vaccination_time]
            @test 5.0 <= delay <= 10.0
        end
    end

    @testset "Records severity efficacy and immunity time on every member" begin
        gv = GroupVaccination(efficacy = 0.0, severity_efficacy = 0.4,
            delay_to_immunity = 5.0, dose_delay = 1.0)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 2.0)
        member = _group_member(gv, 2, :A, test_positive = false)
        new_contacts = [confirmed, member]
        append!(state.individuals, new_contacts)

        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        for ind in new_contacts
            @test severity_efficacy(ind) == 0.4
            @test immunity_time(ind) == 8.0
        end
        @test GroupVaccination(efficacy = 0.9).severity_efficacy == 0.0
    end

    @testset "Waning decays protection against infection but not severity efficacy" begin
        decay(dt) = exp(-dt / 10.0)
        gv = GroupVaccination(efficacy = 0.9, severity_efficacy = 0.4,
            delay_to_immunity = 5.0, dose_delay = 1.0, waning = decay)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 2.0)
        member = _group_member(gv, 2, :A, test_positive = false)
        new_contacts = [confirmed, member]
        append!(state.individuals, new_contacts)

        EpiBranch.apply_post_transmission!(gv, state, new_contacts)

        # Vaccinated at 2 + 1 = 3, immune from 3 + 5 = 8.
        risk = EpiBranch._susceptibility_risk(gv, member)
        @test risk.event_time == 8.0
        for exposure in (8.0, 18.0, 58.0)
            member.infection_time = exposure
            block = EpiBranch._sample_value(
                risk.block_probability, StableRNG(1), nothing, member, nothing)
            @test block ≈ 0.9 * decay(exposure - 8.0)
            @test severity_efficacy(member) == 0.4
        end
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

    @testset "Coverage is drawn once per member, however often the group returns" begin
        gv = GroupVaccination(efficacy = 0.9, eligibility = OnLabConfirmation(),
            coverage = 0.5)
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(2))

        confirmed = _group_member(gv, 1, :A, test_positive = true)
        set_isolated!(confirmed, 5.0)
        members = [_group_member(gv, i, :A, test_positive = false) for i in 2:201]
        append!(state.individuals, [confirmed; members])

        EpiBranch.apply_post_transmission!(gv, state, [confirmed])
        first_round = Set(ind.id for ind in members if is_vaccinated(ind))
        # Half of 200 members, up to the sampling noise of a fair coin.
        @test 70 < length(first_round) < 130

        # The group comes back in each later round, once for every member that
        # turns up among the new contacts. Nobody who declined is asked again.
        for _ in 1:10
            EpiBranch.apply_post_transmission!(gv, state, [members[1]])
        end
        @test Set(ind.id for ind in members if is_vaccinated(ind)) == first_round
    end
end

@testset "Repeat visits composed from coverage and dose delay" begin
    visit_times = [0.0, 7.0, 14.0]
    reach = 0.6
    first_reached = [reach * (1 - reach)^(i - 1) for i in eachindex(visit_times)]
    campaign_reach = sum(first_reached)
    delay = DiscreteNonParametric(visit_times, first_reached ./ campaign_reach)
    willingness = (rng, ind) -> (ind.state[:willing] = rand(rng) < 0.9)
    gv = GroupVaccination(efficacy = 0.8, delay_to_immunity = 2.0,
        coverage = (rng, ind) -> ind.state[:willing] ? campaign_reach : 0.0,
        dose_delay = delay)

    function campaign(seed)
        campaign_state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(5.0)),
            EpiBranch.AbstractClinicalTransition[], willingness, StableRNG(seed))
        EpiBranch.add_individuals!(campaign_state, 4000, [gv];
            setup = (ind, i) -> (ind.state[:group] = :A))
        campaign_state.individuals[1].state[:test_positive] = true
        set_isolated!(campaign_state.individuals[1], 5.0)
        EpiBranch.apply_post_transmission!(gv, campaign_state, campaign_state.individuals)
        return campaign_state
    end

    state = campaign(19)
    for (i, time) in enumerate(visit_times)
        share = count(
            ind -> is_vaccinated(ind) &&
                   ind.state[:vaccination_time] == 5.0 + time,
            state.individuals) / length(state.individuals)
        @test isapprox(share, 0.9 * first_reached[i]; atol = 0.025)
    end
    refused = filter(ind -> !ind.state[:willing], state.individuals)
    absent = filter(ind -> ind.state[:willing] && !is_vaccinated(ind), state.individuals)
    later = filter(ind -> is_vaccinated(ind) && ind.state[:vaccination_time] > 5.0,
        state.individuals)
    @test !isempty(refused)
    @test !isempty(absent)
    @test !isempty(later)
    @test all(ind -> !is_vaccinated(ind), refused)
    @test all(ind -> immunity_time(ind) == ind.state[:vaccination_time] + 2.0, later)

    outcomes(st) = [(ind.state[:willing], is_vaccinated(ind),
                        ind.state[:vaccination_time]) for ind in st.individuals]
    original = outcomes(state)
    @test isequal(outcomes(campaign(19)), original)
    for _ in 1:5
        EpiBranch.apply_post_transmission!(gv, state, state.individuals)
    end
    @test isequal(outcomes(state), original)

    # Another labelled dose can have different willingness without altering this one.
    boost = GroupVaccination(efficacy = 0.8, dose_label = :boost, dose_delay = Dirac(21.0))
    for ind in state.individuals
        EpiBranch.initialise_individual!(boost, ind, state)
    end
    EpiBranch.apply_post_transmission!(boost, state, state.individuals)
    @test all(ind -> is_vaccinated(ind; dose_label = :boost), refused)
    @test isequal(outcomes(state), original)
end
