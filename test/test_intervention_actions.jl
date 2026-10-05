struct AppointmentAction <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::AppointmentAction) = :attended
EpiBranch.capacity_time_key(::AppointmentAction) = :appointment_time
function EpiBranch.intervention_actions(iv::AppointmentAction, state, candidates)
    return [
        EpiBranch.InterventionAction(
            ind, ind.state[:appointment_time],
            (person, time, st) -> (person.state[:attended] = true)
        )
            for ind in candidates if !get(ind.state, :attended, false)
    ]
end

@testset "Intervention action admission" begin
    newstate() = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    for wrap in (
            v -> Scheduled(
                CapacityConstrained(
                    v; budget_per_period = 1.0, period = 1.0,
                    carry_over = false
                );
                start_time = 10.0, end_time = 12.0
            ),
            v -> CapacityConstrained(
                Scheduled(v; start_time = 10.0, end_time = 12.0);
                budget_per_period = 1.0, period = 1.0, carry_over = false
            ),
        )
        state = newstate()
        contacts = [
            Individual(id = i, state = Dict{Symbol, Any}(:appointment_time => t))
                for (i, t) in enumerate([9.0, 11.0, 13.0])
        ]
        append!(state.individuals, contacts)
        EpiBranch.apply_actions!(wrap(AppointmentAction()), state, contacts)
        @test [get(i.state, :attended, false) for i in contacts] == [false, true, false]
        @test contacts[2].state[:capacity_admission_time_attended] == 0.0
        @test state.max_infection_time == 0.0
    end

    state = newstate()
    calls = Ref(0)
    rv = RingVaccination(
        efficacy = (rng, ind) -> (calls[] += 1; 0.8),
        dose_delay = (rng, ind) -> (calls[] += 1; 2.0)
    )
    ind = Individual(
        id = 1, state = Dict{Symbol, Any}(
            :traced => true, :trace_time =>
                10.0
        )
    )
    EpiBranch.initialise_individual!(rv, ind, state)
    push!(state.individuals, ind)
    blocked = CapacityConstrained(rv; budget_per_period = 0.0)
    EpiBranch.apply_actions!(blocked, state, [ind])
    @test calls[] == 1
    @test !is_vaccinated(ind)
    # Earlier discovery reuses the delay until admission fixes the recorded date.
    ind.state[:trace_time] = 8.0
    @test only(EpiBranch.intervention_actions(rv, state, [ind])).time == 10.0
    EpiBranch.apply_actions!(rv, state, [ind])
    @test calls[] == 2
    @test ind.state[:vaccination_time] == 10.0
    ind.state[:trace_time] = 5.0
    EpiBranch.apply_actions!(rv, state, [ind])
    @test calls[] == 2
    @test ind.state[:vaccination_time] == 10.0
    @test !("_intervention_actions" in names(linelist(state)))

    declined = RingVaccination(efficacy = 0.8, coverage = (rng, ind) -> (calls[] += 1; 0.0))
    other = Individual(
        id = 2, state = Dict{Symbol, Any}(
            :traced => true, :trace_time =>
                1.0
        )
    )
    EpiBranch.initialise_individual!(declined, other, state)
    push!(state.individuals, other)
    EpiBranch.apply_actions!(declined, state, [other])
    EpiBranch.apply_actions!(declined, state, [other])
    @test calls[] == 3
    @test !is_vaccinated(other)
    @test_throws ArgumentError EpiBranch.apply_actions!(
        Isolation(onset_to_isolation_delay = Exponential(1.0)), state, [other]
    )
end

@testset "Ring vaccination times a dose only from a recorded isolation" begin
    # A traced contact with no `:trace_time` falls back to its isolation, which
    # counts only when the isolation was recorded as a detection.
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    rv = RingVaccination(efficacy = 0.8, dose_delay = 2.0)
    function traced_contact(id, unrecorded)
        ind = Individual(id = id, state = Dict{Symbol, Any}(:traced => true))
        EpiBranch.initialise_individual!(rv, ind, state)
        set_isolated!(ind, 5.0)
        unrecorded && (ind.state[:_isolation_unrecorded] = true)
        push!(state.individuals, ind)
        return ind
    end

    unrecorded = traced_contact(1, true)
    @test isempty(EpiBranch.intervention_actions(rv, state, [unrecorded]))

    recorded = traced_contact(2, false)
    @test only(EpiBranch.intervention_actions(rv, state, [recorded])).time == 7.0
end

EpiBranch.continuous_actions(::AppointmentAction) = true
@testset "Continuous action boundaries" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    append!(
        state.individuals,
        [
            Individual(id = i, state = Dict{Symbol, Any}(:appointment_time => t))
                for (i, t) in enumerate([12.0, 12.0, 12.0, 10.0])
        ]
    )
    state.max_infection_time = 11.0
    EpiBranch._apply_continuous_actions!(
        state, state.individuals[1],
        [AppointmentAction()], [1, 2, 3, 4], [true, false, true, false]
    )
    @test [get(i.state, :attended, false) for i in state.individuals] ==
        [true, true, false, false]
    @test EpiBranch._apply_continuous_actions!(
        state, state.individuals[1], [],
        [1, 2, 3, 4], [true, false, true, false]
    ) === nothing
    @test !EpiBranch.continuous_actions(MassVaccination(efficacy = 0.8, eligibility_time = 1.0))
    @test EpiBranch.continuous_actions(GroupVaccination(efficacy = 0.8))
    @test EpiBranch.continuous_actions(RingVaccination(efficacy = 0.8))
    @test !EpiBranch.continuous_actions(RingVaccination(efficacy = 0.8, eligibility_window = 2.0))
    @test EpiBranch.continuous_actions(RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.8))
    @test EpiBranch.continuous_actions(Scheduled(RingVaccination(efficacy = 0.8); start_time = 1.0))
    broken = Scheduled(AppointmentAction(), state -> error("predicate failed"))
    @test_throws ErrorException EpiBranch.apply_actions!(broken, state, [state.individuals[3]])
    @test state.max_infection_time == 11.0
end

@testset "A dose's revise-earlier policy is a dispatched trait, not a branch in the body" begin
    @test !EpiBranch.may_revise(AppointmentAction(), 20.0, 5.0)
    @test !EpiBranch.may_revise(RingVaccination(efficacy = 0.8), 20.0, 5.0)
    gv = GroupVaccination(efficacy = 0.8)
    @test EpiBranch.may_revise(gv, 20.0, 5.0)
    @test !EpiBranch.may_revise(gv, 5.0, 20.0)
    @test !EpiBranch.may_revise(gv, 5.0, 5.0)
    @test EpiBranch.may_revise(Scheduled(gv; start_time = 1.0), 20.0, 5.0)
end

@testset "is_settled reflects a continuous-time race's own round of discovery" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    append!(
        state.individuals,
        [Individual(id = i, state = Dict{Symbol, Any}(:appointment_time => 1.0)) for i in 1:2]
    )
    current, other = state.individuals
    @test !EpiBranch.is_settled(state, current)
    @test !EpiBranch.is_settled(state, other)
    EpiBranch._apply_continuous_actions!(
        state, current, [AppointmentAction()], [1, 2], [true, false]
    )
    # `current`'s own round is done, so it is now settled; `other`, never the
    # current case of a round, is still pending.
    @test EpiBranch.is_settled(state, current)
    @test !EpiBranch.is_settled(state, other)
end

@testset "Continuous-time candidates scale with the ring or group, not the population" begin
    # A settled case's own ring or group is tiny; most of the population is
    # neither traced by it nor in its group, and must never be materialised
    # as a candidate just because it is still pending.
    n = 500
    members = collect(1:n)
    processed = falses(n)
    pos = Dict(id => k for (k, id) in enumerate(members))
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    append!(state.individuals, [Individual(id = i) for i in 1:n])
    current = state.individuals[1]

    rv = RingVaccination(efficacy = 0.8)
    contacts = (id, st) -> id == 1 ? (2, 3) : ()
    ring_candidates = EpiBranch._continuous_candidates(
        rv, state, current, members, processed, contacts, pos
    )
    @test length(ring_candidates) == 3
    @test Set(ind.id for ind in ring_candidates) == Set([1, 2, 3])

    gv = GroupVaccination(efficacy = 0.8)
    for i in 1:3
        state.individuals[i].state[:group] = :A
    end
    for i in 4:n
        state.individuals[i].state[:group] = i
    end
    group_candidates = EpiBranch._continuous_candidates(
        gv, state, current, members, processed, contacts, pos
    )
    @test length(group_candidates) == 3
    @test Set(ind.id for ind in group_candidates) == Set([1, 2, 3])

    # The default candidate strategy, used by any other custom
    # `continuous_actions` intervention, keeps the wider (expensive)
    # contract every still-pending member.
    generic_candidates = EpiBranch._continuous_candidates(
        AppointmentAction(), state, current, members, processed, contacts, pos
    )
    @test length(generic_candidates) == n
end

@testset "Existing-dose effects require admission" begin
    rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0)
    for active in (false, true),
            wrap in (
                v -> Scheduled(v, st -> active),
                v -> Scheduled(CapacityConstrained(v; budget_per_period = 0.0), st -> active),
                v -> CapacityConstrained(Scheduled(v, st -> active); budget_per_period = 0.0),
            )

        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
        )
        ind = Individual(
            id = 1, infection_time = 1.0,
            state = Dict{Symbol, Any}(
                :traced => true, :trace_time => 1.0,
                :incubation_period => 10.0, :vaccinated => true, :vaccination_time => 2.0
            )
        )
        push!(state.individuals, ind)
        state.max_infection_time = 3.0
        iv = wrap(rv)
        actions = EpiBranch.intervention_actions(iv, state, [ind])
        @test !haskey(ind.state, :infection_aborted_time)
        @test only(actions).time == 3.0
        EpiBranch.apply_actions!(iv, state, [ind])
        @test get(ind.state, :infection_aborted_time, Inf) == (active ? 2.0 : Inf)
        @test ind.state[:vaccination_time] == 2.0
        @test !haskey(ind.state, :capacity_admission_time_vaccinated)
    end
end

struct RecordedProtection <: EpiBranch.AbstractIntervention end
EpiBranch.persistent_competing_risks(::RecordedProtection) = true
function EpiBranch.competing_risk(::RecordedProtection, parent, contact, state)
    haskey(contact.state, :protected_at) || return nothing
    return Risk(event_time = contact.state[:protected_at], block_probability = 1.0)
end

@testset "Recorded effects persist beyond delivery admission" begin
    process = BranchingProcess(Dirac(9), Dirac(20.0))
    vaccine = MassVaccination(efficacy = 1.0, eligibility_time = 10.0)
    wraps = (
        v -> Scheduled(v; start_time = 10.0, end_time = 10.0),
        v -> Scheduled(
            CapacityConstrained(v; budget_per_period = Inf);
            start_time = 10.0, end_time = 10.0
        ),
        v -> CapacityConstrained(
            Scheduled(v; start_time = 10.0, end_time = 10.0);
            budget_per_period = Inf
        ),
    )
    for wrap in wraps
        iv = wrap(vaccine)
        state = simulate(
            ModelSpec(process; interventions = [iv]);
            max_generations = 1, rng = MersenneTwister(1)
        )
        @test count(is_vaccinated, state.individuals) == 9
        @test state.cumulative_cases == 1
        contact = state.individuals[2]
        state.max_infection_time = 30.0
        risk = EpiBranch.competing_risk(iv, state.individuals[1], contact, state)
        @test risk.event_time == 10.0
        @test risk.block_probability == 1.0
    end
    state = EpiBranch.new_state(
        process,
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    parent = Individual(id = 1)
    contact = Individual(id = 2)
    iv = Scheduled(RecordedProtection(), st -> false)
    @test EpiBranch.competing_risk(iv, parent, contact, state) === nothing
    contact.state[:protected_at] = 10.0
    @test EpiBranch.competing_risk(iv, parent, contact, state).event_time == 10.0
    @test !EpiBranch.persistent_competing_risks(AppointmentAction())
    @test EpiBranch.competing_risk(
        Scheduled(AppointmentAction(), st -> false),
        parent, contact, state
    ) === nothing
end

struct LegacyService <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::LegacyService) = :served
EpiBranch.capacity_time_key(::LegacyService) = :service_time
function EpiBranch.apply_post_transmission!(::LegacyService, state, candidates)
    for ind in candidates
        ind.state[:visits] = get(ind.state, :visits, 0) + 1
        get(ind.state, :served, false) && continue
        get(ind.state, :eligible, true) || continue
        ind.state[:served] = true
        ind.state[:service_time] = state.max_infection_time + 10.0
    end
    return
end

@testset "Legacy batch admission remains supported" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    contacts = [Individual(id = i) for i in 1:5]
    contacts[1].state[:eligible] = false
    contacts[5].state[:served] = true
    contacts[5].state[:service_time] = 0.0
    append!(state.individuals, contacts)
    cc = CapacityConstrained(
        LegacyService(); budget_per_period = 2.0,
        period = 10.0, carry_over = false
    )
    EpiBranch.apply_post_transmission!(cc, state, contacts)
    @test [get(i.state, :served, false) for i in contacts] ==
        [false, true, false, false, true]
    @test contacts[1].state[:visits] == 1
    @test contacts[5].state[:visits] == 1
    @test contacts[2].state[:service_time] == 10.0
    @test capacity_usage(cc, state) == (used = 2, available = 2.0)
    state.max_infection_time = 10.0
    @test capacity_usage(cc, state) == (used = 0, available = 2.0)
    EpiBranch.apply_post_transmission!(cc, state, contacts[3:4])
    @test all(i -> i.state[:served], contacts[2:5])
    @test contacts[3].state[:service_time] == 20.0
    @test contacts[3].state[:capacity_admission_time_served] == 10.0
    @test capacity_usage(cc, state) == (used = 2, available = 2.0)
    EpiBranch.apply_post_transmission!(cc, state, contacts[3:4])
    @test contacts[3].state[:visits] == 2
    @test capacity_usage(cc, state) == (used = 2, available = 2.0)
end

@testset "A dose recorded while pending is reconsidered once infection settles" begin
    # `on_infection_settled!` is what the continuous-time race calls on every
    # intervention straight after stamping an infection time. The household and
    # network suites run the whole path; here each way out of
    # `RingVaccination`'s method is driven directly, and the default is checked
    # to be a no-op so the seam carries the dispatch.
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(5)
    )
    rv = RingVaccination(
        efficacy = 0.0, post_exposure_efficacy = 1.0, delay_to_immunity = 1.0
    )
    function pending(; vaccinated = true, incubation = 5.0)
        ind = Individual(
            id = 1,
            state = Dict{Symbol, Any}(
                :vaccinated => vaccinated,
                :vaccination_time => 2.0,
                :incubation_period => incubation
            )
        )
        ind.infection_time = 2.0
        return ind
    end

    # Immunity at 3.0 falls between the exposure at 2.0 and the onset at 7.0,
    # and an efficacy of 1 covers everyone, so the infection ends at 3.0 and
    # never reaches onset.
    aborted = pending()
    EpiBranch.on_infection_settled!(rv, aborted, state, StableRNG(1))
    @test aborted.state[:infection_aborted_time] == 3.0
    @test isnan(onset_time(aborted))

    # Immunity at 3.0 falls after an onset at 2.5, too late to abort anything.
    late = pending(incubation = 0.5)
    EpiBranch.on_infection_settled!(rv, late, state, StableRNG(1))
    @test !haskey(late.state, :infection_aborted_time)

    # No dose on the member: nothing to reconsider.
    undosed = pending(vaccinated = false)
    EpiBranch.on_infection_settled!(rv, undosed, state, StableRNG(1))
    @test !haskey(undosed.state, :infection_aborted_time)

    # A dose with no post-exposure efficacy returns before its time is read.
    plain = pending()
    EpiBranch.on_infection_settled!(
        RingVaccination(efficacy = 0.8), plain, state, StableRNG(1)
    )
    @test !haskey(plain.state, :infection_aborted_time)

    # An intervention with no method of its own gets the no-op default, so the
    # engine needs no knowledge of which types take part.
    for iv in (
            Isolation(onset_to_isolation_delay = Dirac(0.0)),
            ContactTracing(OnIsolation(), 1.0, Dirac(0.0)),
            AppointmentAction(),
        )
        untouched = pending()
        @test EpiBranch.on_infection_settled!(iv, untouched, state, StableRNG(1)) ===
            nothing
        @test !haskey(untouched.state, :infection_aborted_time)
    end

    # A wrapper delegates, so a scheduled dose aborts too. The protection comes
    # from the recorded dose, so a window that has closed does not withdraw it.
    wrapped = pending()
    EpiBranch.on_infection_settled!(
        Scheduled(rv; end_time = 0.0), wrapped, state, StableRNG(1)
    )
    @test wrapped.state[:infection_aborted_time] == 3.0
end
