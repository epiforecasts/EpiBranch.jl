struct AppointmentAction <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::AppointmentAction) = :attended
EpiBranch.capacity_time_key(::AppointmentAction) = :appointment_time
function EpiBranch.intervention_actions(iv::AppointmentAction, state, candidates)
    [EpiBranch.InterventionAction(ind, ind.state[:appointment_time],
         (person, time, st) -> (person.state[:attended] = true))
     for ind in candidates if !get(ind.state, :attended, false)]
end

@testset "Intervention action admission" begin
    newstate() = EpiBranch.new_state(BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18))
    for wrap in (
        v -> Scheduled(
        CapacityConstrained(v; budget_per_period = 1.0, period = 1.0,
            carry_over = false);
        start_time = 10.0, end_time = 12.0),
        v -> CapacityConstrained(Scheduled(v; start_time = 10.0, end_time = 12.0);
        budget_per_period = 1.0, period = 1.0, carry_over = false))
        state = newstate()
        contacts = [Individual(id = i, state = Dict{Symbol, Any}(:appointment_time => t))
                    for (i, t) in enumerate([9.0, 11.0, 13.0])]
        append!(state.individuals, contacts)
        EpiBranch.apply_actions!(wrap(AppointmentAction()), state, contacts)
        @test [get(i.state, :attended, false) for i in contacts] == [false, true, false]
        @test contacts[2].state[:capacity_admission_time_attended] == 0.0
        @test state.max_infection_time == 0.0
    end

    state = newstate()
    calls = Ref(0)
    rv = RingVaccination(efficacy = (rng, ind) -> (calls[] += 1; 0.8),
        dose_delay = (rng, ind) -> (calls[] += 1; 2.0))
    ind = Individual(id = 1, state = Dict{Symbol, Any}(:traced => true, :trace_time =>
        10.0))
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
    other = Individual(id = 2, state = Dict{Symbol, Any}(:traced => true, :trace_time =>
        1.0))
    EpiBranch.initialise_individual!(declined, other, state)
    push!(state.individuals, other)
    EpiBranch.apply_actions!(declined, state, [other])
    EpiBranch.apply_actions!(declined, state, [other])
    @test calls[] == 3
    @test !is_vaccinated(other)
    @test_throws ArgumentError EpiBranch.apply_actions!(
        Isolation(onset_to_isolation_delay = Exponential(1.0)), state, [other])
end

EpiBranch.continuous_actions(::AppointmentAction) = true
@testset "Continuous action boundaries" begin
    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18))
    append!(state.individuals,
        [Individual(id = i, state = Dict{Symbol, Any}(:appointment_time => t))
         for (i, t) in enumerate([12.0, 12.0, 12.0, 10.0])])
    state.max_infection_time = 11.0
    EpiBranch._apply_continuous_actions!(state, state.individuals[1],
        [AppointmentAction()], [1, 2, 3, 4], [true, false, true, false])
    @test [get(i.state, :attended, false) for i in state.individuals] ==
          [true, true, false, false]
    @test EpiBranch._apply_continuous_actions!(state, state.individuals[1], [],
        [1, 2, 3, 4], [true, false, true, false]) === nothing
    @test !EpiBranch.continuous_actions(MassVaccination(efficacy = 0.8, eligibility_time = 1.0))
    @test EpiBranch.continuous_actions(GroupVaccination(efficacy = 0.8))
    @test EpiBranch.continuous_actions(RingVaccination(efficacy = 0.8))
    @test !EpiBranch.continuous_actions(RingVaccination(efficacy = 0.8, eligibility_window = 2.0))
    @test !EpiBranch.continuous_actions(RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.8))
    @test EpiBranch.continuous_actions(Scheduled(RingVaccination(efficacy = 0.8); start_time = 1.0))
    broken = Scheduled(AppointmentAction(), state -> error("predicate failed"))
    @test_throws ErrorException EpiBranch.apply_actions!(broken, state, [state.individuals[3]])
    @test state.max_infection_time == 11.0
end

@testset "Existing-dose effects require admission" begin
    rv = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0)
    for active in (false, true),
        wrap in (
            v -> Scheduled(v, st -> active),
            v -> Scheduled(CapacityConstrained(v; budget_per_period = 0.0), st -> active),
            v -> CapacityConstrained(Scheduled(v, st -> active); budget_per_period = 0.0))

        state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18))
        ind = Individual(id = 1, infection_time = 1.0,
            state = Dict{Symbol, Any}(:traced => true, :trace_time => 1.0,
                :incubation_period => 10.0, :vaccinated => true, :vaccination_time => 2.0))
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
    process = BranchingProcess(Poisson(10.0), Dirac(20.0))
    vaccine = MassVaccination(efficacy = 1.0, eligibility_time = 10.0)
    wraps = (
        v -> Scheduled(v; start_time = 10.0, end_time = 10.0),
        v -> Scheduled(CapacityConstrained(v; budget_per_period = Inf);
            start_time = 10.0, end_time = 10.0),
        v -> CapacityConstrained(Scheduled(v; start_time = 10.0, end_time = 10.0);
            budget_per_period = Inf))
    for wrap in wraps
        iv = wrap(vaccine)
        state = simulate(ModelSpec(process; interventions = [iv]);
            max_generations = 1, rng = MersenneTwister(1))
        @test count(is_vaccinated, state.individuals) == 9
        @test state.cumulative_cases == 1
        contact = state.individuals[2]
        state.max_infection_time = 30.0
        risk = EpiBranch.competing_risk(iv, state.individuals[1], contact, state)
        @test risk.event_time == 10.0
        @test risk.block_probability == 1.0
    end
    state = EpiBranch.new_state(process,
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18))
    parent = Individual(id = 1)
    contact = Individual(id = 2)
    iv = Scheduled(RecordedProtection(), st -> false)
    @test EpiBranch.competing_risk(iv, parent, contact, state) === nothing
    contact.state[:protected_at] = 10.0
    @test EpiBranch.competing_risk(iv, parent, contact, state).event_time == 10.0
    @test !EpiBranch.persistent_competing_risks(AppointmentAction())
    @test EpiBranch.competing_risk(Scheduled(AppointmentAction(), st -> false),
        parent, contact, state) === nothing
end
