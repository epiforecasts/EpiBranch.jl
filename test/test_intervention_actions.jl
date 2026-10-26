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

# Proposes one action dated well into the future, whose effect reads the
# clock it actually runs on (`state.max_infection_time`) rather than the one
# in force when it was admitted, and returns a further action of its own.
struct DeferredCheckIn <: EpiBranch.AbstractIntervention end
function EpiBranch.intervention_actions(::DeferredCheckIn, state, candidates)
    return [
        EpiBranch.InterventionAction(
            ind, 10.0,
            (person, at, st) -> begin
                person.state[:seen_max_infection_time] = st.max_infection_time
                return [
                    EpiBranch.InterventionAction(
                        person, at + 1.0,
                        (follow, at2, st2) -> (follow.state[:followed_up] = true)
                    ),
                ]
            end
        )
            for ind in candidates if !haskey(ind.state, :seen_max_infection_time)
    ]
end

# An intervention with an `action_candidates` method of its own, narrowing its
# candidates to a single, fixed person rather than taking the default's
# "current plus the contacts it can reach".
struct SoleTargetAction <: EpiBranch.AbstractIntervention
    target::Int
end
function EpiBranch.action_candidates(
        iv::SoleTargetAction, state, current, traced, is_pending, members, contacts
    )
    return [state.individuals[iv.target]]
end

# A `start_after` trigger written outside the package, gating each
# `InterventionAction` by a group count carried on the action's own
# individual rather than by the population as a whole.
struct _DistrictTrigger <: EpiBranch.AbstractTrigger
    threshold::Int
end
EpiBranch.is_triggered!(::_DistrictTrigger, state) = true
function EpiBranch.is_triggered!(t::_DistrictTrigger, state, action)
    return get(action.individual.state, :district_cases, 0) >= t.threshold
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
        iv = wrap(AppointmentAction())
        EpiBranch.apply_actions!(iv, state, contacts)
        @test [get(i.state, :attended, false) for i in contacts] == [false, true, false]
        @test state.max_infection_time == 0.0

        # The admitted appointment used the budget of day 11, its own date,
        # though the call that admitted it ran on day 0.
        cc = iv isa CapacityConstrained ? iv : iv.intervention
        state.max_infection_time = 11.0
        @test capacity_usage(cc, state).used == 1
        state.max_infection_time = 0.0
        @test capacity_usage(cc, state).used == 0
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
    # A trace found earlier than the admitted dose's own trigger moves it,
    # reusing the cached delay rather than drawing a fresh one.
    ind.state[:trace_time] = 5.0
    EpiBranch.apply_actions!(rv, state, [ind])
    @test calls[] == 2
    @test ind.state[:vaccination_time] == 7.0
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
        Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf), state, [other]
    )
end

@testset "a start_after trigger can read the action it gates" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(18)
    )
    low, high = (
        Individual(
            id = i, state = Dict{Symbol, Any}(
                :appointment_time => 1.0, :district_cases => cases
            )
        )
            for (i, cases) in enumerate((2, 9))
    )
    append!(state.individuals, [low, high])
    scheduled = Scheduled(AppointmentAction(); start_after = _DistrictTrigger(5))

    EpiBranch.apply_actions!(scheduled, state, [low, high])
    @test !get(low.state, :attended, false)
    @test get(high.state, :attended, false)

    # The two-argument form still answers call sites with no action to offer,
    # such as `resolve_individual!`'s own gate.
    @test EpiBranch.is_active(scheduled, state)
end

@testset "A pending contact's ring dose moves earlier when a later infector traces it sooner" begin
    # On a continuous-time race, infectors settle in order of infection, not
    # of isolation: one that is infected later can still isolate, and so
    # trace a shared contact earlier than one infected before it. A dose
    # already given from the first trace found must move to a genuinely
    # earlier one found afterwards, for a contact still pending its own
    # infection.
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    rv = RingVaccination(efficacy = 0.9, dose_delay = 1.0, delay_to_immunity = 3.0)
    a = Individual(id = 1, state = Dict{Symbol, Any}())
    b = Individual(id = 2, state = Dict{Symbol, Any}())
    ind = Individual(id = 3, state = Dict{Symbol, Any}(:traced => true))
    EpiBranch.initialise_individual!(rv, ind, state)
    append!(state.individuals, [a, b, ind])
    members = [1, 2, 3]
    processed = falses(3)
    # The walk's reach is passed as `traced`, so `contacts` only stands in for
    # a model that names contacts, which ring vaccination needs to be offered.
    contacts(id, state) = Int[]

    # `a` settles first, tracing `ind` late.
    ind.state[:trace_time] = 20.0
    processed[1] = true
    EpiBranch._apply_case_actions!(state, a, [rv], members, processed, contacts, nothing, [ind.id])
    @test ind.state[:vaccination_time] == 21.0

    # `b` settles next, despite having isolated, and so traced `ind`, earlier.
    ind.state[:trace_time] = 5.0
    processed[2] = true
    EpiBranch._apply_case_actions!(state, b, [rv], members, processed, contacts, nothing, [ind.id])
    @test ind.state[:vaccination_time] == 6.0
    @test immunity_time(ind) == 9.0
end

@testset "An earlier re-trace does not move a dose before the dose it requires" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    rv = RingVaccination(
        efficacy = 0.9, dose_delay = 1.0, requires_dose = :prime, dose_label = :boost
    )
    a = Individual(id = 1, state = Dict{Symbol, Any}())
    b = Individual(id = 2, state = Dict{Symbol, Any}())
    ind = Individual(id = 3, state = Dict{Symbol, Any}(:traced => true))
    ind.state[EpiBranch._vaccinated_key(:prime)] = true
    ind.state[EpiBranch._vaccination_time_key(:prime)] = 10.0
    EpiBranch.initialise_individual!(rv, ind, state)
    append!(state.individuals, [a, b, ind])
    members = [1, 2, 3]
    processed = falses(3)
    contacts(id, state) = Int[]
    boost_time = EpiBranch._vaccination_time_key(:boost)

    ind.state[:trace_time] = 20.0
    processed[1] = true
    EpiBranch._apply_case_actions!(state, a, [rv], members, processed, contacts, nothing, [ind.id])
    @test ind.state[boost_time] == 21.0

    # A trace at 5 would date the booster at 6, before the prime at 10.
    ind.state[:trace_time] = 5.0
    processed[2] = true
    EpiBranch._apply_case_actions!(state, b, [rv], members, processed, contacts, nothing, [ind.id])
    @test ind.state[boost_time] == 21.0
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
        set_isolated!(ind, 5.0; release_time = Inf)
        unrecorded && (ind.state[:_isolation_unrecorded] = true)
        push!(state.individuals, ind)
        return ind
    end

    unrecorded = traced_contact(1, true)
    @test isempty(EpiBranch.intervention_actions(rv, state, [unrecorded]))

    recorded = traced_contact(2, false)
    @test only(EpiBranch.intervention_actions(rv, state, [recorded])).time == 7.0
end

# Offered `current` and every member not yet settled, so the boundaries below
# are drawn by settlement and timing alone.
function EpiBranch.action_candidates(
        ::AppointmentAction, state, current, traced, is_pending, members, contacts
    )
    return [state.individuals[id] for id in members if id == current.id || is_pending(id)]
end
@testset "Per-case action boundaries" begin
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
    # `AppointmentAction` needs no named contacts (its candidates are
    # `current` and everyone not yet processed), so a `contacts` function is
    # passed only to stand in for a model that supplies one;
    # `needs_contacts`'s conservative default would otherwise skip it on a
    # call with none, as a model without named contacts does on the race.
    contacts(id, state) = Int[]
    EpiBranch._apply_case_actions!(
        state, state.individuals[1],
        [AppointmentAction()], [1, 2, 3, 4], [true, false, true, false], contacts
    )
    @test [get(i.state, :attended, false) for i in state.individuals] ==
        [true, true, false, false]
    @test EpiBranch._apply_case_actions!(
        state, state.individuals[1], [],
        [1, 2, 3, 4], [true, false, true, false], contacts
    ) === nothing
    @test EpiBranch.acts_on_population(MassVaccination(efficacy = 0.8, eligibility_time = 1.0))
    @test !EpiBranch.acts_on_population(GroupVaccination(efficacy = 0.8))
    @test !EpiBranch.acts_on_population(RingVaccination(efficacy = 0.8))
    @test EpiBranch.acts_on_population(
        Scheduled(MassVaccination(efficacy = 0.8, eligibility_time = 1.0); start_time = 1.0)
    )
    @test !EpiBranch.needs_contacts(GroupVaccination(efficacy = 0.8))
    @test EpiBranch.needs_contacts(RingVaccination(efficacy = 0.8))
    @test EpiBranch.needs_contacts(AppointmentAction())
    @test !EpiBranch.needs_contacts(
        Scheduled(GroupVaccination(efficacy = 0.8); start_time = 1.0)
    )
    broken = Scheduled(AppointmentAction(), state -> error("predicate failed"))
    @test_throws ErrorException EpiBranch.apply_actions!(broken, state, [state.individuals[3]])
    @test state.max_infection_time == 11.0
end

@testset "A dose's revise-earlier policy is a dispatched trait, not a branch in the body" begin
    @test !EpiBranch.may_revise(AppointmentAction(), 20.0, 5.0)
    rv = RingVaccination(efficacy = 0.8)
    @test EpiBranch.may_revise(rv, 20.0, 5.0)
    @test !EpiBranch.may_revise(rv, 5.0, 20.0)
    @test !EpiBranch.may_revise(rv, 5.0, 5.0)
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
    EpiBranch._apply_case_actions!(
        state, current, [AppointmentAction()], [1, 2], [true, false]
    )
    # `current`'s own round is done, so it is now settled; `other`, never the
    # current case of a round, is still pending.
    @test EpiBranch.is_settled(state, current)
    @test !EpiBranch.is_settled(state, other)
end

struct _UndeclaredAction <: AbstractIntervention end
EpiBranch.intervention_actions(::_UndeclaredAction, state, candidates) = nothing

@testset "Continuous-time candidates scale with the ring or group, not the population" begin
    # A settled case's own ring or group is tiny; most of the population is
    # neither traced by it nor in its group, and must never be materialised
    # as a candidate just because it is still pending.
    n = 500
    members = collect(1:n)
    processed = falses(n)
    pos = Dict(id => k for (k, id) in enumerate(members))
    is_pending = id -> EpiBranch._pending(id, members, processed, pos)
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    append!(state.individuals, [Individual(id = i) for i in 1:n])
    current = state.individuals[1]

    rv = RingVaccination(efficacy = 0.8)
    contacts = (id, st) -> id == 1 ? (2, 3) : ()
    ring_candidates = EpiBranch.action_candidates(
        rv, state, current, nothing, is_pending, members, contacts
    )
    @test length(ring_candidates) == 3
    @test Set(ind.id for ind in ring_candidates) == Set([1, 2, 3])

    # A wrapper such as `CapacityConstrained` forwards to the intervention it
    # wraps, instead of falling back to the default.
    wrapped_candidates = EpiBranch.action_candidates(
        CapacityConstrained(rv; budget_per_period = Inf),
        state, current, nothing, is_pending, members, contacts
    )
    @test Set(ind.id for ind in wrapped_candidates) == Set(ind.id for ind in ring_candidates)

    gv = GroupVaccination(efficacy = 0.8)
    for i in 1:3
        state.individuals[i].state[:group] = :A
    end
    for i in 4:n
        state.individuals[i].state[:group] = i
    end
    group_candidates = EpiBranch.action_candidates(
        gv, state, current, nothing, is_pending, members, contacts
    )
    @test length(group_candidates) == 3
    @test Set(ind.id for ind in group_candidates) == Set([1, 2, 3])

    # Any other intervention proposing actions is offered, by default, the
    # case and the contacts it names, the ring's own set.
    undeclared_candidates = EpiBranch.action_candidates(
        _UndeclaredAction(), state, current, nothing, is_pending, members, contacts
    )
    @test Set(ind.id for ind in undeclared_candidates) == Set([1, 2, 3])
end

@testset "A custom intervention can narrow its own action candidates" begin
    n = 500
    members = collect(1:n)
    processed = falses(n)
    pos = Dict(id => k for (k, id) in enumerate(members))
    is_pending = id -> EpiBranch._pending(id, members, processed, pos)
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
    )
    append!(state.individuals, [Individual(id = i) for i in 1:n])

    # `current` here is not the target: a custom method answers however it
    # likes, not just the default's "current plus its contacts".
    current = state.individuals[2]
    candidates = EpiBranch.action_candidates(
        SoleTargetAction(1), state, current, nothing, is_pending, members, nothing
    )
    @test length(candidates) == 1
    @test only(candidates).id == 1
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
            Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf),
            ContactTracing(OnIsolation(), 1.0, Dirac(0.0), Quarantine(duration = Inf)),
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

@testset "MassVaccination coverage declines without redrawing" begin
    mv = MassVaccination(efficacy = 0.9, eligibility_time = 0.0, coverage = 0.5)
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(1.0), Exponential(5.0)),
        EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(3)
    )
    candidates = [Individual(id = i, state = Dict{Symbol, Any}()) for i in 1:200]
    append!(state.individuals, candidates)

    EpiBranch.apply_actions!(mv, state, candidates)
    first_round = Set(ind.id for ind in candidates if is_vaccinated(ind))
    # Half of 200 candidates, up to the sampling noise of a fair coin.
    @test 70 < length(first_round) < 130

    # Asking again does not redraw a coin for anyone who already declined.
    for _ in 1:10
        EpiBranch.apply_actions!(mv, state, candidates)
    end
    @test Set(ind.id for ind in candidates if is_vaccinated(ind)) == first_round

    # Full coverage keeps the previous, deterministic behaviour.
    full = MassVaccination(efficacy = 0.9, eligibility_time = 0.0)
    full_candidates = [Individual(id = i, state = Dict{Symbol, Any}()) for i in 1:20]
    append!(state.individuals, full_candidates)
    EpiBranch.apply_actions!(full, state, full_candidates)
    @test all(is_vaccinated, full_candidates)

    # No coverage vaccinates nobody.
    none = MassVaccination(efficacy = 0.9, eligibility_time = 0.0, coverage = 0.0)
    none_candidates = [Individual(id = i, state = Dict{Symbol, Any}()) for i in 1:20]
    append!(state.individuals, none_candidates)
    EpiBranch.apply_actions!(none, state, none_candidates)
    @test !any(is_vaccinated, none_candidates)
end

@testset "An admitted action runs at its own time and may return further actions" begin
    # One offspring, five days after its parent, so the index case's own
    # action, dated day 10, falls after the contact has settled and moved
    # `max_infection_time` on from the day the action was admitted (day 0).
    state = simulate(
        ModelSpec(BranchingProcess(Dirac(1), Dirac(5.0)); interventions = [DeferredCheckIn()]);
        n_initial = 1, max_generations = 1, rng = StableRNG(1)
    )
    index = state.individuals[1]
    @test index.infection_time == 0.0
    @test index.state[:seen_max_infection_time] == 5.0
    @test index.state[:followed_up]
end
