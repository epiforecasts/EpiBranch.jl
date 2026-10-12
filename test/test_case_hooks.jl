# Both engines call one set of per-case intervention hooks: `trace_contacts!`
# with an infector and the contacts it reached, `intervention_actions` for
# actions with a delivery time, and `keep_active` for who goes on generating
# contacts. The generation engine calls them for each case of the previous
# generation rather than handing an intervention the whole batch.

# Records, on each contact, the infector it was offered with and the state the
# hook saw, and on each `keep_active` call whether every target came from one
# infector.
struct _RecordCaseHooks <: AbstractIntervention end
function EpiBranch.trace_contacts!(::_RecordCaseHooks, state, infector, contacts)
    for c in contacts
        c.state[:_offered_by] = infector.id
        c.state[:_offered_clock] = state.max_infection_time
        c.state[:_offered_count] = state.cumulative_cases
    end
    return nothing
end
function EpiBranch.keep_active(::_RecordCaseHooks, state, targets, is_new)
    push!(
        get!(Vector{Bool}, state.scratch, :_one_infector_per_call),
        length(unique(t.parent_id for t in targets)) == 1
    )
    return ()
end

# Records on each contact the infector that traced it, and nothing else.
struct _TraceOnly <: AbstractIntervention end
function EpiBranch.trace_contacts!(::_TraceOnly, state, infector, contacts)
    for c in contacts
        c.state[:_traced_from] = infector.id
    end
    return nothing
end

# Doses every contact it is offered at a fixed time, through actions alone.
struct _ActionsOnly <: AbstractIntervention end
function EpiBranch.intervention_actions(::_ActionsOnly, state, candidates)
    return [
        EpiBranch.InterventionAction(c, 3.0, (ind, at, st) -> ind.state[:_served_at] = at)
            for c in candidates
    ]
end

# Written against the deprecated batch hook only.
struct _BatchOnly <: AbstractIntervention end
function EpiBranch.apply_post_transmission!(::_BatchOnly, state, new_contacts)
    for c in new_contacts
        c.state[:_batch_seen] = true
    end
    return nothing
end

# A ring written against the batch hook: it keeps active the contacts its own
# batch hook marked.
struct _BatchRing <: AbstractIntervention end
function EpiBranch.apply_post_transmission!(::_BatchRing, state, new_contacts)
    for c in new_contacts
        c.state[:_batch_marked] = true
    end
    return nothing
end
function EpiBranch.keep_active(::_BatchRing, state, targets, is_new)
    return [t.id for t in targets if get(t.state, :_batch_marked, false)]
end
# Leaves half the contacts uninfected, so there is someone to keep active.
EpiBranch.competing_risk(::_BatchRing, parent, contact, state) =
    Risk(block_probability = 0.5)

# A tracer written against the batch hook, marking contacts as traced.
struct _BatchTracer <: AbstractIntervention end
EpiBranch.initialise_individual!(::_BatchTracer, ind, state) = (ind.state[:traced] = false)
function EpiBranch.apply_post_transmission!(::_BatchTracer, state, new_contacts)
    for c in new_contacts
        c.state[:traced] = true
        c.state[:trace_time] = c.infection_time
    end
    return nothing
end

# Traces and proposes actions, without declaring `continuous_actions`.
struct _TraceAndAct <: AbstractIntervention end
EpiBranch.trace_contacts!(::_TraceAndAct, state, infector, contacts) = nothing
EpiBranch.intervention_actions(::_TraceAndAct, state, candidates) = nothing

# A model that can name each case's contacts.
struct _NamesContacts <: EpiBranch.TransmissionModel end
EpiBranch.supplies_contacts(::_NamesContacts) = true

_case_hook_model(ivs) = ModelSpec(
    BranchingProcess(Poisson(2.0), Exponential(3.0)); interventions = ivs
)

@testset "Per-case intervention hooks on the generation engine" begin
    @testset "trace_contacts! is called once per case with its own contacts" begin
        state = simulate(
            _case_hook_model([_RecordCaseHooks()]);
            max_cases = 200, max_generations = 6, rng = StableRNG(11)
        )
        contacts = filter(ind -> ind.parent_id != 0, state.individuals)
        offered = filter(ind -> haskey(ind.state, :_offered_by), contacts)
        @test !isempty(contacts)
        @test length(offered) == length(contacts)
        @test all(ind -> ind.state[:_offered_by] == ind.parent_id, offered)
        # The hook sees the infector's own clock, as `resolve_individual!` does.
        @test all(
            ind -> ind.state[:_offered_clock] ==
                state.individuals[ind.parent_id].infection_time, offered
        )
        # The count is the infector's place among the cases settled up to it.
        cases = filter(is_infected, state.individuals)
        for ind in offered
            infector = state.individuals[ind.parent_id]
            @test ind.state[:_offered_count] == count(
                c -> c.generation < infector.generation ||
                    (
                    c.generation == infector.generation &&
                        c.infection_time <= infector.infection_time
                ), cases
            )
        end
        @test state.max_infection_time ==
            maximum(ind.infection_time for ind in cases)
        @test all(state.scratch[:_one_infector_per_call])
    end

    @testset "A schedule gates tracing at the infector's own clock" begin
        state = simulate(
            _case_hook_model([Scheduled(_RecordCaseHooks(); start_time = 5.0)]);
            max_cases = 200, max_generations = 6, rng = StableRNG(15)
        )
        contacts = filter(ind -> ind.parent_id != 0, state.individuals)
        infected_from(ind) = state.individuals[ind.parent_id].infection_time
        @test any(ind -> infected_from(ind) < 5.0, contacts)
        @test any(ind -> infected_from(ind) >= 5.0, contacts)
        @test all(
            ind -> haskey(ind.state, :_offered_by) == (infected_from(ind) >= 5.0),
            contacts
        )
    end

    @testset "Writing trace_contacts! is enough for the race to call it" begin
        @test EpiBranch.traces_contacts(_TraceOnly())
        @test !EpiBranch.traces_contacts(_ActionsOnly())
        # Node 1 is seeded at 0 and reaches node 2 at 1, which reaches node 3.
        interventions = AbstractIntervention[_TraceOnly()]
        rng = StableRNG(18)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(1.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), rng
        )
        EpiBranch.add_individuals!(state, 3, interventions)
        neighbours = Dict(1 => [2], 2 => [1, 3], 3 => [2])
        targets(inf, st) = (
            (to, Dirac(1.0)) for to in neighbours[inf]
                if !get(st.individuals[to].state, :infected, false)
        )
        EpiBranch._sellke_race!(
            state, [1, 2, 3], rng;
            routes = ((RouteWindow(:one; kernel = Dirac(1.0)), targets),),
            interventions, contacts = (inf, st) -> neighbours[inf],
            seed! = (best, members, r) -> (best[1] = 0.0)
        )
        @test state.individuals[2].state[:_traced_from] == 1
        @test state.individuals[3].state[:_traced_from] == 2
    end

    @testset "A tracer that also proposes actions is honoured where contacts are named" begin
        @test EpiBranch._sellke_honours(_NamesContacts(), _TraceAndAct())
        @test !EpiBranch._sellke_honours(_NamesContacts(), _ActionsOnly())
    end

    @testset "An intervention with only intervention_actions acts" begin
        state = simulate(
            _case_hook_model([_ActionsOnly()]);
            max_cases = 50, max_generations = 4, rng = StableRNG(12)
        )
        contacts = filter(ind -> ind.parent_id != 0, state.individuals)
        @test !isempty(contacts)
        @test all(ind -> get(ind.state, :_served_at, nothing) == 3.0, contacts)
    end

    @testset "An intervention with only the batch hook works, with a warning" begin
        model = _case_hook_model([_BatchOnly()])
        state = @test_logs (:warn, r"_BatchOnly.*apply_post_transmission!.*deprecated"s) match_mode = :any simulate(
            model; max_cases = 50, max_generations = 4, rng = StableRNG(13)
        )
        contacts = filter(ind -> ind.parent_id != 0, state.individuals)
        @test !isempty(contacts)
        @test all(ind -> get(ind.state, :_batch_seen, false), contacts)
    end

    @testset "Each batch-hook intervention is named in its own warning" begin
        @test_logs (:warn, r"^_BatchOnly") (:warn, r"^_BatchTracer") begin
            for iv in (_BatchOnly(), _BatchTracer()), seed in 1:2
                simulate(
                    _case_hook_model([iv]); max_cases = 10, max_generations = 2,
                    rng = StableRNG(seed)
                )
            end
        end
    end

    @testset "keep_active of a batch-hook intervention sees its batch writes" begin
        state = @test_logs (:warn,) match_mode = :any simulate(
            _case_hook_model([_BatchRing()]);
            max_cases = 10_000, max_generations = 3, rng = StableRNG(16)
        )
        # Every contact, infected or not, stays active and makes contacts.
        uninfected = filter(
            ind -> !is_infected(ind) && ind.generation < 3, state.individuals
        )
        @test !isempty(uninfected)
        @test any(ind -> !isempty(ind.secondary_case_ids), uninfected)
    end

    @testset "A batch-hook intervention feeds the interventions after it" begin
        rv = RingVaccination(efficacy = 0.9)
        state = @test_logs (:warn,) match_mode = :any simulate(
            _case_hook_model([_BatchTracer(), rv]);
            max_cases = 50, max_generations = 3, rng = StableRNG(19)
        )
        contacts = filter(ind -> ind.parent_id != 0, state.individuals)
        @test !isempty(contacts)
        @test all(is_vaccinated, contacts)
    end

    @testset "A node reached by a case and a ring member keeps the wider ring" begin
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(20)
        )
        case = Individual(id = 1, infection_time = 0.0)
        case.state[:onset_time] = 1.0
        case.state[:asymptomatic] = false
        set_isolated!(case, 2.0; release_time = Inf)
        member = Individual(id = 2, parent_id = 1, infection_time = 1.0)
        node = Individual(id = 3, parent_id = 1, infection_time = 3.0)
        append!(state.individuals, [case, member, node])
        ct = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.5),
            action = FlagOnly(), depth = 2
        )
        EpiBranch.initialise_individual!(ct, member, state)
        EpiBranch.initialise_individual!(ct, node, state)
        EpiBranch.trace_contacts!(ct, state, case, [member, node])
        member.state[:infected] = false
        @test node.state[:_ring_remaining] == 1
        EpiBranch.trace_contacts!(ct, state, member, [node])
        @test node.state[:_ring_remaining] == 1
    end

    @testset "Calling the batch hook on a built-in warns and still acts" begin
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(14)
        )
        infector = Individual(id = 1, infection_time = 0.0)
        infector.state[:onset_time] = 1.0
        infector.state[:asymptomatic] = false
        set_isolated!(infector, 2.0; release_time = Inf)
        contact = Individual(id = 2, parent_id = 1, infection_time = 3.0)
        append!(state.individuals, [infector, contact])
        ct = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.5),
            action = FlagOnly()
        )
        EpiBranch.initialise_individual!(ct, contact, state)
        @test_logs (:warn, r"apply_post_transmission!.*deprecated"s) EpiBranch.apply_post_transmission!(
            ct, state, [contact]
        )
        @test is_traced(contact)
        @test contact.state[:trace_time] == 2.5
    end
end
