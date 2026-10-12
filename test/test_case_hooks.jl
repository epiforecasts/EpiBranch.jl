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
end
