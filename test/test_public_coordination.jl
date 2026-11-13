using Test
using EpiBranch
using Distributions
using StableRNGs
using Dates

# The channels through which built-in interventions coordinate are public, so
# an intervention written outside the package can use them.

# A tracer of another shape than `ContactTracing`: it marks every contact of a
# case as followed up, through an action it hands to `admit_trace_actions!`.
struct ExternalTracer <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::ExternalTracer) = :_external_followed_up
EpiBranch.capacity_time_key(::ExternalTracer) = :_external_followed_up_time

function EpiBranch.trace_contacts!(::ExternalTracer, state, infector, contacts)
    actions = EpiBranch.InterventionAction[]
    for contact in contacts
        push!(
            actions, EpiBranch.InterventionAction(
                contact, 1.0, (person, at, st) -> begin
                    person.state[:_external_followed_up] = true
                    person.state[:_external_followed_up_time] = at
                end
            )
        )
    end
    EpiBranch.admit_trace_actions!(ExternalTracer(), state, actions)
    return nothing
end

# A tracer that notes whether it was called for a ring member.
struct RingAwareTracer <: EpiBranch.AbstractIntervention end
function EpiBranch.trace_contacts!(::RingAwareTracer, state, infector, contacts)
    push!(get!(Vector{Tuple{Int, Bool}}, state.scratch, :ring_calls), (infector.id, EpiBranch.is_ring_hop(infector)))
    return nothing
end

@testset "admit_trace_actions! lets a wrapper ration an external tracer" begin
    function followed_up(wrap)
        state = simulate(BranchingProcess(Poisson(0.0)); n_initial = 4, rng = StableRNG(1))
        people = state.individuals
        EpiBranch.trace_contacts!(wrap(ExternalTracer()), state, people[1], people[2:4])
        return count(p -> get(p.state, :_external_followed_up, false), people)
    end
    @test followed_up(identity) == 3
    @test followed_up(iv -> CapacityConstrained(iv; budget_per_period = 1.0)) == 1
    @test followed_up(iv -> Scheduled(CapacityConstrained(iv; budget_per_period = 2.0); start_time = 0.0)) == 2
    @test followed_up(iv -> HeldCapacity(iv; capacity = 1.0)) == 1
end

@testset "flag_for_isolation! hands a traced contact to Isolation" begin
    state = simulate(BranchingProcess(Poisson(0.0)); n_initial = 1, rng = StableRNG(1))
    ind = Individual(id = 1, infection_time = 0.0)
    @test EpiBranch.traced_isolation_time(ind) == Inf
    EpiBranch.flag_for_isolation!(ind, 4.0)
    EpiBranch.flag_for_isolation!(ind, 6.0)
    @test EpiBranch.traced_isolation_time(ind) == 4.0
    EpiBranch.flag_for_isolation!(ind, 3.0)
    @test EpiBranch.traced_isolation_time(ind) == 3.0

    # Held back to the onset when the flag precedes it, and not applied at all
    # to a contact with no onset.
    iso = Isolation(onset_to_isolation_delay = Dirac(100.0), duration = Inf)
    ind.state[:onset_time] = 5.0
    EpiBranch.resolve_individual!(iso, ind, state)
    @test isolation_time(ind) == 5.0
    noonset = Individual(id = 2, infection_time = 0.0)
    noonset.state[:onset_time] = NaN
    EpiBranch.flag_for_isolation!(noonset, 3.0)
    EpiBranch.resolve_individual!(iso, noonset, state)
    @test isolation_time(noonset) == Inf
end

@testset "FlagOnly records its trace through flag_for_isolation!" begin
    contact = Individual(id = 1, infection_time = 1.0)
    EpiBranch.apply_trace!(FlagOnly(), contact, nothing, 2.0, StableRNG(1))
    @test EpiBranch.traced_isolation_time(contact) == 2.0
end

@testset "is_ring_hop is true for a case the ring walk passes through" begin
    @test !EpiBranch.is_ring_hop(Individual(id = 1))
    clinical = clinical_presentation(incubation_period = Dirac(1.0))
    ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly(); depth = 3)
    model = ModelSpec(
        BranchingProcess(Dirac(2), Dirac(2.0));
        interventions = [ct, RingAwareTracer()], attributes = clinical
    )
    state = simulate(model; max_generations = 3, rng = StableRNG(1))
    calls = get(state.scratch, :ring_calls, Tuple{Int, Bool}[])
    @test !isempty(calls)
    @test any(last, calls)
    @test !all(last, calls)
    @test all(!EpiBranch.is_ring_hop, state.individuals)
end

@testset "reported_value sets what the line list shows" begin
    EpiBranch.reported_value(::Val{:visits}, ind) = length(get(ind.state, :visits, ()))
    state = simulate(BranchingProcess(Poisson(0.0)); n_initial = 2, rng = StableRNG(1))
    state.individuals[1].state[:visits] = [1.0, 2.0, 3.0]
    state.individuals[2].state[:visits] = Float64[]
    df = linelist(state)
    @test df.visits == [3, 0]
    @test EpiBranch.reported_value(Val(:no_such_key), state.individuals[1]) |> ismissing
end
