# Ending an infection before onset through the engine's abort API, from an
# intervention the package does not ship.

# Ends every infection it sees `delay` days after it starts, unless symptoms
# would come first: the shape of a post-exposure antiviral.
struct EndAfter <: AbstractIntervention
    delay::Float64
end
function _end_after!(e::EndAfter, ind)
    incubation = get(ind.state, :incubation_period, NaN)
    ends = ind.infection_time + e.delay
    isnan(incubation) || ends < ind.infection_time + incubation || return nothing
    return EpiBranch.abort_infection!(ind, ends)
end
function EpiBranch.apply_post_transmission!(e::EndAfter, state, contacts)
    foreach(c -> _end_after!(e, c), contacts)
    return nothing
end
function EpiBranch.on_infection_settled!(e::EndAfter, ind, state, rng)
    return _end_after!(e, ind)
end

# One transition before the abort, one after it.
const ABORT_PROGRESSION = [
    Transition(:early; from = :infection, delay = 1.0),
    Transition(:late; from = :infection, delay = 10.0),
]

@testset "abort_infection!" begin
    function exposed(; infection_time = 1.0, incubation = 6.0)
        ind = Individual(id = 1, infection_time = infection_time)
        ind.state[:incubation_period] = incubation
        EpiBranch._set_onset_from_incubation!(ind)
        return ind
    end

    @testset "records the earliest abort and suppresses onset" begin
        ind = exposed()
        @test EpiBranch.infection_aborted_time(ind) == Inf
        @test onset_time(ind) == 7.0
        EpiBranch.abort_infection!(ind, 4.0)
        @test EpiBranch.infection_aborted_time(ind) == 4.0
        @test isnan(onset_time(ind))
        EpiBranch.abort_infection!(ind, 5.0)
        @test EpiBranch.infection_aborted_time(ind) == 4.0
        EpiBranch.abort_infection!(ind, 3.0)
        @test EpiBranch.infection_aborted_time(ind) == 3.0
        # An asymptomatic infection has no onset to come first.
        asymptomatic = exposed(incubation = NaN)
        EpiBranch.abort_infection!(asymptomatic, 50.0)
        @test EpiBranch.infection_aborted_time(asymptomatic) == 50.0
    end

    @testset "only ends an infection between its start and onset" begin
        @test_throws ArgumentError EpiBranch.abort_infection!(exposed(), 1.0)
        @test_throws ArgumentError EpiBranch.abort_infection!(exposed(), 7.0)
        @test_throws ArgumentError EpiBranch.abort_infection!(
            exposed(infection_time = NaN), 4.0
        )
    end

    @testset "an abort is dropped when resolution does not bear it out" begin
        function aborted(infected, infection_time)
            ind = exposed(infection_time = 0.0)
            EpiBranch.abort_infection!(ind, 4.0)
            ind.state[:infected] = infected
            ind.infection_time = infection_time
            return ind
        end
        # Infected before the abort: the abort stands.
        ind = aborted(true, 1.0)
        EpiBranch._drop_stale_abort!(ind)
        @test EpiBranch.infection_aborted_time(ind) == 4.0
        @test isnan(onset_time(ind))
        # Infected through a later exposure at or after the abort time.
        ind = aborted(true, 4.0)
        EpiBranch._drop_stale_abort!(ind)
        @test EpiBranch.infection_aborted_time(ind) == Inf
        @test onset_time(ind) == 10.0
        # Not infected at all.
        ind = aborted(false, 0.0)
        EpiBranch._drop_stale_abort!(ind)
        @test EpiBranch.infection_aborted_time(ind) == Inf
        @test onset_time(ind) == 6.0
        # Nothing to drop.
        ind = exposed()
        EpiBranch._drop_stale_abort!(ind)
        @test onset_time(ind) == 7.0
    end
end

@testset "An outside intervention ends infections on the generation engine" begin
    spec = ModelSpec(
        BranchingProcess(Poisson(3.0), Exponential(2.0));
        interventions = [EndAfter(2.0)],
        attributes = clinical_presentation(
            incubation_period = Dirac(5.0), prob_asymptomatic = 0.0
        ),
        progression = ABORT_PROGRESSION
    )
    aborted = 0
    misplaced = 0
    after_abort = 0
    before_abort = 0
    for seed in 1:20
        state = simulate(spec; rng = StableRNG(seed), max_cases = 200)
        for ind in filter(is_infected, state.individuals)
            ind.generation == 0 && continue
            aborted += 1
            t = EpiBranch.infection_aborted_time(ind)
            t == ind.infection_time + 2.0 &&
                isnan(onset_time(ind)) &&
                get(ind.state, :early_time, Inf) == ind.infection_time + 1.0 &&
                get(ind.state, :late_time, Inf) == Inf ||
                (misplaced += 1)
            for id in ind.secondary_case_ids
                child = state.individuals[id]
                is_infected(child) || continue
                child.infection_time >= t ? (after_abort += 1) : (before_abort += 1)
            end
        end
    end
    @test aborted > 0
    @test misplaced == 0
    @test before_abort > 0
    @test after_abort == 0
end

@testset "An outside intervention ends infections on a continuous-time race" begin
    # Node 1 is seeded at 0 and reaches node 2 at 1; node 2 reaches node 3 three
    # days after its own infection. Ending each infection two days in stops node
    # 2 at 3, before the contact at 4.
    function race(interventions)
        rng = StableRNG(1)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(1.0)),
            ABORT_PROGRESSION,
            clinical_presentation(incubation_period = Dirac(5.0), prob_asymptomatic = 0.0),
            rng
        )
        EpiBranch.add_individuals!(state, 3, interventions)
        function edge(from, to, t)
            return (inf, st) -> inf == from &&
                !get(st.individuals[to].state, :infected, false) ?
                ((to, Dirac(t)),) : ()
        end
        routes = (
            (RouteWindow(:one; until = (:late,), kernel = Dirac(1.0)), edge(1, 2, 1.0)),
            (RouteWindow(:two; until = (:late,), kernel = Dirac(3.0)), edge(2, 3, 3.0)),
        )
        EpiBranch._sellke_race!(
            state, [1, 2, 3], rng; routes, interventions,
            seed! = (best, members, r) -> (best[1] = 0.0)
        )
        return state
    end
    infected(state) = [get(ind.state, :infected, false) for ind in state.individuals]

    untreated = race(AbstractIntervention[])
    @test infected(untreated) == [true, true, true]
    @test onset_time(untreated.individuals[2]) == 6.0
    @test untreated.individuals[2].state[:late_time] == 11.0

    treated = race(AbstractIntervention[EndAfter(2.0)])
    @test infected(treated) == [true, true, false]
    node2 = treated.individuals[2]
    @test EpiBranch.infection_aborted_time(node2) == 3.0
    @test isnan(onset_time(node2))
    @test node2.state[:early_time] == 2.0
    @test get(node2.state, :late_time, Inf) == Inf
end
