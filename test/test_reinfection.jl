using Random

@testset "Reinfection after waning" begin
    @testset "A second episode no longer overwrites the first in silence" begin
        ind = Individual(; id = 1, infection_time = 5.0)
        ind.state[:recovered] = true
        ind.state[:outcome_time] = 12.0

        close_episode!(ind, InfectionEpisode(ind))
        ind.infection_time = 40.0

        @test ind.infection_time == 40.0
        @test length(ind.episodes) == 1
        @test ind.episodes[1].infection_time == 5.0
        @test ind.episodes[1].state[:outcome_time] == 12.0
    end

    @testset "susceptible_again_time defaults to Inf" begin
        ind = Individual(; id = 1)
        @test susceptible_again_time(ind) == Inf
        ind.state[:susceptible_again_time] = 10.0
        @test susceptible_again_time(ind) == 10.0
    end

    # Two nodes that keep exposing each other every generation. Transmission
    # always succeeds (no susceptibility/infectiousness block), so a
    # recovered node's only barrier to a fresh exposure is `HostImmunity`:
    # blocked while `susceptible_again_time` is still ahead of the exposure,
    # open once it is behind it. `delay = 0.0` on `:susceptible_again` makes
    # a node eligible again the moment it recovers, hence every ping-pong
    # exposure after the first one is a reinfection.
    struct PingPongModel <: EpiBranch.TransmissionModel end
    EpiBranch.population_size(::PingPongModel) = EpiBranch.NoPopulation()
    function EpiBranch.initialise_state(
            m::PingPongModel, sim_opts::EpiBranch.SimOpts,
            interventions, transitions, attributes, rng::AbstractRNG
        )
        state = EpiBranch.new_state(m, transitions, attributes, rng)
        EpiBranch.add_individuals!(state, 2, interventions)
        EpiBranch.seed!(state, 1:1, interventions, transitions)
        return state
    end
    function EpiBranch.collect_exposures(m::PingPongModel, state::EpiBranch.SimulationState)
        return EpiBranch.gather_by_target(m, state)
    end
    function EpiBranch.contacts_of(::PingPongModel, parent, state::EpiBranch.SimulationState)
        other = state.individuals[parent.id == 1 ? 2 : 1]
        return [(other, parent.infection_time + 1.0)]
    end

    # A custom reinfection step written exactly as the `FollowupVisit` idiom
    # in the docs: one `transition_time` call, gated on `probability`, that
    # only sometimes sets its target field. `_allows_reinfection` must read
    # this as capable regardless of how a given probe draw would have landed.
    struct _ProbabilisticReinfection <: EpiBranch.AbstractClinicalTransition
        probability::Float64
    end
    function EpiBranch.resolve_individual!(t::_ProbabilisticReinfection, ind, state)
        time = EpiBranch.transition_time(
            state.rng, ind, ind.infection_time, 0.0; probability = t.probability
        )
        time === nothing || (ind.state[:susceptible_again_time] = time)
        return nothing
    end

    @testset "A low-probability custom reinfection step is still detected" begin
        for seed in 1:20
            transitions = EpiBranch.AbstractClinicalTransition[_ProbabilisticReinfection(0.05)]
            state = EpiBranch.new_state(
                PingPongModel(), transitions, NoAttributes(), MersenneTwister(seed)
            )
            @test EpiBranch._allows_reinfection(state)
        end
    end

    @testset "A custom step that can never reinfect is not detected" begin
        transitions = EpiBranch.AbstractClinicalTransition[_ProbabilisticReinfection(0.0)]
        state = EpiBranch.new_state(
            PingPongModel(), transitions, NoAttributes(), MersenneTwister(1)
        )
        @test !EpiBranch._allows_reinfection(state)
    end

    @testset "A model that keeps offering a recovered host gets fresh episodes" begin
        progression = [
            Transition(:recovered, from = :infection, delay = 1.0),
            Transition(:susceptible_again, from = :recovered, delay = 0.0),
        ]
        spec = ModelSpec(PingPongModel(); progression)
        state = simulate(spec; n_initial = 1, stopping_rules = [MaxGenerations(4)])
        one, two = state.individuals

        @test length(one.episodes) >= 1
        @test length(two.episodes) >= 1
        # The archived episode is the first one (seeded at time 0 for node
        # 1, infected at time 1 for node 2); the live fields have moved on to
        # a later one.
        @test one.episodes[1].infection_time == 0.0
        @test one.infection_time > one.episodes[1].infection_time
        @test two.episodes[1].infection_time == 1.0
        @test two.infection_time > two.episodes[1].infection_time
        # The closing episode's own outcome survives rather than being
        # clobbered by the next one's.
        @test one.episodes[1].state[:recovered_time] == 1.0
        @test one.state[:recovered_time] > 1.0
        # Each ping-pong exchange produces exactly one secondary case per
        # episode; a leftover count from the previous episode would show
        # up as two.
        @test all(length(e.secondary_case_ids) == 1 for e in one.episodes)
        @test all(length(e.secondary_case_ids) == 1 for e in two.episodes)
    end

    # Counts, on each individual, how many times `initialise_individual!` has
    # run: once when they are created, and again every time a reinfection
    # settles a new episode. A host whose count stalls at 1 past its first
    # episode would be reusing an earlier episode's intervention state (such
    # as `Isolation`'s `:test_positive`) for every later one.
    struct _CountInitialisations <: EpiBranch.AbstractIntervention end
    function EpiBranch.initialise_individual!(::_CountInitialisations, ind, state)
        ind.state[:_init_count] = get(ind.state, :_init_count, 0) + 1
        return nothing
    end

    @testset "A reinfection re-initialises intervention state for the new episode" begin
        progression = [
            Transition(:recovered, from = :infection, delay = 1.0),
            Transition(:susceptible_again, from = :recovered, delay = 0.0),
        ]
        spec = ModelSpec(
            PingPongModel(); progression,
            interventions = [_CountInitialisations()]
        )
        state = simulate(spec; n_initial = 1, stopping_rules = [MaxGenerations(4)])
        one, two = state.individuals

        @test length(one.episodes) >= 1
        @test one.state[:_init_count] == length(one.episodes) + 1
        @test two.state[:_init_count] == length(two.episodes) + 1
    end

    @testset "ReportedCases counts each episode's report once" begin
        # Every infection is reported half a day in, and each ping-pong
        # exchange reinfects a host, so the count has to find the reports
        # that reinfections archived.
        progression = [
            Reporting(delay = 0.5, from = :infection),
            Transition(:recovered, from = :infection, delay = 1.0),
            Transition(:susceptible_again, from = :recovered, delay = 0.0),
        ]
        # The schedule only has the gate asked during the run; its isolation
        # comes too late to stop the exchanges.
        iso = Isolation(onset_to_isolation_delay = 100.0, duration = Inf)
        spec = ModelSpec(
            PingPongModel(); progression,
            attributes = clinical_presentation(incubation_period = Dirac(0.2)),
            interventions = [Scheduled(iso; start_after = ReportedCases(3))]
        )
        state = simulate(spec; n_initial = 1, stopping_rules = [MaxGenerations(6)])
        @test any(ind -> !isempty(ind.episodes), state.individuals)

        due(st, clock) = get(st, :reported, false) && st[:reporting_time] <= clock
        function recount(clock)
            return sum(state.individuals) do ind
                count(e -> due(e.state, clock), ind.episodes) + due(ind.state, clock)
            end
        end
        for clock in 0.0:0.25:8.0
            state.max_infection_time = clock
            @test EpiBranch._reported_cases_due(state) == recount(clock)
        end
    end
end
