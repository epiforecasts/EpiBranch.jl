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
        # episode; a stale carry-over from the previous episode would
        # double it up.
        @test all(length(e.secondary_case_ids) == 1 for e in one.episodes)
        @test all(length(e.secondary_case_ids) == 1 for e in two.episodes)
    end
end
