using Distributions
using StableRNGs

@testset "A reinfection keeps a quarantine standing on the person" begin
    iso = Isolation(onset_to_isolation_delay = 0.5, duration = Inf)
    tracing = ContactTracing(
        TraceEveryone(), 1.0, Exponential(0.5), Quarantine(duration = 14.0)
    )
    interventions = (iso, tracing)
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)), Transition[], NoAttributes(), StableRNG(1)
    )
    EpiBranch.add_individuals!(state, 1, interventions)
    ind = state.individuals[1]
    ind.state[:test_positive] = false

    # Quarantined between episodes, then reinfected inside the quarantine.
    EpiBranch.close_episode!(ind, InfectionEpisode(ind))
    ind.state[:quarantined] = true
    EpiBranch.record_removal!(ind, 3.0, 17.0; key = EpiBranch.QUARANTINE_STRETCHES_KEY)
    ind.infection_time = 5.0
    EpiBranch._at_case!(state, ind, 1)

    EpiBranch._resolve_interventions!(state, ind, interventions)

    @test ind.state[:quarantined]
    @test EpiBranch.removal_stretches(ind, EpiBranch.QUARANTINE_STRETCHES_KEY) ==
        [(3.0, 17.0)]
    # The per-episode draw is still refreshed.
    @test ind.state[:test_positive] isa Bool
    ind.state[:test_positive] = :stale
    EpiBranch.initialise_episode!(iso, ind, state)
    @test ind.state[:test_positive] isa Bool
    # A wrapper forwards the hook to the intervention it wraps.
    ind.state[:test_positive] = :stale
    EpiBranch.initialise_episode!(Scheduled(iso; start_time = 0.0), ind, state)
    @test ind.state[:test_positive] isa Bool
    # The default does nothing.
    @test EpiBranch.initialise_episode!(tracing, ind, state) === nothing
end
