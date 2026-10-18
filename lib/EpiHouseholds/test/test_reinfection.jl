# A household with a two-day infectious period and immunity that wanes three
# days after recovery, under a steady community hazard: over a long enough
# follow-up a susceptible household should see several episodes.
function _waning_progression(; delay = 3.0)
    return [
        Transition(:recovered; from = :infection, delay = 2.0),
        Transition(:susceptible_again; from = :recovered, delay = delay),
    ]
end

@testset "A household process reinfects a host once immunity has waned" begin
    model = ModelSpec(
        HouseholdProcess(
            fill(4, 150), Exponential(1.5);
            external_hazard = 0.05, obs_end = 200.0
        );
        progression = _waning_progression()
    )
    state = simulate(model; rng = StableRNG(1))

    infected = filter(is_infected, state.individuals)
    @test !isempty(infected)
    @test any(ind -> !isempty(ind.episodes), infected)
    @test any(ind -> length(ind.episodes) >= 2, infected)

    reinfected = first(filter(ind -> !isempty(ind.episodes), infected))
    first_episode = reinfected.episodes[1]
    # The archived episode is the earlier infection; the live fields moved on.
    @test reinfected.infection_time > first_episode.infection_time
    # Immunity held the second episode back until it waned.
    @test reinfected.infection_time >=
        first_episode.infection_time + 2.0 + 3.0
    # A fresh episode starts with no secondary cases of its own.
    @test length(reinfected.secondary_case_ids) <=
        count(ind -> ind.parent_id == reinfected.id, state.individuals)
end

@testset "Without a susceptible_again step nobody is reinfected" begin
    model = ModelSpec(
        HouseholdProcess(
            fill(4, 150), Exponential(1.5);
            external_hazard = 0.05, obs_end = 200.0
        );
        progression = [Transition(:recovered; from = :infection, delay = 2.0)]
    )
    state = simulate(model; rng = StableRNG(1))
    @test all(isempty(ind.episodes) for ind in state.individuals)
end

@testset "A probabilistic susceptible_again step leaves some hosts immune for good" begin
    # Only half of recoveries ever wane; the other half carry a permanent
    # (`susceptible_again_time == Inf`) block, which the race must stop
    # redrawing towards rather than looping forever.
    model = ModelSpec(
        HouseholdProcess(
            fill(4, 150), Exponential(1.5);
            external_hazard = 0.05, obs_end = 200.0
        );
        progression = [
            Transition(:recovered; from = :infection, delay = 2.0),
            Transition(:susceptible_again; from = :recovered, delay = 3.0, probability = 0.5),
        ]
    )
    state = simulate(model; rng = StableRNG(1))
    infected = filter(is_infected, state.individuals)
    @test any(ind -> isempty(ind.episodes) && isfinite(ind.infection_time), infected)
    @test any(ind -> !isempty(ind.episodes), infected)
end

@testset "household_infections warns when episodes are present" begin
    model = ModelSpec(
        HouseholdProcess(
            fill(4, 150), Exponential(1.5);
            external_hazard = 0.05, obs_end = 200.0
        );
        progression = _waning_progression()
    )
    state = simulate(model; rng = StableRNG(1))
    @test any(ind -> !isempty(ind.episodes), state.individuals)
    @test_logs (:warn, r"episode") match_mode = :any household_infections(state, model)

    once_only = ModelSpec(
        HouseholdProcess(fill(4, 50), Exponential(1.5));
        progression = [Transition(:recovered; from = :infection, delay = 2.0)]
    )
    @test_logs household_infections(simulate(once_only; rng = StableRNG(1)), once_only)
end
