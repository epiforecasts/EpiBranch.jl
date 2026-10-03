@testset "Chosen household initial cases" begin
    process = HouseholdProcess([2, 3], Exponential(1.0))
    model = ModelSpec(
        process;
        progression = [
            Transition(
                :recovered;
                from = :infection, delay = 0.0, terminal = true
            ),
        ]
    )
    state = simulate(model; initial_cases = [2, 5], rng = StableRNG(42))
    @test [i.id for i in state.individuals if is_infected(i)] == [2, 5]
    @test all(i -> i.infection_time == 0, filter(is_infected, state.individuals))
    @test simulate(model; initial_cases = Int[], rng = StableRNG(42)).cumulative_cases == 0
    @test_throws ArgumentError simulate(model; initial_cases = [6])
end

@testset "Chosen initial cases with an external hazard" begin
    process = HouseholdProcess([2, 3], Exponential(1.0); external_hazard = 0.5, obs_end = 5.0)
    model = ModelSpec(
        process;
        progression = [
            Transition(
                :recovered;
                from = :infection, delay = 0.0, terminal = true
            ),
        ]
    )
    state = simulate(model; initial_cases = [2, 5], rng = StableRNG(42))
    chosen = filter(i -> i.id in (2, 5), state.individuals)
    @test all(i -> is_infected(i) && i.infection_time == 0, chosen)
    # With a household-wide external hazard active, some other member is
    # infected too, which the disallowed combination could never produce.
    @test any(i -> is_infected(i) && !(i.id in (2, 5)), state.individuals)
end
