@testset "Chosen household initial cases" begin
    process = HouseholdProcess([2, 3], Exponential(1.0))
    model = ModelSpec(process;
        progression = [Transition(:recovered;
            from = :infection, delay = 0.0, terminal = true)])
    state = simulate(model; initial_cases = [2, 5], rng = StableRNG(42))
    @test [i.id for i in state.individuals if is_infected(i)] == [2, 5]
    @test all(i -> i.infection_time == 0, filter(is_infected, state.individuals))
    @test simulate(model; initial_cases = Int[], rng = StableRNG(42)).cumulative_cases == 0
    @test_throws ArgumentError simulate(model; initial_cases = [6])
    external = HouseholdProcess([2, 3], Exponential(1.0); external_hazard = 0.1, obs_end = 5.0)
    @test_throws ArgumentError simulate(external; initial_cases = [2])
end
