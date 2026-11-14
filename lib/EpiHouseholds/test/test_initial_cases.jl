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

@testset "Chosen initial cases are infected whatever the risks" begin
    prog = [Transition(:recovered; delay = 5.0, terminal = true)]
    vaccine = MassVaccination(eligibility_time = -30.0, efficacy = 1.0)
    process = HouseholdProcess(
        fill(3, 10), Exponential(2.0); external_hazard = 0.001, obs_end = 50.0
    )
    state = simulate(
        ModelSpec(process; progression = prog, interventions = [vaccine]);
        initial_cases = [1, 4], rng = StableRNG(1)
    )
    for id in (1, 4)
        ind = state.individuals[id]
        @test is_infected(ind)
        @test ind.infection_time == 0
        @test ind.state[:seeded]
        @test !haskey(ind.state, :infection_route)
    end
end

@testset "Chosen initial cases are conditioned on under an external hazard" begin
    prog = [Transition(:recovered; from = :infection, delay = Gamma(3.0, 2.0), terminal = true)]
    spec(α) = ModelSpec(
        HouseholdProcess(fill(4, 1500), Exponential(4.0); external_hazard = α, obs_end = 30.0);
        progression = prog
    )
    seeded = [4(h - 1) + 1 for h in 1:500]
    state = simulate(spec(0.005); rng = StableRNG(301), initial_cases = seeded)
    data = household_infections(state, spec(0.005))
    @test findall(data.seeded) == seeded
    grid = 0.003:0.0005:0.009
    αhat = grid[argmax([loglikelihood(data, spec(a)) for a in grid])]
    @test 0.004 <= αhat <= 0.006
end

@testset "Chosen initial cases do not change what an explained index case means" begin
    prog = [Transition(:recovered; from = :infection, delay = Gamma(3.0, 2.0), terminal = true)]
    spec = ModelSpec(
        HouseholdProcess(fill(4, 50), Exponential(4.0); external_hazard = 0.005, obs_end = 30.0);
        progression = prog
    )
    state = simulate(spec; rng = StableRNG(5), initial_cases = [1, 5])
    data = household_infections(state, spec)
    @test findall(data.seeded) == [1, 5]
    # Marked as index cases too, so a layout without the hazard conditions on them.
    @test all(data.is_index[[1, 5]])
    unseeded = HouseholdInfections(
        data.household_of, data.infection_time, data.infectious_time, data.removal_time,
        data.is_index; obs_end = data.obs_end
    )
    @test !any(unseeded.seeded)
    @test loglikelihood(data, spec) > loglikelihood(unseeded, spec)
end
