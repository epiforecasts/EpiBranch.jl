include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "stateful_kernels.jl"))
test_stateful_simulation(
    (k; kwargs...) -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k; kwargs...),
    network_infections)

@testset "Simultaneous contacts survive refresh at an inexact opening" begin
    # Host 2 opens at 0.7, so its two contacts are stored at 0.7 + 0.1, which
    # rounds below 0.8; settling one moves the other's record.
    kernel = StatefulKernel(tick_state,
        (c, a, b) -> c.infector == 1 ? Dirac(0.7) : Dirac(0.1))
    model = ModelSpec(NetworkProcess([[2], [1, 3, 4], [2], [2]], kernel);
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [TickEveryCase()])
    state = simulate(model; initial_cases = [1], rng = StableRNG(1))
    @test all(is_infected, state.individuals)
    @test state.individuals[3].infection_time == state.individuals[4].infection_time ==
          0.7 + 0.1
end
