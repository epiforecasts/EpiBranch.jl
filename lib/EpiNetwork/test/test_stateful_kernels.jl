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

@testset "A redraw adds no contact outside the kernel's support" begin
    # Host 3 settles at exactly 0.8 while host 2's early atom for host 4 is
    # stored at 0.7 + 0.1, one float below it; the redraw must not move that
    # past contact onto 0.8.
    function callback(c, a, b)
        c.infector == 1 && return c.susceptible == 2 ? Dirac(0.7) : Dirac(0.8)
        return DiscreteNonParametric([0.1, 0.3], [0.5, 0.5])
    end
    model = ModelSpec(
        NetworkProcess([[2, 3], [1, 4], [1], [2]], StatefulKernel(tick_state, callback));
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [TickEveryCase()])
    times = [simulate(model; initial_cases = [1], rng = StableRNG(seed)).individuals[4].infection_time
             for seed in 1:500]
    @test all(t -> t == 0.7 + 0.1 || t == 0.7 + 0.3, times)
    @test 0.4 < count(==(0.7 + 0.1), times) / 500 < 0.6
end
