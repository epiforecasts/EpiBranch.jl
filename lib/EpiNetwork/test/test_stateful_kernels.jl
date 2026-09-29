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

struct HalfBlockOneToFour <: AbstractIntervention end
function EpiBranch.competing_risk(::HalfBlockOneToFour, parent, contact, state)
    Risk(block_probability = (parent.id, contact.id) == (1, 4) ? 0.5 : 0.0)
end

@testset "A zero-length contact leaves resolved atoms resolved" begin
    # At t = 1 the race blocks or settles host 4, settles host 5, and then host
    # 5's zero-length contact sends it back to host 2. Host 4's contact at t = 1
    # has been resolved by then and must not be offered again.
    kernel = StatefulKernel(tick_state, (c, a, b) -> c.infector == 5 ? Dirac(0.0) :
                                                     Dirac(1.0))
    model = ModelSpec(NetworkProcess([[4, 5], [5], Int[], [1], [1, 2]], kernel);
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [HalfBlockOneToFour(), TickEveryCase()])
    n = 2000
    at_one = count(1:n) do seed
        state = simulate(model; initial_cases = [1], rng = StableRNG(seed))
        state.individuals[4].infection_time == 1.0
    end
    @test isapprox(at_one / n, 0.5; atol = 0.04)
end
