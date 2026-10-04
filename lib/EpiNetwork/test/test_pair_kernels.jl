include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "pair_kernels.jl"))
test_contextual_simulation(k -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k), network_infections)
test_calendar_simulation(k -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k), network_infections)
test_stateful_simulation(
    (k; kwargs...) -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k; kwargs...),
    network_infections
)

@testset "Simultaneous contacts survive refresh at an inexact opening" begin
    # Host 2 opens at 0.7, so its two contacts are stored at 0.7 + 0.1, which
    # rounds below 0.8; settling one moves the other's record.
    kernel = PairKernel(
        (c, a, b) -> c.infector == 1 ? Dirac(0.7) : Dirac(0.1);
        state = tick_state, watches = (:tick,)
    )
    model = ModelSpec(
        NetworkProcess([[2], [1, 3, 4], [2], [2]], kernel);
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [TickEveryCase()]
    )
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
        NetworkProcess([[2, 3], [1, 4], [1], [2]], PairKernel(callback; state = tick_state, watches = (:tick,)));
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [TickEveryCase()]
    )
    times = [
        simulate(model; initial_cases = [1], rng = StableRNG(seed)).individuals[4].infection_time
            for seed in 1:500
    ]
    @test all(t -> t == 0.7 + 0.1 || t == 0.7 + 0.3, times)
    @test 0.4 < count(==(0.7 + 0.1), times) / 500 < 0.6
end

struct HalfBlockOneToFour <: AbstractIntervention end
function EpiBranch.competing_risk(::HalfBlockOneToFour, parent, contact, state)
    return Risk(block_probability = (parent.id, contact.id) == (1, 4) ? 0.5 : 0.0)
end

@testset "A zero-length contact leaves resolved atoms resolved" begin
    # At t = 1 the race blocks or settles host 4, settles host 5, and then host
    # 5's zero-length contact sends it back to host 2. Host 4's contact at t = 1
    # has been resolved by then and must not be offered again.
    kernel = PairKernel(
        (c, a, b) -> c.infector == 5 ? Dirac(0.0) : Dirac(1.0);
        state = tick_state, watches = (:tick,)
    )
    model = ModelSpec(
        NetworkProcess([[4, 5], [5], Int[], [1], [1, 2]], kernel);
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [HalfBlockOneToFour(), TickEveryCase()]
    )
    n = 2000
    at_one = count(1:n) do seed
        state = simulate(model; initial_cases = [1], rng = StableRNG(seed))
        state.individuals[4].infection_time == 1.0
    end
    @test isapprox(at_one / n, 0.5; atol = 0.04)
end

@testset "A zero-length contact made after the race passes a member stays due" begin
    # At t = 1 the race passes hosts 4 and 5; host 5 then makes zero-length
    # contacts to hosts 2 and 3. When host 2 settles and records move, the
    # contact to host 3 has not been resolved yet and must still happen.
    kernel = PairKernel(
        (c, a, b) -> c.infector == 5 ? Dirac(0.0) : Dirac(1.0);
        state = tick_state, watches = (:tick,)
    )
    model = ModelSpec(
        NetworkProcess([[4, 5], [5], [5], [1], [1, 2, 3]], kernel);
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [TickEveryCase()]
    )
    state = simulate(model; initial_cases = [1], rng = StableRNG(1))
    @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 1.0, 1.0, 1.0]
end

@testset "Simultaneous contacts go to the infector whose opening came first" begin
    # Hosts 2 and 3 both reach host 4 at t = 3; host 2 opened first, so it is
    # the infector however many unrelated leaves host 1 also reaches.
    function law(i, j)
        i == 1 || return i == 2 ? Dirac(2.0) : Dirac(1.0)
        return Dirac(j == 2 ? 1.0 : j == 3 ? 2.0 : j == 5 ? 2.5 : 0.5)
    end
    kernel = PairKernel((c, a, b) -> law(c.infector, c.susceptible); state = tick_state, watches = (:tick,))
    for leaves in 0:8
        adjacency = [Int[] for _ in 1:(5 + leaves)]
        for (x, y) in [(1, 2), (1, 3), (1, 5), (2, 4), (3, 4)]
            push!(adjacency[x], y)
            push!(adjacency[y], x)
        end
        for leaf in 6:(5 + leaves)
            push!(adjacency[1], leaf)
            push!(adjacency[leaf], 1)
        end
        model = ModelSpec(
            NetworkProcess(adjacency, kernel);
            progression = [Transition(:recovered; delay = 5.0, terminal = true)],
            interventions = [TickEveryCase()]
        )
        state = simulate(model; initial_cases = [1], rng = StableRNG(1))
        @test state.individuals[4].infection_time == 3.0
        @test state.individuals[4].parent_id == 2
    end
end

@testset "A routed live route redraws to the hazard the record leaves" begin
    # Ported from the closed #387. The exact tests elsewhere pin which pairs a
    # redraw touches; this one pins the arithmetic it draws them with, against
    # the closed form for a hazard that steps from 0.1 to 1.0 when a policy
    # dates the record at 1.5.
    project(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
    callback = function (c, a, b)
        (c.infector, c.susceptible) == (1, 2) && return Dirac(1.0)
        (c.infector, c.susceptible) == (1, 3) &&
            return state_policy_law(0.1, 1.0, b.date)
        return Dirac(20.0)
    end
    kernel = PairKernel(callback; state = project, watches = (:policy_time,))
    model = ModelSpec(
        RoutedNetwork(
            [
                RouteWindow(
                    :only; until = (:recovered,), kernel = kernel,
                    reach = [[2, 3], [1, 3], [1, 2]]
                ),
            ]
        );
        progression = [Transition(:recovered; delay = 5.0, terminal = true)],
        interventions = [RecordKernelPolicy()]
    )
    n = 1500
    by_three = count(1:n) do seed
        state = simulate(model; initial_cases = [1], rng = StableRNG(seed))
        is_infected(state.individuals[3]) && state.individuals[3].infection_time <= 3.0
    end
    @test by_three / n ≈ 1 - exp(-0.1 * 1.5 - 1.0 * 1.5) atol = 0.04
end
