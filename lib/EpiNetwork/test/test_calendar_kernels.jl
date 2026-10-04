include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "calendar_kernels.jl"))
test_calendar_simulation(k -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k), network_infections)

@testset "Calendar kernels with per-edge distributions" begin
    adjacency = [[2, 3], [1, 3], [1, 2]]
    calendar = Weibull(2.0, 3.0)
    edges = [[calendar, calendar] for _ in 1:3]
    shared = NetworkProcess(adjacency, CalendarKernel(calendar))
    per_edge = NetworkProcess(adjacency, CalendarKernel(edges))
    a = simulate(shared; rng = StableRNG(234))
    b = simulate(per_edge; rng = StableRNG(234))
    @test isequal(
        [i.infection_time for i in a.individuals],
        [i.infection_time for i in b.individuals]
    )
    data = network_infections(a, shared)
    @test loglikelihood(data, shared) ≈ loglikelihood(data, per_edge)
end

@testset "calendar_time keyword matches an explicit calendar kernel" begin
    adjacency = [[2, 3], [1, 3], [1, 2]]
    calendar = Weibull(2.0, 3.0)
    progression = [
        Transition(:infectious; delay = Uniform(0.4, 0.8)),
        Transition(:recovered; from = :infectious, delay = 4.0, terminal = true),
    ]
    keyword = ModelSpec(NetworkProcess(adjacency, calendar; calendar_time = true); progression)
    wrapped = ModelSpec(NetworkProcess(adjacency, CalendarKernel(calendar)); progression)
    @test keyword.process.edge_kernel isa CalendarKernel
    a = simulate(keyword; rng = StableRNG(234))
    b = simulate(wrapped; rng = StableRNG(234))
    @test isequal(
        [i.infection_time for i in a.individuals],
        [i.infection_time for i in b.individuals]
    )
    data = network_infections(a, keyword)
    @test loglikelihood(data, keyword) == loglikelihood(data, wrapped)
    @test !(NetworkProcess(adjacency, calendar).edge_kernel isa CalendarKernel)
end
