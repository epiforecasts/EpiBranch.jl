# Tests for the Graphs.jl extension: ingesting an `AbstractGraph` as the
# contact network. Loading Graphs.jl activates the `NetworkProcess(g, kernel)`
# method, so a model can be built from any Graphs.jl generator.

@testset "Graphs.jl ingestion" begin
    # An undirected graph's neighbours become each node's contacts, matching a
    # hand-built adjacency list.
    g = path_graph(4)                       # 1—2—3—4
    m = NetworkProcess(g, Exponential(2.0))
    @test m isa NetworkProcess
    @test length(m.adjacency) == nv(g)
    @test m.adjacency[1] == [2]
    @test sort(m.adjacency[2]) == [1, 3]
    @test m.adjacency[4] == [3]

    # Keyword arguments pass through to the adjacency-list constructor.
    mh = NetworkProcess(g, Exponential(2.0); external_hazard = 0.05)
    @test mh.external_hazard == 0.05

    # The graph is the population, and a fast kernel on a clique reaches every
    # node, so an outbreak saturates at the node count.
    model = ModelSpec(NetworkProcess(complete_graph(6), Exponential(0.5));
        progression = [Transition(:recovered; from = :infection, delay = 10.0,
            terminal = true)])
    state = simulate(model; n_initial = 1, rng = StableRNG(1))
    @test state.cumulative_cases == 6

    # A directed graph uses out-neighbours: the nodes a case can infect.
    dg = SimpleDiGraph(3)
    add_edge!(dg, 1, 2)
    add_edge!(dg, 1, 3)
    md = NetworkProcess(dg, Exponential(1.0))
    @test sort(md.adjacency[1]) == [2, 3]
    @test isempty(md.adjacency[2])
end
