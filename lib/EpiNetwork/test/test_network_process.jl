# Tests for NetworkProcess: a rate-based (contact-interval) network run
# on the shared continuous-time Sellke race.

# A ring graph on `n` nodes: each node linked to its two neighbours.
ring_adjacency(n) = [[mod1(i - 1, n), mod1(i + 1, n)] for i in 1:n]

# Number of infected nodes in a finished simulation.
n_infected(state) = count(is_infected, state.individuals)

# The within-host natural history, composed onto the process with a ModelSpec:
# a recovery removal (SIR) unless a latent step is prepended (SEIR).
_sir(ip) = [Transition(:recovered; from = :infection, delay = ip, terminal = true)]

@testset "NetworkProcess" begin
    @testset "construction" begin
        ring = ring_adjacency(5)
        m = NetworkProcess(ring, Exponential(2.0))
        @test m isa NetworkProcess
        @test length(m.adjacency) == 5
        @test m.adjacency[1] == [5, 2]
        @test m.from === nothing              # window start derived from the progression
        @test m.external_hazard == 0.0
        @test EpiBranch.population_size(m) isa EpiBranch.NoPopulation

        # with no latent transition the window opens at :infection
        @test EpiBranch._resolve_infectious_from(m.from, _sir(6.0)) === :infection

        # a latent transition anchors the window at :infectious
        seir = [Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.4)),
            Transition(:recovered; from = :infectious, delay = Gamma(6, 1),
                terminal = true)]
        @test EpiBranch._resolve_infectious_from(nothing, seir) === :infectious

        # a scalar and a distribution external hazard are both accepted
        @test NetworkProcess(ring, Exponential(2.0); external_hazard = 0.05) isa
              NetworkProcess
        @test NetworkProcess(ring, Exponential(2.0); external_hazard = Exponential(20.0)) isa
              NetworkProcess
        @test_throws ArgumentError NetworkProcess(ring, Exponential(2.0);
            external_hazard = -1.0)

        # a matrix builds the same ring structure
        A = zeros(5, 5)
        for i in 1:5
            A[i, mod1(i + 1, 5)] = 1.0
            A[mod1(i + 1, 5), i] = 1.0
        end
        mA = NetworkProcess(A, Exponential(2.0))
        @test Set(Set.(mA.adjacency)) == Set(Set.(ring))

        # a per-edge kernel must line up with the adjacency list
        edge_k = [[Exponential(2.0) for _ in nbrs] for nbrs in ring]
        @test NetworkProcess(ring, edge_k) isa NetworkProcess
        bad = [[Exponential(2.0)] for _ in ring]   # wrong per-node lengths
        @test_throws ArgumentError NetworkProcess(ring, bad)
    end

    @testset "moderate contact rate spreads to some of a ring" begin
        n = 30
        m = ModelSpec(NetworkProcess(ring_adjacency(n), Exponential(2.0));
            progression = _sir(5.0))
        state = simulate(m; rng = StableRNG(1))
        k = n_infected(state)
        @test 1 < k <= n                        # spread beyond the index, bounded by the graph
    end

    @testset "very short contact interval infects the whole connected graph" begin
        n = 40
        # contact intervals (mean 0.05) almost always fall within a long
        # infectious period (20), so every edge transmits: the whole ring.
        m = ModelSpec(NetworkProcess(ring_adjacency(n), Exponential(0.05));
            progression = _sir(20.0))
        state = simulate(m; rng = StableRNG(2))
        @test n_infected(state) == n
    end

    @testset "isolation cuts transmission" begin
        # A rate-based network lets a *shortened infectious window* curtail
        # onward transmission — the thing a coin-flip-per-edge model cannot
        # express. Same graph, same fast kernel; isolating each case at 0.3
        # (well before the 20-unit recovery) closes its window early.
        n = 60
        ring = ring_adjacency(n)
        kernel = Exponential(0.5)

        baseline = ModelSpec(NetworkProcess(ring, kernel);
            progression = [
                Transition(:recovered; from = :infection,
                delay = (rng, ind) -> 20.0, terminal = true)])
        isolating = ModelSpec(NetworkProcess(ring, kernel);
            progression = [
                Transition(:recovered; from = :infection,
                    delay = (rng, ind) -> 20.0, terminal = true),
                Transition(:isolated; from = :infection,
                    delay = (rng, ind) -> 0.3)])   # :isolated ∈ until closes the window

        k_base = n_infected(simulate(baseline; rng = StableRNG(3)))
        k_iso = n_infected(simulate(isolating; rng = StableRNG(3)))
        @test k_base == n                          # unimpeded: the whole ring
        @test k_iso < k_base                        # isolation curtails the outbreak
    end

    @testset "state carries Float64 timing and renders a line list" begin
        m = ModelSpec(NetworkProcess(ring_adjacency(20), Exponential(1.0));
            progression = [
                Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.3)),
                Transition(:recovered; from = :infectious, delay = 6.0, terminal = true)])
        state = simulate(m; rng = StableRNG(4))
        @test EpiBranch._timetype(state) === Float64
        df = linelist(state)
        @test size(df, 1) == n_infected(state)          # one row per infected case
        @test :date_infectious in propertynames(df)     # the latent-period transition
        @test :date_recovered in propertynames(df)      # the removal transition
    end

    @testset "external force of infection introduces community cases" begin
        n = 50
        m = ModelSpec(
            NetworkProcess(ring_adjacency(n), Exponential(2.0);
                external_hazard = 0.05, obs_end = 30.0);
            progression = _sir(6.0))
        state = simulate(m; rng = StableRNG(5))
        df = linelist(state)
        @test count(df.index) >= 1                       # community introductions happened
        @test size(df, 1) > count(df.index)              # plus onward spread on the graph
    end

    @testset "seeding multiple index nodes" begin
        n = 40
        m = ModelSpec(NetworkProcess(ring_adjacency(n), Exponential(50.0));
            progression = _sir(0.001))                    # kernel far out of the window
        state = simulate(m; rng = StableRNG(6), n_initial = 4)
        df = linelist(state)
        @test count(df.index) == 4                        # four distinct seeds, no spread
        @test n_infected(state) == 4
    end

    @testset "per-edge and callable kernels spread on the graph" begin
        n = 30
        ring = ring_adjacency(n)
        # a per-edge kernel: one fast contact-interval distribution per listed edge
        edge_k = [[Exponential(0.05) for _ in nbrs] for nbrs in ring]
        state = simulate(
            ModelSpec(NetworkProcess(ring, edge_k); progression = _sir(20.0));
            rng = StableRNG(11))
        @test n_infected(state) == n                      # every edge transmits: whole ring

        # a callable kernel: (infector, susceptible) -> Distribution, for covariates
        callable = (i, j) -> Exponential(0.05)
        state2 = simulate(
            ModelSpec(NetworkProcess(ring, callable); progression = _sir(20.0));
            rng = StableRNG(12))
        @test n_infected(state2) == n
    end

    @testset "calendar-time external hazard introduces community cases" begin
        n = 40
        # a distribution external hazard (introduction times), not a constant rate
        m = ModelSpec(
            NetworkProcess(ring_adjacency(n), Exponential(2.0);
                external_hazard = Uniform(0.0, 20.0), obs_end = 25.0);
            progression = _sir(6.0))
        df = linelist(simulate(m; rng = StableRNG(13)))
        @test count(df.index) >= 1

        # an external hazard with the default (infinite) obs_end is rejected:
        # over an unbounded window it would seed every node
        @test_throws ArgumentError simulate(
            ModelSpec(
                NetworkProcess(ring_adjacency(n), Exponential(2.0);
                    external_hazard = 0.05);
                progression = _sir(6.0));
            rng = StableRNG(14))
    end

    @testset "show and invalid external hazard" begin
        ring = ring_adjacency(5)
        # show, with and without an active external hazard
        @test occursin("NetworkProcess", repr(NetworkProcess(ring, Exponential(2.0))))
        @test occursin("external_hazard",
            repr(NetworkProcess(ring, Exponential(2.0); external_hazard = 0.05)))
        # an external hazard that is neither a number nor a distribution is rejected
        @test_throws ArgumentError NetworkProcess(ring, Exponential(2.0);
            external_hazard = "not a hazard")
    end

    @testset "conditioned simulation" begin
        # `condition` retries until the outbreak size falls in the range
        m = ModelSpec(NetworkProcess(ring_adjacency(40), Exponential(0.5));
            progression = _sir(20.0))
        state = simulate(m; condition = 5:40, n_initial = 1, rng = StableRNG(1))
        @test state.cumulative_cases in 5:40
    end
end
