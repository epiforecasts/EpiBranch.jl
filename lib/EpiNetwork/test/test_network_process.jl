# Tests for NetworkProcess: a rate-based (contact-interval) network run
# on the shared continuous-time Sellke race.

# A ring graph on `n` nodes: each node linked to its two neighbours.
ring_adjacency(n) = [[mod1(i - 1, n), mod1(i + 1, n)] for i in 1:n]

# Number of infected nodes in a finished simulation.
n_infected(state) = count(is_infected, state.individuals)

@testset "NetworkProcess" begin
    @testset "construction" begin
        ring = ring_adjacency(5)
        m = NetworkProcess(ring, Exponential(2.0); infectious_period = 6.0)
        @test m isa NetworkProcess
        @test length(m.adjacency) == 5
        @test m.adjacency[1] == [5, 2]
        @test m.from == :infection            # no latent period
        @test length(m.progression) == 1      # the recovery removal transition
        @test m.external_hazard == 0.0
        @test EpiNetwork.population_size(m) isa EpiBranch.NoPopulation
        @test EpiNetwork.interventions(m) == AbstractIntervention[]

        # a latent period anchors the kernel at :infectious and adds a transition
        ml = NetworkProcess(ring, Exponential(2.0);
            latent_period = LogNormal(1.0, 0.4), infectious_period = Gamma(6, 1))
        @test ml.from == :infectious
        @test length(ml.progression) == 2

        # an explicit progression is used as given, and sets the anchor
        prog = AbstractClinicalTransition[
            Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.4)),
            Transition(:recovered; from = :infectious, delay = Gamma(6, 1), terminal = true)]
        mp = NetworkProcess(ring, Exponential(2.0); progression = prog)
        @test mp.from == :infectious
        @test length(mp.progression) == 2

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
        mA = NetworkProcess(A, Exponential(2.0); infectious_period = 6.0)
        @test Set(Set.(mA.adjacency)) == Set(Set.(ring))

        # a per-edge kernel must line up with the adjacency list
        edge_k = [[Exponential(2.0) for _ in nbrs] for nbrs in ring]
        @test NetworkProcess(ring, edge_k; infectious_period = 6.0) isa
              NetworkProcess
        bad = [[Exponential(2.0)] for _ in ring]   # wrong per-node lengths
        @test_throws ArgumentError NetworkProcess(ring, bad; infectious_period = 6.0)
    end

    @testset "moderate contact rate spreads to some of a ring" begin
        n = 30
        m = NetworkProcess(ring_adjacency(n), Exponential(2.0);
            infectious_period = 5.0)
        state = simulate(m; rng = StableRNG(1))
        k = n_infected(state)
        @test 1 < k <= n                        # spread beyond the index, bounded by the graph
    end

    @testset "very short contact interval infects the whole connected graph" begin
        n = 40
        # contact intervals (mean 0.05) almost always fall within a long
        # infectious period (20), so every edge transmits: the whole ring.
        m = NetworkProcess(ring_adjacency(n), Exponential(0.05);
            infectious_period = 20.0)
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

        baseline = NetworkProcess(ring, kernel;
            progression = AbstractClinicalTransition[
                Transition(:recovered; from = :infection,
                delay = (rng, ind) -> 20.0, terminal = true)])
        isolating = NetworkProcess(ring, kernel;
            progression = AbstractClinicalTransition[
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
        m = NetworkProcess(ring_adjacency(20), Exponential(1.0);
            latent_period = LogNormal(1.0, 0.3), infectious_period = 6.0)
        state = simulate(m; rng = StableRNG(4))
        @test EpiBranch._timetype(state) === Float64
        df = linelist(state)
        @test size(df, 1) == n_infected(state)          # one row per infected case
        @test :date_infectious in propertynames(df)     # the latent-period transition
        @test :date_recovered in propertynames(df)      # the removal transition
    end

    @testset "external force of infection introduces community cases" begin
        n = 50
        m = NetworkProcess(ring_adjacency(n), Exponential(2.0);
            infectious_period = 6.0, external_hazard = 0.05)
        state = simulate(m; rng = StableRNG(5), obs_end = 30.0)
        df = linelist(state)
        @test count(df.index) >= 1                       # community introductions happened
        @test size(df, 1) > count(df.index)              # plus onward spread on the graph
    end

    @testset "seeding multiple index nodes" begin
        n = 40
        m = NetworkProcess(ring_adjacency(n), Exponential(50.0);
            infectious_period = 0.001)                    # kernel far out of the window
        state = simulate(m; rng = StableRNG(6), n_initial = 4)
        df = linelist(state)
        @test count(df.index) == 4                        # four distinct seeds, no spread
        @test n_infected(state) == 4
    end
end
