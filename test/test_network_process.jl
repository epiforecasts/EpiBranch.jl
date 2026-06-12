@testset "Network process" begin
    # A small ring graph used by several tests.
    ring(n) = [sort([mod1(i - 1, n), mod1(i + 1, n)]) for i in 1:n]

    sim1(model; attributes = NoAttributes(),
        interventions = AbstractIntervention[], kwargs...) = simulate(
        with_attributes(with_interventions(model, interventions), attributes);
        n_initial = 1,
        stopping_rules = [Extinction(), MaxGenerations(100)],
        kwargs...)

    @testset "Constructors" begin
        adj = ring(5)
        m = NetworkProcess(adj, 0.5, LogNormal(1.6, 0.5))
        @test m isa NetworkProcess
        @test length(m.adjacency) == 5

        # Weighted adjacency matrix is read as undirected; entries are
        # per-edge transmission probabilities.
        A = [0.0 0.5 0.0; 0.5 0.0 0.5; 0.0 0.5 0.0]
        mm = NetworkProcess(A, LogNormal(1.6, 0.5))
        @test mm.adjacency[2] == [1, 3]
        @test mm.adjacency[1] == [2]
        @test mm.edge_probability[2] == [0.5, 0.5]

        # Non-square matrix is rejected.
        @test_throws ArgumentError NetworkProcess([0 1; 1 0; 0 0])

        # population_size is determined by the graph; passing one warns.
        @test_logs (:warn,) NetworkProcess(adj, 0.5, LogNormal(1.6, 0.5);
            population_size = 1000)
    end

    @testset "Population is the graph" begin
        n = 50
        model = NetworkProcess(ring(n), 0.9, LogNormal(1.6, 0.5))
        state = sim1(model; rng = StableRNG(1))
        # Every individual is a pre-instantiated node; nothing is minted.
        @test length(state.individuals) == n
        @test state.cumulative_cases <= n
    end

    @testset "Node identity: no reinfection" begin
        model = NetworkProcess(ring(60), 0.9, LogNormal(1.6, 0.5))
        for seed in 1:10
            state = sim1(model; rng = StableRNG(seed))
            nodes = [ind.state[:network_node]
                     for ind in state.individuals if is_infected(ind)]
            @test length(nodes) == length(unique(nodes))
        end
    end

    @testset "Transmission stays on the graph" begin
        # Two disconnected triangles: an outbreak seeded in one cannot
        # cross to the other.
        adj = [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5]]
        model = NetworkProcess(adj, 1.0, LogNormal(1.6, 0.5))
        # Seed node 1; only nodes 1–3 can ever be infected.
        state = simulate(model;
            n_initial = 1,
            stopping_rules = [Extinction(), MaxGenerations(20)],
            rng = StableRNG(1))
        # With n_initial = 1 the seed is a random node; check the
        # component containment property over several seeds.
        for seed in 1:20
            s = simulate(model;
                n_initial = 1,
                stopping_rules = [Extinction(), MaxGenerations(20)],
                rng = StableRNG(seed))
            infected = [ind.state[:network_node]
                        for ind in s.individuals if is_infected(ind)]
            comp1 = all(x -> x in (1, 2, 3), infected)
            comp2 = all(x -> x in (4, 5, 6), infected)
            @test comp1 || comp2
        end
    end

    @testset "Transmission probability scales outbreak size" begin
        n = 200
        # A connected ring with extra chords so outbreaks can grow.
        adj = [sort(unique([mod1(i - 1, n), mod1(i + 1, n),
                   mod1(i + 7, n), mod1(i - 7, n)])) for i in 1:n]
        sizes = Float64[]
        for p in (0.1, 0.5, 0.9)
            model = NetworkProcess(adj, p, LogNormal(1.6, 0.5))
            res = simulate(model, 100;
                n_initial = 1,
                stopping_rules = [Extinction(), MaxGenerations(100)],
                rng = StableRNG(1))
            push!(sizes, sum(s.cumulative_cases for s in res) / 100)
        end
        @test issorted(sizes)               # higher p ⇒ larger outbreaks
        @test sizes[end] > sizes[1]
    end

    @testset "Attributes fixed per node; onset derived at infection" begin
        model = NetworkProcess(ring(80), 0.8, LogNormal(1.6, 0.5))
        attrs = compose(
            demographics(age_distribution = Uniform(0, 80)),
            clinical_presentation(incubation_period = LogNormal(1.6, 0.5)))
        state = sim1(model; attributes = attrs, rng = StableRNG(3))
        for ind in filter(is_infected, state.individuals)
            # onset = infection_time + incubation (host property).
            @test haskey(ind.state, :incubation_period)
            inc = ind.state[:incubation_period]
            if !isnan(inc)
                @test ind.state[:onset_time] ≈ ind.infection_time + inc
            end
        end
    end

    @testset "Age-dependent susceptibility flows through" begin
        n = 300
        adj = [sort(unique([mod1(i - 1, n), mod1(i + 1, n),
                   mod1(i + 5, n), mod1(i - 5, n)])) for i in 1:n]
        model = NetworkProcess(adj, 0.5, LogNormal(1.6, 0.5))
        attrs = compose(
            demographics(age_distribution = Uniform(0, 80)),
            clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
            transmission_traits(
                susceptibility = (rng, ind) -> ind.state[:age] >= 60 ? 0.95 : 0.15))
        old_pop = 0
        tot_pop = 0
        old_inf = 0
        tot_inf = 0
        for s in simulate(with_attributes(model, attrs), 40; n_initial = 3,
            stopping_rules = [Extinction(), MaxGenerations(100)], rng = StableRNG(11))
            for ind in s.individuals
                tot_pop += 1
                ind.state[:age] >= 60 && (old_pop += 1)
                if is_infected(ind)
                    tot_inf += 1
                    ind.state[:age] >= 60 && (old_inf += 1)
                end
            end
        end
        # Higher-susceptibility (older) nodes over-represented among cases.
        @test old_inf / tot_inf > old_pop / tot_pop
    end

    @testset "Hubs are infected before leaves" begin
        # Star-of-stars: a few high-degree hubs, many degree-1 leaves.
        n_hubs = 5
        leaves_per_hub = 30
        n = n_hubs + n_hubs * leaves_per_hub
        adj = [Int[] for _ in 1:n]
        # connect hubs in a clique
        for i in 1:n_hubs, j in 1:n_hubs

            i != j && push!(adj[i], j)
        end
        # attach leaves to each hub
        leaf = n_hubs
        for h in 1:n_hubs, _ in 1:leaves_per_hub

            leaf += 1
            push!(adj[h], leaf)
            push!(adj[leaf], h)
        end
        degree = length.(adj)
        model = NetworkProcess(adj, 0.5, LogNormal(1.6, 0.5))

        hub_gens = Int[]
        leaf_gens = Int[]
        for rep in 1:30
            s = simulate(model;
                n_initial = 1,
                stopping_rules = [Extinction(), MaxGenerations(100)],
                rng = StableRNG(100 + rep))
            s.cumulative_cases < 20 && continue
            for ind in filter(is_infected, s.individuals)
                node = ind.state[:network_node]
                degree[node] >= leaves_per_hub ? push!(hub_gens, ind.generation) :
                push!(leaf_gens, ind.generation)
            end
        end
        # Hubs are reached at earlier generations than leaves on average.
        @test !isempty(hub_gens) && !isempty(leaf_gens)
        @test sum(hub_gens) / length(hub_gens) < sum(leaf_gens) / length(leaf_gens)
    end

    @testset "Interventions reduce outbreak size" begin
        n = 200
        adj = [sort(unique([mod1(i - 1, n), mod1(i + 1, n),
                   mod1(i + 7, n), mod1(i - 7, n)])) for i in 1:n]
        model = NetworkProcess(adj, 0.6, LogNormal(1.6, 0.5))
        opts = (; n_initial = 1,
            stopping_rules = [Extinction(), MaxGenerations(100)])
        attrs = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))

        base = simulate(with_attributes(model, attrs), 100; opts..., rng = StableRNG(5))
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
        ct = ContactTracing(probability = 0.7,
            isolation_to_trace_delay = Exponential(1.5))
        treated = simulate(with_attributes(with_interventions(model, [iso, ct]), attrs),
            100; opts..., rng = StableRNG(5))

        mean_base = sum(s.cumulative_cases for s in base) / 100
        mean_treated = sum(s.cumulative_cases for s in treated) / 100
        @test mean_treated < mean_base
    end
end
