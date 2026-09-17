# Tests for NetworkProcess: a rate-based (contact-interval) network run
# on the shared continuous-time Sellke race.

# A ring graph on `n` nodes: each node linked to its `k` nearest neighbours on
# either side.
function ring_adjacency(n, k = 1)
    [vcat([mod1(i - d, n) for d in 1:k], [mod1(i + d, n) for d in 1:k]) for i in 1:n]
end

# Number of infected nodes in a finished simulation.
n_infected(state) = count(is_infected, state.individuals)

# The within-host natural history, composed onto the process with a ModelSpec:
# a recovery removal (SIR) unless a latent step is prepended (SEIR).
_sir(ip) = [Transition(:recovered; from = :infection, delay = ip, terminal = true)]

# A per-contact risk written from outside the package, blocking every
# transmission it is asked about.
struct BlockEverything <: EpiBranch.AbstractIntervention end
function EpiBranch.competing_risk(::BlockEverything, parent, contact, state)
    Risk(block_probability = 1.0)
end

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

        # obs_end must be non-negative (Inf allowed); negative and NaN are rejected
        @test NetworkProcess(ring, Exponential(2.0); obs_end = Inf) isa NetworkProcess
        @test NetworkProcess(ring, Exponential(2.0); obs_end = 0.0) isa NetworkProcess
        @test_throws ArgumentError NetworkProcess(ring, Exponential(2.0); obs_end = -1.0)
        @test_throws ArgumentError NetworkProcess(ring, Exponential(2.0); obs_end = NaN)
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

    @testset "Isolation intervention curtails the outbreak" begin
        # The Isolation *intervention* (not a Transition) now runs on the
        # continuous-time network path: its resolve_individual! fires in the
        # Sellke race and its isolation time closes the infectious window.
        n = 60
        ring = ring_adjacency(n)
        kernel = Exponential(0.5)
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:recovered; from = :infection,
                delay = (rng, ind) -> 20.0, terminal = true)]

        baseline = ModelSpec(NetworkProcess(ring, kernel); progression = prog)
        isolating = ModelSpec(NetworkProcess(ring, kernel); progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.2))])

        @test n_infected(simulate(isolating; rng = StableRNG(3))) <
              n_infected(simulate(baseline; rng = StableRNG(3)))
    end

    @testset "per-individual susceptibility and infectiousness apply" begin
        # The race resolves the composed competing risks on each infection it
        # proposes along an edge, so both multipliers mean here what they mean
        # on the generation engine: a per-contact block.
        n = 300
        ring = ring_adjacency(n, 2)
        build(attrs) = ModelSpec(NetworkProcess(ring, Exponential(1.5));
            progression = _sir(Exponential(4.0)), attributes = attrs)
        meansize(attrs) = sum(simulate(build(attrs);
                                  rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:10) / 10

        full = meansize(transmission_traits(susceptibility = 1.0))
        half = meansize(transmission_traits(susceptibility = 0.5))
        @test half < full
        @test meansize(transmission_traits(susceptibility = 0.2)) < half

        # Susceptibility 0 blocks every proposal, so only the seeds are infected.
        blocked = simulate(build(transmission_traits(susceptibility = 0.0));
            rng = StableRNG(1), n_initial = 3)
        @test blocked.cumulative_cases == 3

        @test meansize(transmission_traits(infectiousness = 0.5)) < full
        silent = simulate(build(transmission_traits(infectiousness = 0.0));
            rng = StableRNG(1), n_initial = 3)
        @test silent.cumulative_cases == 3
    end

    @testset "leaky ring vaccination changes network results" begin
        # Ring vaccination doses the contacts a case reaches when the race
        # settles it, and its efficacy is then a per-contact block against every
        # infection proposed to a dosed node afterwards.
        n = 300
        ring = ring_adjacency(n, 2)
        clinical = clinical_presentation(incubation_period = LogNormal(0.0, 0.3),
            prob_asymptomatic = 0.0)
        # Isolation is the trigger tracing fires from, and nothing else: a
        # residual of 1 leaves transmission untouched, so what the outbreak sizes
        # below show is the vaccine on its own.
        iso = Isolation(onset_to_isolation_delay = Exponential(0.5),
            test_sensitivity = 1.0, post_isolation_transmission = 1.0)
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.2),
            quarantine_on_trace = false)
        build(ivs) = ModelSpec(NetworkProcess(ring, Exponential(4.0));
            progression = _sir(Exponential(8.0)), interventions = ivs,
            attributes = clinical)
        runs(ivs) = [simulate(build(ivs); rng = StableRNG(s), n_initial = 3)
                     for s in 1:10]
        meansize(ivs) = sum(st.cumulative_cases for st in runs(ivs)) / 10

        base = meansize([iso, ct])
        @test !any(st -> any(is_vaccinated, st.individuals), runs([iso, ct]))
        dosed_runs = runs([iso, ct, RingVaccination(efficacy = 0.5)])
        @test all(st -> any(is_vaccinated, st.individuals), dosed_runs)
        # A node reached by several cases is dosed at its earliest trace, even
        # when the case that reached it first was settled later.
        @test all(
            st -> all(
                ind -> !is_vaccinated(ind) ||
                       ind.state[:vaccination_time] == ind.state[:trace_time],
                st.individuals),
            dosed_runs)

        @test isapprox(meansize([iso, ct, RingVaccination(efficacy = 0.0)]), base;
            rtol = 0.05)
        # A dosed pair goes on meeting, so an efficacy of 0.5 thins that edge's
        # hazard by half rather than halving its transmissions: it cuts the
        # outbreak, but by less than the same efficacy would on the generation
        # engine, where a blocked contact is simply lost.
        leaky = meansize([iso, ct, RingVaccination(efficacy = 0.5)])
        @test leaky < 0.8 * base
        @test meansize([iso, ct, RingVaccination(efficacy = 1.0)]) < leaky
    end

    @testset "a traced node is offered a ring dose once" begin
        # A node is traced again by every neighbour that settles after it, and a
        # coverage draw on each would vaccinate far more than `coverage` of them.
        clinical = clinical_presentation(incubation_period = LogNormal(0.0, 0.3),
            prob_asymptomatic = 0.0)
        ivs = [
            Isolation(onset_to_isolation_delay = Exponential(0.5), test_sensitivity = 1.0,
                post_isolation_transmission = 1.0),
            ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.2),
                quarantine_on_trace = false),
            RingVaccination(efficacy = 0.0, coverage = 0.5)]
        model = ModelSpec(NetworkProcess(ring_adjacency(300, 2), Exponential(4.0));
            progression = _sir(Exponential(8.0)), interventions = ivs,
            attributes = clinical)
        traced = [ind
                  for s in 1:20
                  for ind in simulate(model; rng = StableRNG(s), n_initial = 3).individuals
                  if is_traced(ind) && isfinite(ind.state[:trace_time])]
        @test length(traced) > 1000
        @test isapprox(count(is_vaccinated, traced) / length(traced), 0.5; atol = 0.05)
    end

    @testset "a dose given along another case's trace protects before exposure" begin
        # With incomplete tracing and asymptomatic cases, a node is often dosed by
        # the trace of a case other than the one that later infects it, and that
        # trace can settle after the infector did. The race resolves each
        # proposal once everything infected before its time has settled, so a
        # fully effective dose in place by then blocks it.
        clinical = clinical_presentation(incubation_period = LogNormal(0.0, 0.3),
            prob_asymptomatic = 0.4)
        ivs = [
            Isolation(onset_to_isolation_delay = Exponential(0.5), test_sensitivity = 1.0,
                post_isolation_transmission = 1.0),
            ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(0.2),
                quarantine_on_trace = false),
            RingVaccination(efficacy = 1.0)]
        model = ModelSpec(NetworkProcess(ring_adjacency(400, 3), Exponential(3.0));
            progression = _sir(Exponential(6.0)), interventions = ivs,
            attributes = clinical)
        immune_at_infection(ind) = is_vaccinated(ind) && ind.parent_id != 0 &&
                                   ind.state[:immunity_time] <= ind.infection_time
        runs = [simulate(model; rng = StableRNG(s), n_initial = 3) for s in 1:15]
        @test any(st -> count(is_vaccinated, st.individuals) > 0, runs)
        @test !any(st -> any(immune_at_infection, st.individuals), runs)
    end

    @testset "onset is measured from each case's own infection time" begin
        # Nodes are created, and their incubation periods drawn, before the
        # race sets their infection times. Isolation depends on onset, so onset
        # must be counted from the time each case was infected.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        m = ModelSpec(NetworkProcess(ring_adjacency(200), Exponential(2.0));
            progression = _sir(10.0), attributes = clinical, interventions = [iso])
        state = simulate(m; n_initial = 1, rng = StableRNG(3))
        secondary = [ind
                     for ind in state.individuals
                     if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(secondary)
        @test all(onset_time(ind) >= ind.infection_time for ind in secondary)
        @test all(onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
        for ind in secondary)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in secondary)

        # Tracing that flags contacts without quarantining them isolates a
        # traced contact no earlier than its own onset, which is only known once
        # the race has infected it.
        flag = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5), quarantine_on_trace = false)
        traced_model = ModelSpec(NetworkProcess(ring_adjacency(300), Exponential(2.0));
            progression = _sir(10.0), attributes = clinical, interventions = [iso, flag])
        traced = [ind
                  for s in 1:20
                  for ind in simulate(traced_model; n_initial = 1, rng = StableRNG(s)).individuals
                  if is_infected(ind) && is_traced(ind) && isfinite(isolation_time(ind))]
        @test !isempty(traced)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in traced)

        routed = ModelSpec(
            RoutedNetwork([RouteWindow(:contact; until = (:recovered,),
                kernel = Exponential(2.0), reach = ring_adjacency(200))]);
            progression = _sir(10.0), attributes = clinical)
        state = simulate(routed; n_initial = 1, rng = StableRNG(3))
        secondary = [ind
                     for ind in state.individuals
                     if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(secondary)
        @test all(onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
        for ind in secondary)
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

    @testset "risks reach community introductions" begin
        # An introduction comes from outside the population, and is put to the
        # same risks as a contact from a neighbour: the person's susceptibility
        # scales the community hazard, and an intervention's block stops the
        # introduction it is in force for.
        n = 200
        infected(state) = count(is_infected, state.individuals)
        function community(ext, susceptibility, ivs = AbstractIntervention[])
            m = ModelSpec(
                # a contact interval far beyond the window leaves only
                # community introductions
                NetworkProcess(ring_adjacency(n), Exponential(1e6);
                    external_hazard = ext, obs_end = 30.0);
                progression = _sir(6.0), interventions = ivs,
                attributes = transmission_traits(; susceptibility))
            return infected(simulate(m; rng = StableRNG(5)))
        end
        for ext in (0.05, Exponential(20.0))
            @test community(ext, 0.0) == 0
            @test 0 < community(ext, 0.2) < community(ext, 1.0)
            # A user's own risk blocking everything blocks them too.
            @test community(ext, 1.0, [BlockEverything()]) == 0
        end
    end

    @testset "a fixed seed reproduces a pinned outbreak" begin
        # The race draws from the RNG stream in settling order, so these tests pin
        # that order, including ties. Several index cases share time 0, and a
        # deterministic kernel makes most candidate times coincide. Equal times
        # settle in member order.
        ring = ring_adjacency(10)
        st = simulate(
            ModelSpec(NetworkProcess(ring, Exponential(1.0));
                progression = _sir(3.0));
            n_initial = 3, rng = StableRNG(42))
        @test [ind.infection_time for ind in st.individuals] ≈
              [0.9132388449809826, 0.0, 1.4907960097331951, 3.0946387152619605,
            1.5146027102204256, 0.02580085388107159, 0.0, 0.6196859650483677,
            0.7149127125395202, 0.0]
        @test [ind.parent_id for ind in st.individuals] == [10, 0, 2, 5, 6, 7, 0, 7, 10, 0]

        st = simulate(
            ModelSpec(NetworkProcess(ring, (i, j) -> Dirac(1.0));
                progression = _sir(3.0));
            n_initial = 3, rng = StableRNG(42))
        @test [ind.infection_time for ind in st.individuals] ≈
              [1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        @test [ind.parent_id for ind in st.individuals] == [2, 0, 2, 3, 6, 7, 0, 7, 10, 0]
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

    @testset "RoutedNetwork: per-route censoring" begin
        # Two routes over the same 120 people: households of 4 as cliques, and a
        # ring of community contacts. The only difference between the scenarios
        # is which routes isolation is allowed to cut.
        nh, hs = 30, 4
        n = nh * hs
        hh = [Int[] for _ in 1:n]
        for h in 0:(nh - 1), i in (h * hs + 1):(h * hs + hs),
            j in (h * hs + 1):(h * hs + hs)
            i != j && push!(hh[i], j)
        end
        comm = ring_adjacency(n)

        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0),
            test_sensitivity = 1.0)
        hk, ck = Weibull(1.5, 4.0), Exponential(20.0)
        REM = EpiBranch.INTERVENTION_REMOVAL

        routes(hh_until) = [
            RouteWindow(:household; until = hh_until, kernel = hk, reach = hh),
            RouteWindow(:community; until = (:recovered, REM), kernel = ck,
                reach = comm)]

        run(ws, ivs) = sum(simulate(
                               ModelSpec(RoutedNetwork(ws);
                                   progression = _sir(10.0), interventions = ivs,
                                   attributes = clinical);
                               n_initial = 2, rng = StableRNG(s)).cumulative_cases
        for s in 1:40) / 40

        removed = run(routes((:recovered, REM)), [iso])   # isolation cuts both
        selfiso = run(routes((:recovered,)), [iso])       # household route survives

        # Self-isolation must be strictly worse than being removed outright:
        # the household route keeps running either way it is cut.
        @test selfiso > removed

        # Both must beat no control at all.
        @test removed < run(routes((:recovered,)), AbstractIntervention[])

        # Under self-isolation the surviving transmission is mostly within
        # households, which is the whole point of separating the routes.
        st = simulate(
            ModelSpec(RoutedNetwork(routes((:recovered,)));
                progression = _sir(10.0), interventions = [iso],
                attributes = clinical);
            n_initial = 2,
            rng = StableRNG(3))
        hh_of(i) = (i - 1) ÷ hs
        pairs = [(ind.id, ind.parent_id)
                 for ind in st.individuals
                 if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(pairs)
        @test count(p -> hh_of(p[1]) == hh_of(p[2]), pairs) > length(pairs) ÷ 2
    end

    @testset "RoutedNetwork: route start under a latent period" begin
        # Two nodes, a fixed 5-day latent period and near-immediate contact. A
        # route left at the default opens when the case becomes infectious; an
        # explicit `from = :infection` opens at infection.
        seir = [Transition(:infectious; from = :infection, delay = 5.0),
            Transition(:recovered; from = :infection, delay = 20.0, terminal = true)]
        contact_time(from) = begin
            w = RouteWindow(:pair; from, until = (:recovered,),
                kernel = Exponential(0.1), reach = [[2], [1]])
            st = simulate(ModelSpec(RoutedNetwork([w]); progression = seir);
                n_initial = 1, rng = StableRNG(1))
            maximum(ind.infection_time for ind in st.individuals)
        end
        @test contact_time(nothing) >= 5.0
        @test contact_time(:infection) < 5.0
    end

    @testset "RoutedNetwork: construction" begin
        a = ring_adjacency(6)
        w(name, adj) = RouteWindow(name; until = (:recovered,),
            kernel = Exponential(2.0), reach = adj)
        m = RoutedNetwork([w(:a, a), w(:b, a)])
        @test m.n == 6
        @test occursin("RoutedNetwork", repr(m))
        @test occursin(":a", repr(m))
        @test EpiBranch.supplies_contacts(m)
        # a model-level start fills in routes that leave theirs unset, so the
        # stored routes are the ones the simulation runs
        mf = RoutedNetwork(
            [w(:a, a), RouteWindow(:b; from = :died, kernel = Exponential(1.0),
                reach = a)];
            from = :onset)
        @test mf.windows[1].from === :onset
        @test mf.windows[2].from === :died
        # routes must agree on the node set, and there must be at least one
        @test_throws ArgumentError RoutedNetwork([w(:a, a), w(:b, ring_adjacency(5))])
        @test_throws ArgumentError RoutedNetwork(RouteWindow[])
        # a reach that is not an adjacency list is rejected
        @test_throws ArgumentError RoutedNetwork([RouteWindow(:x;
            kernel = Exponential(1.0), reach = :not_an_adjacency)])
        # as on NetworkProcess, the importation window must be non-negative
        @test_throws ArgumentError RoutedNetwork([w(:a, a)]; obs_end = -1.0)
        @test_throws ArgumentError RoutedNetwork([w(:a, a)]; obs_end = NaN)
    end

    @testset "RoutedNetwork: conditioned simulation" begin
        # `condition` retries until the outbreak size falls in the range
        route = RouteWindow(:ring; until = (:recovered,), kernel = Exponential(0.5),
            reach = ring_adjacency(40))
        m = ModelSpec(RoutedNetwork([route]); progression = _sir(20.0))
        state = simulate(m; condition = 5:40, n_initial = 1, rng = StableRNG(1))
        @test state.cumulative_cases in 5:40
    end

    @testset "RoutedNetwork: one route traces as NetworkProcess" begin
        # A single route opening at the infectious start is a NetworkProcess, so
        # isolation and tracing must give the same outbreaks, including under a
        # latent period where a case can isolate before it becomes infectious.
        REM = EpiBranch.INTERVENTION_REMOVAL
        adj = ring_adjacency(80)
        k = Exponential(1.0)
        prog = [Transition(:onset; from = :infection, delay = 1.0),
            Transition(:infectious; from = :infection, delay = 3.0),
            Transition(:recovered; from = :infection, delay = 12.0, terminal = true)]
        ivs = [Isolation(onset_to_isolation_delay = Exponential(0.5)),
            ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5))]
        routed = RoutedNetwork([RouteWindow(:all; until = (:recovered, REM),
            kernel = k, reach = adj)])
        plain = NetworkProcess(adj, k; until = (:recovered,))
        run(proc, s) = simulate(ModelSpec(proc; progression = prog, interventions = ivs);
            n_initial = 2, rng = StableRNG(s))
        for s in 1:10
            a, b = run(routed, s), run(plain, s)
            @test a.cumulative_cases == b.cumulative_cases
            @test count(is_traced, a.individuals) == count(is_traced, b.individuals)
        end
        @test sum(count(is_traced, run(routed, s).individuals) for s in 1:10) > 0
    end

    @testset "RoutedNetwork: a route's infectiousness start does not delay tracing" begin
        # Setting `from = :onset` on each route or on the model describes the
        # same outbreak, and both trace household and community contacts from
        # infection, as NetworkProcess does.
        REM = EpiBranch.INTERVENTION_REMOVAL
        adj = ring_adjacency(80)
        k = Exponential(1.0)
        prog = [Transition(:onset; from = :infection, delay = Uniform(2.0, 6.0)),
            Transition(:recovered; from = :infection, delay = 12.0, terminal = true)]
        ivs = [Isolation(onset_to_isolation_delay = Exponential(1.0)),
            ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5))]
        on_route = RoutedNetwork([RouteWindow(:all; from = :onset,
            until = (:recovered, REM), kernel = k, reach = adj)])
        on_model = RoutedNetwork(
            [RouteWindow(:all; until = (:recovered, REM),
                kernel = k, reach = adj)];
            from = :onset)
        run(proc, s) = simulate(ModelSpec(proc; progression = prog, interventions = ivs);
            n_initial = 2, rng = StableRNG(s))
        for s in 1:10
            a, b = run(on_route, s), run(on_model, s)
            @test a.cumulative_cases == b.cumulative_cases
            @test count(is_traced, a.individuals) == count(is_traced, b.individuals)
        end
    end

    @testset "RoutedNetwork: tracing follows only opened routes" begin
        # Node 1 lives with node 2 and would meet node 3 only at its funeral.
        # Nobody dies, so no funeral route ever opens and node 3 is never a
        # contact of anyone: tracing must not reach it, whichever node is the
        # index case.
        REM = EpiBranch.INTERVENTION_REMOVAL
        household = RouteWindow(:household; until = (:recovered, REM),
            kernel = Exponential(1.0), reach = [[2], [1], Int[]])
        funeral = RouteWindow(:funeral; from = :died, until = (:recovered,),
            kernel = Exponential(1.0), reach = [[3], Int[], [1]], contacts_from = :died)
        clinical = clinical_presentation(incubation_period = LogNormal(0.0, 0.3),
            prob_asymptomatic = 0.0)
        m = ModelSpec(RoutedNetwork([household, funeral]);
            progression = _sir(10.0), attributes = clinical,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.5)),
                ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Exponential(0.5))])
        for s in 1:20
            st = simulate(m; n_initial = 1, rng = StableRNG(s))
            @test !is_traced(st.individuals[3])
            # node 3's own funeral route never opens either
            is_infected(st.individuals[3]) && @test !is_traced(st.individuals[1])
        end
    end

    @testset "RoutedNetwork: a late-opening route is traced once it opens" begin
        # Node 1 dies on day 8 and meets node 3 only at its funeral. It isolates
        # and is traced long before that, but node 3 cannot be reached before
        # the funeral, while its household contact node 2 is reached at once.
        REM = EpiBranch.INTERVENTION_REMOVAL
        household = RouteWindow(:household; until = (:died, REM),
            kernel = Exponential(1.0), reach = [[2], [1], Int[]])
        funeral = RouteWindow(:funeral; from = :died, until = (),
            kernel = Exponential(1.0), reach = [[3], Int[], [1]], contacts_from = :died)
        m = ModelSpec(RoutedNetwork([household, funeral]);
            progression = [Transition(:onset; from = :infection, delay = 0.5),
                Transition(:died; from = :infection, delay = 8.0, terminal = true)],
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.5)),
                ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Exponential(0.5))])
        checked = 0
        for s in 1:60
            st = simulate(m; n_initial = 1, rng = StableRNG(s))
            case = st.individuals[1]
            (get(case.state, :index, false) && is_traced(st.individuals[3])) || continue
            checked += 1
            died = case.state[:died_time]
            # a quarantined contact's isolation time is its trace time, and the
            # trace delay runs from the funeral rather than being absorbed by
            # the case's much earlier isolation
            @test isolation_time(st.individuals[3]) > died
            @test isolation_time(st.individuals[2]) < died
        end
        @test checked > 0

        # A safe burial: the funeral route is cut by isolation too, so a case
        # isolated before it dies never holds a funeral and its funeral
        # neighbour is never traced.
        safe = RouteWindow(:funeral; from = :died, until = (REM,),
            kernel = Exponential(1.0), reach = [[3], Int[], [1]], contacts_from = :died)
        m_safe = ModelSpec(RoutedNetwork([household, safe]);
            progression = m.progression, interventions = m.interventions)
        isolated_first = 0
        for s in 1:60
            st = simulate(m_safe; n_initial = 1, rng = StableRNG(s))
            case = st.individuals[1]
            get(case.state, :index, false) || continue
            isolation_time(case) < case.state[:died_time] || continue
            isolated_first += 1
            @test !is_traced(st.individuals[3])
        end
        @test isolated_first > 0
    end

    @testset "RoutedNetwork: naming contacts draws only when a route is uncertain" begin
        ind = Individual(id = 1)
        ind.infection_time = 0.0
        named(ws, rng) = EpiNetwork._route_contacts(ws, (), ind, 1, rng)
        route(name, adj, p) = RouteWindow(name; until = (:recovered,),
            kernel = Exponential(1.0), reach = adj, traceable = p)
        # node 2 is on both routes, node 3 only on the first, node 4 only on the second
        a = [[2, 3], Int[], Int[], Int[]]
        b = [[2, 4], Int[], Int[], Int[]]

        # routes at 1 and 0 decide without using the random number generator
        rng, untouched = StableRNG(7), StableRNG(7)
        @test first.(named([route(:a, a, 1.0), route(:b, b, 0.0)], rng)) == [2, 3]
        @test rand(rng) == rand(untouched)
        rng, untouched = StableRNG(7), StableRNG(7)
        @test first.(named([route(:a, a, 1.0), route(:b, b, 1.0)], rng)) == [2, 3, 4]
        @test rand(rng) == rand(untouched)
        rng, untouched = StableRNG(7), StableRNG(7)
        @test isempty(named([route(:a, a, false), route(:b, b, false)], rng))
        @test rand(rng) == rand(untouched)

        # node 2 is certain through the first route, so only node 4 needs a draw
        rng, untouched = StableRNG(7), StableRNG(7)
        named([route(:a, a, 1.0), route(:b, b, 0.5)], rng)
        rand(untouched)
        @test rand(rng) == rand(untouched)
    end

    @testset "RoutedNetwork: a contact is named with the highest route probability" begin
        # Node 2 is a standing contact on a route naming it 20% of the time, and
        # a funeral contact from day 5 on a route naming it 90% of the time. One
        # draw decides both, so it is named 90% of the time: from infection
        # whenever the first route names it, and otherwise from the funeral.
        ind = Individual(id = 1)
        ind.infection_time = 0.0
        ind.state[:died_time] = 5.0
        standing = RouteWindow(:household; kernel = Exponential(1.0),
            reach = [[2], [1]], traceable = 0.2)
        funeral = RouteWindow(:funeral; from = :died, kernel = Exponential(1.0),
            reach = [[2], [1]], contacts_from = :died, traceable = 0.9)
        rng = StableRNG(11)
        reps = 20_000
        draws = [EpiNetwork._route_contacts([standing, funeral], (), ind, 1, rng)
                 for _ in 1:reps]
        @test all(d -> length(d) <= 1, draws)
        named = count(!isempty, draws) / reps
        from_infection = count(d -> !isempty(d) && d[1][2] == -Inf, draws) / reps
        from_funeral = count(d -> !isempty(d) && d[1][2] == 5.0, draws) / reps
        @test isapprox(named, 0.9; atol = 0.015)
        @test isapprox(from_infection, 0.2; atol = 0.015)
        @test isapprox(from_funeral, 0.7; atol = 0.015)
    end

    @testset "RoutedNetwork: an untraceable route's contacts are never traced" begin
        # Households of four, and a community route linking each node to one node
        # in each neighbouring household. With the community route untraceable,
        # every traced contact must have been traced from a household member.
        nh, hs = 30, 4
        n = nh * hs
        hh_of(i) = (i - 1) ÷ hs
        hh = [[j for j in (hh_of(i) * hs + 1):(hh_of(i) * hs + hs) if j != i] for i in 1:n]
        comm = [[mod1(i + hs, n), mod1(i - hs, n)] for i in 1:n]
        REM = EpiBranch.INTERVENTION_REMOVAL
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        ivs = [
            Isolation(onset_to_isolation_delay = Exponential(1.0),
                test_sensitivity = 1.0),
            ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5))]
        build(p) = ModelSpec(
            RoutedNetwork([
                RouteWindow(:household; until = (:recovered,),
                    kernel = Weibull(1.5, 4.0), reach = hh),
                RouteWindow(:community; until = (:recovered, REM),
                    kernel = Exponential(3.0), reach = comm, traceable = p)]);
            progression = _sir(10.0), interventions = ivs, attributes = clinical)
        pairs(p) = [(ind.id, ind.state[:traced_by])
                    for s in 1:30
                    for ind in simulate(build(p); n_initial = 2,
                            rng = StableRNG(s)).individuals if is_traced(ind)]

        untraceable = pairs(0.0)
        @test !isempty(untraceable)
        @test all(p -> hh_of(p[1]) == hh_of(p[2]), untraceable)
        # with the community route traceable, tracing also crosses households
        @test any(p -> hh_of(p[1]) != hh_of(p[2]), pairs(1.0))
    end

    @testset "RoutedNetwork: a vaccine protects on every route" begin
        # Households of four, and a community route linking each node to one node
        # in each neighbouring household. Isolation cuts only the community route,
        # and tracing doses contacts without quarantining them, so a dosed person
        # is still exposed afterwards, at home as well as in the community. A
        # fully effective dose must then block every one of those exposures.
        nh, hs = 60, 4
        n = nh * hs
        hh_of(i) = (i - 1) ÷ hs
        hh = [[j for j in (hh_of(i) * hs + 1):(hh_of(i) * hs + hs) if j != i] for i in 1:n]
        comm = [[mod1(i + hs, n), mod1(i - hs, n)] for i in 1:n]
        REM = EpiBranch.INTERVENTION_REMOVAL
        clinical = clinical_presentation(incubation_period = LogNormal(0.5, 0.3),
            prob_asymptomatic = 0.0)
        build(ivs) = ModelSpec(
            RoutedNetwork([
                RouteWindow(:household; until = (:recovered,),
                    kernel = Exponential(2.0), reach = hh),
                RouteWindow(:community; until = (:recovered, REM),
                    kernel = Exponential(1.5), reach = comm)]);
            progression = _sir(8.0), interventions = ivs, attributes = clinical)
        tracing = [
            Isolation(onset_to_isolation_delay = Exponential(1.0), test_sensitivity = 1.0),
            ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5),
                quarantine_on_trace = false)]
        immune_at_infection(ind) = is_vaccinated(ind) && ind.parent_id != 0 &&
                                   ind.state[:immunity_time] <= ind.infection_time
        runs(ivs) = [simulate(build(ivs); n_initial = 3, rng = StableRNG(s))
                     for s in 1:10]

        # Dosed with no protection, dosed people are infected after their dose on
        # both routes, so the check below has something to catch.
        placebo_runs = runs([tracing; RingVaccination(efficacy = 0.0)])
        placebo = [ind for st in placebo_runs
                   for ind in st.individuals
                   if immune_at_infection(ind)]
        @test any(ind -> hh_of(ind.id) == hh_of(ind.parent_id), placebo)
        @test any(ind -> hh_of(ind.id) != hh_of(ind.parent_id), placebo)

        # A fully effective dose blocks every one of them, on both routes. The
        # race resolves each proposal when it is popped, so a dose that a trace
        # gave after the infector settled is in force by then.
        full = runs([tracing; RingVaccination(efficacy = 1.0)])
        @test any(st -> any(is_vaccinated, st.individuals), full)
        @test !any(st -> any(immune_at_infection, st.individuals), full)

        # The onward effect applies on every route too: a fully effective one
        # stops a dosed case infecting anyone once its immunity is in place, its
        # household included.
        function infected_by_immune(st)
            filter(st.individuals) do ind
                (is_infected(ind) && ind.parent_id != 0) || return false
                parent = st.individuals[ind.parent_id]
                is_vaccinated(parent) && parent.state[:immunity_time] <= ind.infection_time
            end
        end
        placebo_onward = reduce(vcat, map(infected_by_immune, placebo_runs))
        @test any(ind -> hh_of(ind.id) == hh_of(ind.parent_id), placebo_onward)
        onward = runs([tracing; RingVaccination(efficacy = 0.0, onward_efficacy = 1.0)])
        @test all(st -> isempty(infected_by_immune(st)), onward)
    end

    @testset "RoutedNetwork: route and tracing probabilities multiply" begin
        # A seed on a complete graph with contact too slow to transmit: none of
        # its neighbours is infected before tracing reaches them, so the fraction
        # traced is the route's naming probability times the tracing probability.
        n = 60
        adj = [[j for j in 1:n if j != i] for i in 1:n]
        clinical = clinical_presentation(incubation_period = LogNormal(0.0, 0.3),
            prob_asymptomatic = 0.0)
        ivs = [
            Isolation(onset_to_isolation_delay = Exponential(0.5),
                test_sensitivity = 1.0),
            ContactTracing(probability = 0.5,
                isolation_to_trace_delay = Exponential(0.5))]
        m = ModelSpec(
            RoutedNetwork([RouteWindow(:all; until = (:recovered,),
                kernel = Exponential(1e9), reach = adj, traceable = 0.4)]);
            progression = _sir(5.0), interventions = ivs, attributes = clinical)
        seeds = 200
        traced = sum(count(is_traced,
                         simulate(m; n_initial = 1,
                             rng = StableRNG(s)).individuals)
        for s in 1:seeds)
        # 11,800 contacts at probability 0.2; the tolerance is about five
        # standard deviations
        @test isapprox(traced / (seeds * (n - 1)), 0.2; atol = 0.02)
    end

    @testset "contact tracing" begin
        # A node's contacts are its graph neighbours, so tracing reaches them
        # and quarantining closes their own infectious window in turn.
        adj = ring_adjacency(120)
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0),
            test_sensitivity = 1.0)
        ct = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5))

        build(ivs) = ModelSpec(NetworkProcess(adj, Exponential(2.0));
            progression = _sir(12.0), interventions = ivs, attributes = clinical)
        meansize(ivs) = sum(simulate(build(ivs); n_initial = 1,
                                rng = StableRNG(s)).cumulative_cases for s in 1:60) / 60

        # Tracing is honoured on the continuous-time path, so no warning and a
        # real reduction on top of isolation alone.
        @test meansize([iso, ct]) < meansize([iso])

        # Contacts are actually marked, and a quarantined contact carries a
        # finite isolation time for the window to close on.
        st = simulate(build([iso, ct]); n_initial = 1, rng = StableRNG(3))
        traced = filter(is_traced, st.individuals)
        @test !isempty(traced)
        @test all(is_quarantined, traced)
        @test all(t -> isfinite(isolation_time(t)), traced)

        # Tracing must never *delay* isolation. A quarantine is written on a
        # contact before that contact resolves its own isolation, so unless the
        # self-reporting pathway is still allowed to win, a late trace would
        # replace an earlier self-report and make the outbreak bigger. With a
        # trace delay long enough that tracing can never help, the outbreak must
        # be no worse than isolation alone.
        late = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(500.0))
        @test meansize([iso, late]) <= meansize([iso]) * 1.05

        # Ring vaccination doses along the same trace, so the graph honours it
        # and nothing is reported as unhonoured. A rollout that doses newly
        # created contacts still is: the race creates none.
        @test EpiBranch._sellke_honours(
            build([iso]).process, RingVaccination(efficacy = 0.8))
        # Its eligibility window and its post-exposure abort are timed from a
        # contact's exposure, which a node the race has not settled does not
        # have, so a ring that uses either is reported and doses nobody.
        model = build([iso]).process
        @test !EpiBranch._sellke_honours(
            model, RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.8))
        @test !EpiBranch._sellke_honours(
            model, RingVaccination(efficacy = 0.8, eligibility_window = 21.0))
        pep = @test_logs (:warn, r"RingVaccination") match_mode=:any simulate(
            build([iso, ct,
                RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.8)]);
            n_initial = 1, rng = StableRNG(4))
        @test !any(is_vaccinated, pep.individuals)
        @test_logs (:warn, r"MassVaccination") match_mode=:any simulate(
            build([iso, ct,
                MassVaccination(efficacy = 0.8, eligibility_time = 0.0)]);
            n_initial = 1, rng = StableRNG(4))

        # The package's own interventions that the graph honours warn about
        # nothing, although tracing and ring vaccination implement the
        # generation engine's hooks too.
        honoured = [iso, ct, RingVaccination(efficacy = 0.8),
            Scheduled(RingVaccination(efficacy = 0.5, dose_label = :late);
                start_time = 2.0)]
        @test all(iv -> EpiBranch._sellke_honours(model, iv), honoured)
        @test_logs min_level=Base.CoreLogging.Warn simulate(build(honoured);
            n_initial = 1, rng = StableRNG(4))
    end
end
