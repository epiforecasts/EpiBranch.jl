# Tests for the NetworkProcess pairwise survival likelihood: the infection layer
# read out of a simulation, the exact simulate → loglikelihood round trip, and
# differentiability in the kernel parameters.

using ForwardDiff
using DifferentiationInterface
import Mooncake
using ADTypes: AutoMooncake

# An undirected random graph on `n` nodes with `m` distinct edges.
function _random_graph(n, m, rng)
    adj = [Int[] for _ in 1:n]
    edges = 0
    while edges < m
        a, b = rand(rng, 1:n), rand(rng, 1:n)
        (a == b || b in adj[a]) && continue
        push!(adj[a], b)
        push!(adj[b], a)
        edges += 1
    end
    return adj
end

# A latent period followed by a fixed infectious period, so the window the
# kernel times contacts from opens after infection.
function _seir(ip)
    [Transition(:infectious; from = :infection, delay = LogNormal(0.3, 0.3)),
        Transition(:recovered; from = :infectious, delay = ip, terminal = true)]
end

# Newton's method on a scalar log-likelihood, returning the maximiser and its
# standard error from the observed information.
function _newton_mle(f, x0; steps = 20)
    x = x0
    d2(z) = ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), z)
    for _ in 1:steps
        x -= ForwardDiff.derivative(f, x) / d2(x)
    end
    return x, 1 / sqrt(-d2(x))
end

@testset "NetworkProcess likelihood" begin
    @testset "network_infections reads the infection layer" begin
        adj = _random_graph(300, 900, StableRNG(1))
        m = ModelSpec(NetworkProcess(adj, Exponential(3.0)); progression = _seir(4.0))
        state = simulate(m; n_initial = 3, rng = StableRNG(2))
        data = network_infections(state, m)

        @test data isa NetworkInfections
        @test data isa InfectionLayer
        @test length(data) == 300
        @test data.contacts == adj
        @test count(data.is_index) == 3
        infected = .!isnan.(data.infection_time)
        @test count(infected) == count(is_infected, state.individuals)
        @test all(data.infectious_time[infected] .> data.infection_time[infected])
        @test all(isfinite, data.removal_time[infected])
        @test all(isinf, data.removal_time[.!infected])

        # a bare process reads its window from :infection
        bare = NetworkProcess(adj, Exponential(3.0))
        sb = simulate(ModelSpec(bare; progression = _seir(4.0)); n_initial = 3,
            rng = StableRNG(2))
        @test isequal(network_infections(sb, bare).infectious_time,
            network_infections(sb, bare).infection_time)

        @test_throws ArgumentError NetworkInfections(adj, [0.0], [0.0], [1.0], [true])
    end

    @testset "layout and non-layout forms agree" begin
        adj = _random_graph(400, 1200, StableRNG(3))
        m = ModelSpec(NetworkProcess(adj, Weibull(1.5, 4.0)); progression = _seir(4.0))
        data = network_infections(simulate(m; n_initial = 5, rng = StableRNG(4)), m)
        layout = compile_contact_pairs(data)
        @test layout isa ContactPairsLayout
        @test length(layout) > 0

        for s in 2.0:1.0:6.0
            @test pairwise_surv_loglik(Weibull(1.5, s), data, layout) ==
                  pairwise_surv_loglik(Weibull(1.5, s), data)
        end
        @test loglikelihood(data, m) == pairwise_surv_loglik(Weibull(1.5, 4.0), data)
        @test loglikelihood(data, m.process) == loglikelihood(data, m)
    end

    @testset "possible infectors are in-neighbours on a directed graph" begin
        # 1 → 2 and 1 → 3, with 2 infected by 1: node 1 has no in-neighbours
        out = [[2, 3], Int[], Int[]]
        rev = [Int[], [1], [1]]
        inf = [0.0, 1.5, NaN]
        d_out = NetworkInfections(out, inf, inf, [4.0, 5.5, Inf], [true, false, false])
        d_rev = NetworkInfections(rev, inf, inf, [4.0, 5.5, Inf], [true, false, false])
        k = Exponential(2.0)
        # 2 ← 1 with an event at 1.5 and 3 ← 1 escaping over 1's window [0, 4]
        rows = PairwiseSurvivalData([2, 3], [0.0, 0.0], [1.5, 4.0], [true, false])
        @test pairwise_surv_loglik(k, d_out) ≈ pairwise_surv_loglik(k, rows)
        # reversed, nobody can have infected 2 or 3, and 1 is conditioned on
        @test length(compile_contact_pairs(d_rev)) == 0
    end

    @testset "per-edge and covariate kernels" begin
        adj = _random_graph(300, 900, StableRNG(5))
        shared = ModelSpec(NetworkProcess(adj, Exponential(3.0)); progression = _seir(4.0))
        data = network_infections(simulate(shared; n_initial = 3, rng = StableRNG(6)),
            shared)

        # a per-edge vector of identical kernels is the shared kernel
        same = [[Exponential(3.0) for _ in nbrs] for nbrs in adj]
        @test loglikelihood(data, NetworkProcess(adj, same)) ≈
              loglikelihood(data, shared)

        # heterogeneous per-edge kernels resolve along the edge each row travels,
        # as the equivalent (infector, susceptible) callable does
        scale(i, j) = 2.0 + mod(i + 2j, 5)
        per_edge = [[Exponential(scale(i, j)) for j in adj[i]] for i in eachindex(adj)]
        pe = ModelSpec(NetworkProcess(adj, per_edge); progression = _seir(4.0))
        dpe = network_infections(simulate(pe; n_initial = 3, rng = StableRNG(7)), pe)
        @test isfinite(loglikelihood(dpe, pe))
        @test loglikelihood(dpe, pe) ≈
              pairwise_surv_loglik((i, j) -> Exponential(scale(i, j)), dpe)
    end

    @testset "simulate → loglikelihood round trip recovers the kernel scale" begin
        # the Sellke race is the likelihood's generative model, so maximising it
        # over a simulated infection layer recovers the kernel within its
        # standard error from the observed information
        true_scale = 6.0
        adj = _random_graph(2000, 6000, StableRNG(8))
        m = ModelSpec(NetworkProcess(adj, Exponential(true_scale));
            progression = _seir(4.0))
        data = network_infections(simulate(m; n_initial = 10, rng = StableRNG(9)), m)
        @test count(!isnan, data.infection_time) > 500   # a real outbreak to fit

        layout = compile_contact_pairs(data)
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ)), data, layout)
        θhat, se = _newton_mle(f, log(3.0))
        @test abs(ForwardDiff.derivative(f, θhat)) < 1e-6
        @test se < 0.1
        @test abs(θhat - log(true_scale)) < 3 * se
    end

    @testset "isolation ends the infectious window in the infection layer" begin
        # the race closes a case's window when it is isolated, so the data must
        # too, or the likelihood sees cases infectious after isolation and
        # overestimates the kernel scale
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        adj = _random_graph(3000, 12000, StableRNG(14))
        m = ModelSpec(NetworkProcess(adj, Exponential(4.0));
            progression = [Transition(:recovered; from = :infection, delay = 8.0,
                terminal = true)],
            interventions = [iso], attributes = clinical)
        state = simulate(m; n_initial = 20, rng = StableRNG(15))
        data = network_infections(state, m)

        infected = findall(!isnan, data.infection_time)
        expected = [min(data.infection_time[i] + 8.0,
                        EpiBranch.isolation_time(state.individuals[i])) for i in infected]
        @test data.removal_time[infected] == expected
        @test count(data.removal_time[infected] .< data.infection_time[infected] .+ 8.0) >
              length(infected) / 2

        layout = compile_contact_pairs(data)
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ)), data, layout)
        θhat, se = _newton_mle(f, log(4.0))
        @test abs(θhat - log(4.0)) < 3 * se
    end

    @testset "simulated infection layers never have zero density" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        for seed in 1:4, (ext, Tobs) in ((0.0, Inf), (0.03, 10.0)),
            interventions in ([], [iso])
            adj = _random_graph(400, 900, StableRNG(seed))
            m = ModelSpec(
                NetworkProcess(adj, Weibull(1.5, 4.0); external_hazard = ext,
                    obs_end = Tobs);
                progression = _seir(5.0), interventions, attributes = clinical)
            d = network_infections(simulate(m; n_initial = 3, rng = StableRNG(seed)), m)
            layout = compile_contact_pairs(d; external = ext > 0)
            @test isfinite(loglikelihood(d, m))
            @test isfinite(pairwise_surv_loglik(Weibull(1.5, 4.0), d, layout;
                external_hazard = ext))
        end
    end

    @testset "an impossible component gives -Inf with a zero gradient" begin
        # Two components. In {1, 2} node 1 is a community case at 0 and infects 2
        # at 1.0, which the parameters do move. In {3, 4} node 4 is infected at
        # 8.0, after obs_end and before its only in-neighbour is infectious, so
        # nothing can explain it. The density is -Inf over a whole neighbourhood
        # of the parameters — possibility is fixed by the times — so the gradient
        # must be exactly zero rather than that of the other component's terms.
        inf = [0.0, 1.0, 10.0, 8.0]
        data = NetworkInfections([[2], [1], [4], [3]], inf, inf,
            [5.0, 6.0, 12.0, 13.0], [true, false, true, false]; obs_end = 5.0)
        layout = compile_contact_pairs(data; external = true)
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data;
            external_hazard = exp(θ[2]))
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, layout;
            external_hazard = exp(θ[2]))
        θ = [log(3.0), log(0.1)]
        @test f(θ) == -Inf
        @test g(θ) == -Inf
        @test ForwardDiff.gradient(f, θ) == [0.0, 0.0]
        @test ForwardDiff.gradient(g, θ) == [0.0, 0.0]
        @test DifferentiationInterface.gradient(g, AutoMooncake(), θ) == [0.0, 0.0]
    end

    @testset "community hazard" begin
        adj = _random_graph(1000, 3000, StableRNG(10))
        m = ModelSpec(
            NetworkProcess(adj, Exponential(6.0);
                external_hazard = 0.005, obs_end = 40.0);
            progression = _seir(4.0))
        data = network_infections(simulate(m; rng = StableRNG(11)), m)
        @test data.obs_end == 40.0
        @test count(data.is_index) >= 1

        layout = compile_contact_pairs(data; external = true)
        @test layout.external
        @test loglikelihood(data, m) ==
              pairwise_surv_loglik(Exponential(6.0), data, layout;
            external_hazard = 0.005)
        @test_throws ArgumentError pairwise_surv_loglik(Exponential(6.0), data, layout)

        # the rate and the kernel scale are both recovered jointly
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, layout;
            external_hazard = exp(θ[2]))
        θ = [log(6.0), log(0.005)]
        grad = ForwardDiff.gradient(g, θ)
        H = ForwardDiff.hessian(g, θ)
        θhat = θ - H \ grad
        for _ in 1:10
            θhat -= ForwardDiff.hessian(g, θhat) \ ForwardDiff.gradient(g, θhat)
        end
        Σ = inv(-ForwardDiff.hessian(g, θhat))
        se = sqrt.([Σ[1, 1], Σ[2, 2]])
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "community introductions stop at obs_end and edge spread goes on" begin
        # most infections here come after obs_end, along edges; the likelihood
        # must give them no community hazard and keep uninfected nodes exposed
        # over their neighbours' whole windows, as the simulation does
        adj = _random_graph(2000, 4000, StableRNG(16))
        m = ModelSpec(
            NetworkProcess(adj, Exponential(8.0);
                external_hazard = 0.01, obs_end = 10.0);
            progression = _seir(4.0))
        data = network_infections(simulate(m; rng = StableRNG(17)), m)
        inf = filter(!isnan, data.infection_time)
        @test count(>(10.0), inf) > length(inf) / 2

        layout = compile_contact_pairs(data; external = true)
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, layout;
            external_hazard = exp(θ[2]))
        θ = [log(8.0), log(0.01)]
        θhat = copy(θ)
        for _ in 1:20
            θhat -= ForwardDiff.hessian(g, θhat) \ ForwardDiff.gradient(g, θhat)
        end
        Σ = inv(-ForwardDiff.hessian(g, θhat))
        se = sqrt.([Σ[1, 1], Σ[2, 2]])
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "an outbreak still going at the end of follow-up" begin
        # the data as seen at day 15: later infections unseen and windows still
        # open recorded as `Inf`
        adj = _random_graph(2000, 4000, StableRNG(16))
        m = ModelSpec(
            NetworkProcess(adj, Exponential(8.0);
                external_hazard = 0.01, obs_end = 10.0);
            progression = _seir(4.0))
        state = simulate(m; rng = StableRNG(17))
        tf = 15.0
        full = network_infections(state, m)
        late = .!(full.infection_time .<= tf)
        nan_late(x) = [l ? NaN : v for (v, l) in zip(x, late)]
        ongoing = NetworkInfections(adj, nan_late(full.infection_time),
            nan_late(full.infectious_time),
            [l ? NaN : (r > tf ? Inf : r) for (r, l) in zip(full.removal_time, late)],
            full.is_index .& .!late; obs_end = 10.0, followup_end = tf)
        @test any(isinf, ongoing.removal_time)
        @test count(!isnan, ongoing.infection_time) < count(!isnan, full.infection_time)

        read = network_infections(state, m; followup_end = tf)
        @test read.followup_end == tf
        capped = NetworkInfections(adj, ongoing.infection_time, ongoing.infectious_time,
            min.(ongoing.removal_time, tf), ongoing.is_index; obs_end = 10.0)
        k = Exponential(8.0)
        v = pairwise_surv_loglik(k, ongoing; external_hazard = 0.01)
        @test isfinite(v)
        @test v ≈ pairwise_surv_loglik(k, read; external_hazard = 0.01)
        @test v ≈ pairwise_surv_loglik(k, capped; external_hazard = 0.01)
        @test loglikelihood(ongoing, m) ≈ v

        layout = compile_contact_pairs(ongoing; external = true)
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), ongoing, layout;
            external_hazard = exp(θ[2]))
        θ = [log(8.0), log(0.01)]
        θhat = copy(θ)
        for _ in 1:20
            θhat -= ForwardDiff.hessian(g, θhat) \ ForwardDiff.gradient(g, θhat)
        end
        Σ = inv(-ForwardDiff.hessian(g, θhat))
        se = sqrt.([Σ[1, 1], Σ[2, 2]])
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "differentiable in the kernel parameters (ForwardDiff, Mooncake)" begin
        adj = _random_graph(300, 900, StableRNG(12))
        m = ModelSpec(NetworkProcess(adj, Exponential(3.0)); progression = _seir(4.0))
        data = network_infections(simulate(m; n_initial = 3, rng = StableRNG(13)), m)
        layout = compile_contact_pairs(data)
        x = [log(3.0)]

        f_shared(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, layout)
        h = 1e-6
        fd = (f_shared(x .+ h) - f_shared(x .- h)) / 2h
        @test ForwardDiff.gradient(f_shared, x)[1] ≈ fd rtol = 1e-5

        f_edge(θ) = pairwise_surv_loglik(
            [[Exponential(exp(θ[1]) * (1 + 0.01 * j)) for j in nbrs] for nbrs in adj],
            data, layout)
        @test ForwardDiff.gradient(f_edge, x)[1] ≈
              (f_edge(x .+ h) - f_edge(x .- h)) / 2h rtol = 1e-5

        backend = AutoMooncake(; config = nothing)
        @test DifferentiationInterface.gradient(f_shared, backend, x) ≈
              ForwardDiff.gradient(f_shared, x)
    end
end
