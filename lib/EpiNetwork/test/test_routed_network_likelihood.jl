# Tests for the RoutedNetwork pairwise survival likelihood: the per-route
# infection layer read out of a simulation, the exact simulate → loglikelihood
# round trip with several routes and a lapsing removal gated to one of them,
# and the error paths for what it does not yet evaluate.

# Households of `hs` as cliques, and a ring of community contacts (each
# linked to its `k` nearest neighbours either side) over the same `nh * hs`
# people.
function _routed_adjacency(nh, hs, k = 2)
    n = nh * hs
    hh = [Int[] for _ in 1:n]
    for h in 0:(nh - 1), i in (h * hs + 1):(h * hs + hs), j in (h * hs + 1):(h * hs + hs)
        i != j && push!(hh[i], j)
    end
    return hh, ring_adjacency(n, k)
end

@testset "RoutedNetwork likelihood" begin
    @testset "routed_network_infections reads one window per route" begin
        hh, comm = _routed_adjacency(20, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Weibull(1.5, 4.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        m = ModelSpec(RoutedNetwork(windows); progression = _sir(8.0))
        state = simulate(m; n_initial = 2, rng = StableRNG(1))
        data = routed_network_infections(state, m)

        @test data isa RoutedNetworkInfections
        @test data isa InfectionLayer
        @test length(data) == 80
        @test data.route_names == [:household, :community]
        @test data.contacts == [hh, comm]

        infected = findall(!isnan, data.infection_time)
        @test !isempty(infected)
        for (r, w) in enumerate(windows), i in infected
            ind = state.individuals[i]
            @test data.infectious_time[r][i] == window_open(ind, w)
            @test data.removal_time[r][i] == window_close(ind, w, ())
        end
        @test_throws ArgumentError EpiBranch.contact_structure(data)
    end

    @testset "pairwise_surv_loglik, the bare process and ModelSpec agree" begin
        hh, comm = _routed_adjacency(30, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Weibull(1.5, 4.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        process = RoutedNetwork(windows)
        m = ModelSpec(process; progression = _sir(8.0))
        data = routed_network_infections(simulate(m; n_initial = 3, rng = StableRNG(2)), m)

        direct = pairwise_surv_loglik(windows, data)
        @test isfinite(direct)
        @test loglikelihood(data, process) == direct
        @test loglikelihood(data, m) == direct

        @test_throws ArgumentError pairwise_surv_loglik(windows[1:1], data)
        @test_throws ArgumentError pairwise_surv_loglik(reverse(windows), data)
    end

    @testset "the community hazard can differ by person" begin
        hh, comm = _routed_adjacency(60, 4)
        n = length(hh)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Weibull(1.5, 4.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        rates = [j <= n ÷ 2 ? 0.01 : 0.05 for j in 1:n]
        m = ModelSpec(
            RoutedNetwork(windows; external_hazard = rates, obs_end = 20.0);
            progression = _sir(8.0)
        )
        state = simulate(m; rng = StableRNG(4))
        data = routed_network_infections(state, m)
        @test isfinite(loglikelihood(data, m))
        @test loglikelihood(data, m) ==
            pairwise_surv_loglik(windows, data; external_hazard = rates)

        half1 = count(i -> i.id <= n ÷ 2 && get(i.state, :index, false), state.individuals)
        half2 = count(i -> i.id > n ÷ 2 && get(i.state, :index, false), state.individuals)
        @test half2 > half1

        # a vector not matching the population in `data` is rejected rather
        # than indexed out of bounds
        @test_throws ArgumentError pairwise_surv_loglik(
            windows, data; external_hazard = rates[1:(end - 1)]
        )
    end

    @testset "simulate → loglikelihood recovers both routes' kernel scales jointly" begin
        # survival sums the two routes' hazards and the event term is a
        # log-sum-exp across them: fitting both scales at once is the sharpest
        # check that the combination (not just each route alone) is right.
        true_household, true_community = 3.0, 6.0
        hh, comm = _routed_adjacency(250, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(true_household), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(true_community), reach = comm),
        ]
        m = ModelSpec(RoutedNetwork(windows); progression = _sir(8.0))
        state = simulate(m; n_initial = 5, rng = StableRNG(3))
        data = routed_network_infections(state, m)
        @test count(!isnan, data.infection_time) > 300

        g(θ) = pairwise_surv_loglik(
            [
                RouteWindow(:household; until = (:recovered,), kernel = Exponential(exp(θ[1])), reach = hh),
                RouteWindow(:community; until = (:recovered,), kernel = Exponential(exp(θ[2])), reach = comm),
            ],
            data
        )
        θ = [log(true_household), log(true_community)]
        θhat, se = _newton_mle_joint(g, θ)
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "a lapsing isolation gated to the community route leaves the household route alone" begin
        # `Isolation(duration = 7.0)` releases rather than permanently removing,
        # so the window stays open and the stretch is read from `host_times`;
        # since only the community route lists `INTERVENTION_REMOVAL`, the
        # household route's exposure must be unaffected by it.
        true_household, true_community = 3.0, 6.0
        hh, comm = _routed_adjacency(300, 4)
        REM = EpiBranch.INTERVENTION_REMOVAL
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(true_household), reach = hh),
            RouteWindow(
                :community; until = (:recovered, REM),
                kernel = Exponential(true_community), reach = comm
            ),
        ]
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.0, 0.3), prob_asymptomatic = 0.0
        )
        iso = Isolation(
            onset_to_isolation_delay = Exponential(1.5),
            test_sensitivity = 1.0, duration = 7.0
        )
        m = ModelSpec(
            RoutedNetwork(windows); progression = _sir(10.0),
            interventions = [iso], attributes = clinical
        )
        state = simulate(m; n_initial = 5, rng = StableRNG(4))
        data = routed_network_infections(state, m)
        @test haskey(data.host_times, EpiBranch.REMOVAL_STRETCHES_KEY)
        @test any(!ismissing, data.host_times[EpiBranch.REMOVAL_STRETCHES_KEY])
        # a lapsing isolation never permanently closes a window; the stretch it
        # records takes the exposure out instead, so both routes close at the
        # same recovery time, whichever lists `INTERVENTION_REMOVAL`
        infected = findall(!isnan, data.infection_time)
        @test data.removal_time[1][infected] == data.infection_time[infected] .+ 10.0
        @test data.removal_time[2][infected] == data.infection_time[infected] .+ 10.0

        g(θ) = pairwise_surv_loglik(
            [
                RouteWindow(:household; until = (:recovered,), kernel = Exponential(exp(θ[1])), reach = hh),
                RouteWindow(
                    :community; until = (:recovered, REM),
                    kernel = Exponential(exp(θ[2])), reach = comm
                ),
            ],
            data
        )
        θ = [log(true_household), log(true_community)]
        θhat, se = _newton_mle_joint(g, θ)
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "construction validates its arguments" begin
        hh, comm = _routed_adjacency(5, 4)
        n = length(hh)
        ok = [[0.0 for _ in 1:n], [0.0 for _ in 1:n]]
        @test_throws ArgumentError RoutedNetworkInfections(
            [:household], [hh, comm], ok, ok, ok[1], falses(n)
        )
        @test_throws ArgumentError RoutedNetworkInfections(
            [:household, :community], [hh, comm], ok, ok, [0.0], falses(n)
        )
        @test_throws ArgumentError RoutedNetworkInfections(
            [:household, :community], [hh, hh[1:(end - 1)]], ok, ok, ok[1], falses(n)
        )
        out_of_range = [copy(c) for c in hh]
        push!(out_of_range[1], n + 1)
        @test_throws ArgumentError RoutedNetworkInfections(
            [:household, :community], [out_of_range, comm], ok, ok, ok[1], falses(n)
        )
    end

    @testset "a susceptibility effect is refused" begin
        hh, comm = _routed_adjacency(20, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(3.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
        rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        process = RoutedNetwork(windows)
        clinical = clinical_presentation(incubation_period = Dirac(0.0))
        m = ModelSpec(
            process; progression = _sir(8.0), attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; n_initial = 2, rng = StableRNG(5))
        data = routed_network_infections(state, m)
        @test_throws ArgumentError loglikelihood(data, m)
        # the same data fits fine once the effect it cannot represent is dropped
        unvaccinated = ModelSpec(process; progression = _sir(8.0), attributes = clinical)
        @test isfinite(loglikelihood(data, unvaccinated))
    end

    @testset "differentiable in the route kernels' parameters (ForwardDiff)" begin
        hh, comm = _routed_adjacency(30, 4)
        windows(θ) = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(exp(θ[1])), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(exp(θ[2])), reach = comm),
        ]
        m = ModelSpec(RoutedNetwork(windows([log(3.0), log(10.0)])); progression = _sir(8.0))
        data = routed_network_infections(simulate(m; n_initial = 3, rng = StableRNG(6)), m)

        f(θ) = pairwise_surv_loglik(windows(θ), data)
        x = [log(3.0), log(10.0)]
        h = 1.0e-6
        fd = [
            (f(x + h * e) - f(x - h * e)) / 2h
                for e in ([1.0, 0.0], [0.0, 1.0])
        ]
        @test ForwardDiff.gradient(f, x) ≈ fd rtol = 1.0e-5
    end

    @testset "a structurally infeasible host has a zero gradient, not just -Inf" begin
        # host 1 infects host 2 on the only route, but closes before host 2's
        # infection time. Host 2's survival term then accumulates hazard
        # (depending on the kernel's rate) while its event term stays empty,
        # so the likelihood is -Inf at every parameter value. Its gradient
        # must be the zero vector, not whatever host 2's discarded survival
        # term would otherwise leave behind in the shared running total.
        reach = [[2], Int[]]
        data = RoutedNetworkInfections(
            [:community], [reach], [[0.0, 0.0]], [[2.0, Inf]],
            [0.0, 5.0], [true, false]
        )
        f(θ) = pairwise_surv_loglik(
            [RouteWindow(:community; until = (:recovered,), kernel = Exponential(exp(θ[1])), reach = reach)],
            data
        )
        @test f([0.0]) == -Inf
        @test ForwardDiff.gradient(f, [0.0]) == [0.0]
    end
end
