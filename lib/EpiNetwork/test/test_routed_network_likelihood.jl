# Tests for the RoutedNetwork pairwise survival likelihood: the per-route
# infection layer read out of a simulation, the exact simulate → loglikelihood
# round trip with several routes and a lapsing removal gated to one of them,
# a vaccination's change in susceptibility evaluated across every route, and
# the error paths for what it does not yet evaluate.

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

    @testset "a community hazard per node matches a shared one when all equal" begin
        hh, comm = _routed_adjacency(30, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Weibull(1.5, 4.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        shared = ModelSpec(
            RoutedNetwork(windows; external_hazard = 0.01, obs_end = 20.0);
            progression = _sir(8.0)
        )
        per_node = ModelSpec(
            RoutedNetwork(windows; external_hazard = fill(0.01, 120), obs_end = 20.0);
            progression = _sir(8.0)
        )
        a = simulate(shared; rng = StableRNG(4))
        b = simulate(per_node; rng = StableRNG(4))
        @test [i.infection_time for i in a.individuals] ==
            [i.infection_time for i in b.individuals]
        data = routed_network_infections(a, shared)
        @test loglikelihood(data, per_node) ≈ loglikelihood(data, shared)
        # a node without a community hazard has no external term
        silent = [0.0; fill(0.01, 119)]
        @test pairwise_surv_loglik(windows, data; external_hazard = silent) !=
            loglikelihood(data, shared)
        @test_throws DimensionMismatch RoutedNetwork(
            windows; external_hazard = fill(0.01, 119), obs_end = 20.0
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

    @testset "a vaccination's change in susceptibility is evaluated structurally" begin
        hh, comm = _routed_adjacency(20, 4)
        windows = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(3.0), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm),
        ]
        ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
        process = RoutedNetwork(windows)
        clinical = clinical_presentation(incubation_period = Dirac(0.0))

        @testset "the structured likelihood matches an explicit vaccine effect" begin
            rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
            m = ModelSpec(
                process; progression = _sir(8.0), attributes = clinical,
                interventions = [ct, rv]
            )
            state = simulate(m; n_initial = 2, rng = StableRNG(5))
            data = routed_network_infections(state, m)
            @test any(!ismissing, data.host_times.immunity_time)

            explicit = pairwise_surv_loglik(
                windows, data; susceptibility = EpiBranch.vaccine_effect(rv)
            )
            @test loglikelihood(data, m) ≈ explicit

            unvaccinated = ModelSpec(
                process; progression = _sir(8.0), attributes = clinical,
                interventions = [ct]
            )
            @test loglikelihood(data, m) != loglikelihood(data, unvaccinated)
        end

        @testset "AllOrNothingMode round-trips through the structured likelihood too" begin
            rv = RingVaccination(efficacy = 0.4, mode = AllOrNothingMode())
            m = ModelSpec(
                process; progression = _sir(8.0), attributes = clinical,
                interventions = [ct, rv]
            )
            state = simulate(m; n_initial = 2, rng = StableRNG(6))
            data = routed_network_infections(state, m)
            explicit = pairwise_surv_loglik(
                windows, data; susceptibility = EpiBranch.vaccine_effect(rv)
            )
            @test loglikelihood(data, m) ≈ explicit
            @test isfinite(loglikelihood(data, m))
        end

        @testset "a single route matches NetworkProcess with the same vaccination" begin
            # A single-route `RoutedNetworkInfections` has exactly the columns
            # a `NetworkInfections` over the same adjacency does, so the two
            # likelihoods must agree on the same recorded outbreak and vaccine
            # effect, regardless of how each simulator arrived at it.
            adj = ring_adjacency(80)
            k = Exponential(2.0)
            rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
            routed_m = ModelSpec(
                RoutedNetwork([RouteWindow(:all; until = (:recovered,), kernel = k, reach = adj)]);
                progression = _sir(8.0), attributes = clinical, interventions = [ct, rv]
            )
            state = simulate(routed_m; n_initial = 2, rng = StableRNG(7))
            routed_data = routed_network_infections(state, routed_m)

            plain_m = ModelSpec(
                NetworkProcess(adj, k; until = (:recovered,));
                progression = _sir(8.0), attributes = clinical, interventions = [ct, rv]
            )
            plain_data = NetworkInfections(
                adj, routed_data.infection_time, routed_data.infectious_time[1],
                routed_data.removal_time[1], routed_data.is_index;
                obs_end = routed_data.obs_end, followup_end = routed_data.followup_end,
                host_times = routed_data.host_times
            )
            @test loglikelihood(routed_data, routed_m) ≈ loglikelihood(plain_data, plain_m)
        end

        @testset "an onward or post-exposure effect is still refused" begin
            rv = RingVaccination(efficacy = 0.5, onward_efficacy = 0.3)
            m = ModelSpec(
                process; progression = _sir(8.0), attributes = clinical,
                interventions = [ct, rv]
            )
            state = simulate(m; n_initial = 2, rng = StableRNG(8))
            data = routed_network_infections(state, m)
            @test_throws ArgumentError loglikelihood(data, m)
        end

        @testset "fitting efficacy on a simulated routed outbreak recovers it" begin
            true_efficacy = 0.6
            hh2, comm2 = _routed_adjacency(150, 4)
            windows2 = [
                RouteWindow(:household; until = (:recovered,), kernel = Exponential(3.0), reach = hh2),
                RouteWindow(:community; until = (:recovered,), kernel = Exponential(10.0), reach = comm2),
            ]
            rv = RingVaccination(efficacy = true_efficacy, mode = LeakyMode())
            m = ModelSpec(
                RoutedNetwork(windows2); progression = _sir(8.0), attributes = clinical,
                interventions = [ct, rv]
            )
            state = simulate(m; n_initial = 5, rng = StableRNG(9))
            data = routed_network_infections(state, m)
            @test any(!ismissing, data.host_times.immunity_time)

            logit(p) = log(p / (1 - p))
            invlogit(x) = 1 / (1 + exp(-x))
            g(x) = pairwise_surv_loglik(
                windows2, data;
                susceptibility = VaccineEffect(efficacy = invlogit(x), mode = LeakyMode())
            )
            θ = logit(true_efficacy)
            θhat, se = _newton_mle(g, θ)
            @test abs(θhat - θ) < 3 * se
        end
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

    @testset "still differentiable in the route kernels' parameters under a susceptibility effect" begin
        # A vaccinated host's term goes through `_routed_mixture_loglik`, whose
        # `T` must cover the route kernels' own parameter type as well as the
        # mixture's, or its `T(-Inf)`/`T(l)` conversions throw once `θ` is a
        # `Dual` from differentiating the kernels rather than the mixture.
        hh, comm = _routed_adjacency(30, 4)
        windows(θ) = [
            RouteWindow(:household; until = (:recovered,), kernel = Exponential(exp(θ[1])), reach = hh),
            RouteWindow(:community; until = (:recovered,), kernel = Exponential(exp(θ[2])), reach = comm),
        ]
        ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
        rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        clinical = clinical_presentation(incubation_period = Dirac(0.0))
        m = ModelSpec(
            RoutedNetwork(windows([log(3.0), log(10.0)])); progression = _sir(8.0),
            attributes = clinical, interventions = [ct, rv]
        )
        state = simulate(m; n_initial = 3, rng = StableRNG(6))
        data = routed_network_infections(state, m)
        @test any(!ismissing, data.host_times.immunity_time)

        f(θ) = pairwise_surv_loglik(
            windows(θ), data; susceptibility = EpiBranch.vaccine_effect(rv)
        )
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
