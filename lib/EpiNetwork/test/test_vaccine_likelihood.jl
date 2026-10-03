# An undirected random graph on `n` nodes with `m` distinct edges, exposing a
# susceptible to more than one possible infector — the setting the pairwise
# likelihood's susceptibility effect is for.
function _vax_random_graph(n, m, rng)
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

@testset "Vaccinated networks: simulate → loglikelihood round trip" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
    progression = [Transition(:recovered; delay = 15.0, terminal = true)]
    adj = _vax_random_graph(300, 900, StableRNG(41))

    @testset "the structured likelihood matches an explicit vaccine effect" begin
        rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        process = NetworkProcess(adj, Exponential(3.0))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; n_initial = 3, rng = StableRNG(42))
        data = network_infections(state, m)

        recorded = coalesce.(data.host_times.immunity_time, Inf)
        @test recorded == EpiBranch.immunity_time.(state.individuals)
        @test any(isfinite, recorded)

        explicit = pairwise_surv_loglik(
            m.process.edge_kernel, data;
            external_hazard = m.process.external_hazard,
            susceptibility = EpiBranch.vaccine_effect(rv)
        )
        @test loglikelihood(data, m) ≈ explicit

        unvaccinated = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct]
        )
        @test loglikelihood(data, m) != loglikelihood(data, unvaccinated)
    end

    @testset "AllOrNothingMode round-trips through the structured likelihood too" begin
        rv = RingVaccination(efficacy = 0.4, mode = AllOrNothingMode())
        process = NetworkProcess(adj, Exponential(3.0))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; n_initial = 3, rng = StableRNG(43))
        data = network_infections(state, m)
        explicit = pairwise_surv_loglik(
            m.process.edge_kernel, data;
            external_hazard = m.process.external_hazard,
            susceptibility = EpiBranch.vaccine_effect(rv)
        )
        @test loglikelihood(data, m) ≈ explicit
        @test isfinite(loglikelihood(data, m))
    end

    @testset "a vaccination with an onward or post-exposure effect is rejected" begin
        rv = RingVaccination(efficacy = 0.5, onward_efficacy = 0.3)
        process = NetworkProcess(adj, Exponential(3.0))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; n_initial = 3, rng = StableRNG(44))
        data = network_infections(state, m)
        @test_throws ArgumentError loglikelihood(data, m)
    end
end
