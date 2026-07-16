using DataFrames
using Dates

@testset "Integration" begin
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

    @testset "simulate(model, n) returns correct count" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        results = simulate(model, 50; rng = rng)
        @test length(results) == 50
    end

    @testset "containment_probability" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        results = simulate(model, 200; rng = rng)
        cp = containment_probability(results)
        @test cp > 0.5
    end

    @testset "weekly_incidence" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; max_cases = 100, rng = rng)
        df = weekly_incidence(state)
        @test df isa DataFrame
        @test sum(df.cases) == state.cumulative_cases
    end

    @testset "weekly_incidence by=:reporting excludes unreported (Inf) cases" begin
        # A Reporting transition with probability < 1 leaves unreported cases
        # with :reporting_time = Inf. Binning by :reporting must skip those
        # rather than throw InexactError on Day(floor(Int, Inf)).
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        rep = Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5)
        state = simulate(ModelSpec(model; progression = [rep], attributes = clinical);
            max_cases = 200, rng = rng)
        df = weekly_incidence(state; by = :reporting)
        @test df isa DataFrame
        n_reported = count(ind -> get(ind.state, :reported, false), state.individuals)
        @test sum(df.cases) == n_reported
        @test 0 < n_reported < state.cumulative_cases    # some, not all, reported
    end

    @testset "simulate with condition" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.2), Exponential(5.0))
        state = simulate(model; condition = 10:50, rng = rng)
        @test state.cumulative_cases in 10:50
    end

    @testset "simulate with condition throws on impossible range" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.1), Exponential(5.0))
        @test_throws ErrorException simulate(
            model; condition = 1000:2000, max_attempts = 100, rng = rng)
    end

    @testset "ringbp-style scenario" begin
        rng = StableRNG(314)
        iso = Isolation(onset_to_isolation_delay = LogNormal(1.0, 0.5))
        ct = ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(2.0))

        results = simulate(
            ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
                interventions = [iso, ct], attributes = clinical),
            500; max_cases = 5000, max_generations = 50, rng = rng)

        @test containment_probability(results) >= 0.5
    end

    @testset "Full pipeline: simulate → linelist → chain_statistics" begin
        rng = StableRNG(42)
        state = simulate(
            ModelSpec(BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5)); attributes = clinical);
            max_cases = 100, n_initial = 3, rng = rng)

        ll = linelist(state; reference_date = Date(2024, 1, 1))
        @test nrow(ll) == state.cumulative_cases

        ct = contacts(state; reference_date = Date(2024, 1, 1))
        @test nrow(ct) > 0

        cs = chain_statistics(state)
        @test nrow(cs) == 3
        @test sum(cs.size) == state.cumulative_cases
    end

    @testset "Leaky isolation allows more transmission" begin
        rng1 = StableRNG(42)
        iso_perfect = Isolation(onset_to_isolation_delay = Exponential(1.0), post_isolation_transmission = 0.0)
        results_perfect = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso_perfect], attributes = clinical),
            200; max_cases = 200, rng = rng1)

        rng2 = StableRNG(42)
        iso_leaky = Isolation(onset_to_isolation_delay = Exponential(1.0), post_isolation_transmission = 0.5)
        results_leaky = simulate(
            ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [iso_leaky], attributes = clinical),
            200; max_cases = 200, rng = rng2)

        @test containment_probability(results_perfect) >=
              containment_probability(results_leaky)
    end

    @testset "incubation_linked_generation_time convenience" begin
        gt_fn = incubation_linked_generation_time(presymptomatic_fraction = 0.3)
        @test gt_fn isa Function

        rng = StableRNG(42)
        state = simulate(
            ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt_fn); attributes = clinical);
            max_cases = 50, rng = rng)
        @test state.cumulative_cases > 0
    end

    @testset "incubation_linked_generation_time realises the presymptomatic fraction" begin
        presymp = 0.3
        gt_fn = incubation_linked_generation_time(presymptomatic_fraction = presymp)
        # Incubation period ξ = 5 (≫ ω, so truncation barely perturbs the split).
        ind = Individual(id = 1, infection_time = 0.0,
            state = Dict{Symbol, Any}(:onset_time => 5.0))
        d = gt_fn(ind)

        # Realised fraction of generation times shorter than ξ ≈ presymptomatic.
        rng = StableRNG(1)
        draws = [rand(rng, d) for _ in 1:20_000]
        @test isapprox(count(<(5.0), draws) / length(draws), presymp; atol = 0.02)

        # logpdf is normalised over [0, ∞): the density integrates to ≈ 1
        # (before the truncation constant it did not).
        xs = range(0.0, 20.0; length = 4001)
        dens = [exp(logpdf(d, x)) for x in xs]
        integral = sum((dens[i] + dens[i + 1]) / 2 * step(xs)
        for i in 1:(length(xs) - 1))
        @test isapprox(integral, 1.0; atol = 1e-3)
    end

    @testset "generation_R output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model;
            condition = 20:500, max_cases = 500, rng = rng)

        df = generation_R(state)
        @test df isa DataFrame
        @test nrow(df) > 0
        @test df.offspring_ratio[1] > 0
    end

    @testset "containment_probability with max_cases" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        results = simulate(model, 100;
            max_cases = 50, rng = rng)

        cp_naive = containment_probability(results)
        cp_aware = containment_probability(results; max_cases = 50)
        @test cp_aware <= cp_naive
    end

    @testset "max_time terminates simulation" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model;
            max_cases = 10_000, max_time = 30.0, rng = rng)

        # Check infected individuals only — non-infected contacts may have
        # later generation times but were never active
        infected = filter(is_infected, state.individuals)
        max_inf_time = maximum(ind.infection_time for ind in infected)
        @test max_inf_time <= 40.0  # allow overshoot from last generation
    end

    @testset "Analytical vs simulation extinction" begin
        R, k = 1.5, 0.5
        q_analytical = extinction_probability(R, k)

        rng = StableRNG(42)
        model = BranchingProcess(NegBin(R, k), Exponential(5.0))
        results = simulate(model, 500;
            max_cases = 5000, max_generations = 200, rng = rng)
        q_simulated = containment_probability(results)

        @test abs(q_analytical - q_simulated) < 0.1
    end
end
