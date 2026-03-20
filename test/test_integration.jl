using DataFrames
using Dates

@testset "Integration" begin
    clinical = clinical_presentation(incubation_period=LogNormal(1.5, 0.5))

    @testset "simulate_batch returns correct count" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        results = simulate_batch(model, 50; rng=rng)
        @test length(results) == 50
    end

    @testset "containment_probability" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.8), Exponential(5.0))
        results = simulate_batch(model, 200; rng=rng)
        cp = containment_probability(results)
        @test cp > 0.5
    end

    @testset "weekly_incidence" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; sim_opts=SimOpts(max_cases=100), rng=rng)
        df = weekly_incidence(state)
        @test df isa DataFrame
        @test sum(df.cases) == state.cumulative_cases
    end

    @testset "simulate_conditioned" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.2), Exponential(5.0))
        state = simulate_conditioned(model, 10:50; rng=rng)
        @test state.cumulative_cases in 10:50
    end

    @testset "simulate_conditioned throws on impossible range" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.1), Exponential(5.0))
        @test_throws ErrorException simulate_conditioned(
            model, 1000:2000; max_attempts=100, rng=rng)
    end

    @testset "ringbp-style scenario" begin
        rng = StableRNG(314)
        model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        iso = Isolation(delay=LogNormal(1.0, 0.5))
        ct = ContactTracing(probability=0.5, delay=Exponential(2.0))

        results = simulate_batch(model, 500;
            interventions=[iso, ct], attributes=clinical,
            sim_opts=SimOpts(max_cases=5000, max_generations=50),
            rng=rng)

        @test containment_probability(results) >= 0.5
    end

    @testset "Full pipeline: simulate → linelist → chain_statistics" begin
        rng = StableRNG(42)
        model = BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5))

        state = simulate(model;
            attributes=clinical,
            sim_opts=SimOpts(max_cases=100, n_initial=3),
            rng=rng)

        ll = linelist(state; reference_date=Date(2024, 1, 1), rng=StableRNG(99))
        @test nrow(ll) == state.cumulative_cases

        ct = contacts(state; reference_date=Date(2024, 1, 1))
        @test nrow(ct) > 0

        cs = chain_statistics(state)
        @test nrow(cs) == 3
        @test sum(cs.size) == state.cumulative_cases
    end

    @testset "Leaky isolation allows more transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        rng1 = StableRNG(42)
        iso_perfect = Isolation(delay=Exponential(1.0), residual_transmission=0.0)
        results_perfect = simulate_batch(model, 200;
            interventions=[iso_perfect], attributes=clinical,
            sim_opts=SimOpts(max_cases=200), rng=rng1)

        rng2 = StableRNG(42)
        iso_leaky = Isolation(delay=Exponential(1.0), residual_transmission=0.5)
        results_leaky = simulate_batch(model, 200;
            interventions=[iso_leaky], attributes=clinical,
            sim_opts=SimOpts(max_cases=200), rng=rng2)

        @test containment_probability(results_perfect) >= containment_probability(results_leaky)
    end

    @testset "ringbp_generation_time convenience" begin
        gt_fn = ringbp_generation_time(presymptomatic_fraction=0.3)
        @test gt_fn isa Function

        rng = StableRNG(42)
        model = BranchingProcess(NegBin(2.5, 0.16), gt_fn)
        state = simulate(model;
            attributes=clinical, sim_opts=SimOpts(max_cases=50), rng=rng)
        @test state.cumulative_cases > 0
    end

    @testset "generation_R output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate_conditioned(model, 20:500;
            sim_opts=SimOpts(max_cases=500), rng=rng)

        df = generation_R(state)
        @test df isa DataFrame
        @test nrow(df) > 0
        @test df.R_eff[1] > 0
    end

    @testset "containment_probability with max_cases" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        results = simulate_batch(model, 100;
            sim_opts=SimOpts(max_cases=50), rng=rng)

        cp_naive = containment_probability(results)
        cp_aware = containment_probability(results; max_cases=50)
        @test cp_aware <= cp_naive
    end

    @testset "max_time terminates simulation" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=10_000, max_time=30.0), rng=rng)

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
        results = simulate_batch(model, 5000;
            sim_opts=SimOpts(max_cases=10_000, max_generations=200), rng=rng)
        q_simulated = containment_probability(results)

        @test abs(q_analytical - q_simulated) < 0.05
    end
end
