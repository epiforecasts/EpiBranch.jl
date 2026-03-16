using DataFrames
using Dates

@testset "Integration" begin
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
        # R < 1, so containment probability should be high
        @test cp > 0.5
    end

    @testset "weekly_incidence" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; sim_opts=SimOpts(max_cases=100), rng=rng)
        df = weekly_incidence(state)
        @test df isa DataFrame
        @test "week" in names(df)
        @test "cases" in names(df)
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
        # Very low R, asking for 1000+ cases — should fail
        @test_throws ErrorException simulate_conditioned(
            model, 1000:2000; max_attempts=100, rng=rng
        )
    end

    @testset "ringbp-style scenario" begin
        rng = StableRNG(314)
        model = BranchingProcess(
            NegBin(2.5, 0.16),
            LogNormal(1.6, 0.5)
        )
        iso = Isolation(delay=LogNormal(1.0, 0.5))
        ct = ContactTracing(probability=0.5, delay=Exponential(2.0))

        results = simulate_batch(model, 500;
            interventions=[iso, ct],
            sim_opts=SimOpts(
                max_cases=5000,
                max_generations=50,
                incubation_period=LogNormal(1.5, 0.5),
            ),
            rng=rng)

        cp = containment_probability(results)
        @test cp >= 0.5
    end

    @testset "Full pipeline: simulate → linelist → chain_statistics" begin
        rng = StableRNG(42)
        model = BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5))

        state = simulate(model;
            sim_opts=SimOpts(
                max_cases=100,
                n_initial=3,
                incubation_period=LogNormal(1.5, 0.5),
            ),
            rng=rng)

        # Line list
        rng2 = StableRNG(99)
        ll = linelist(state;
            reference_date=Date(2024, 1, 1),
            delay_opts=DelayOpts(
                onset_to_reporting=Exponential(3.0),
                onset_to_outcome=Exponential(14.0),
            ),
            outcome_opts=OutcomeOpts(prob_death=0.1),
            rng=rng2)

        @test nrow(ll) == state.cumulative_cases
        @test all(ll.date_infection .>= Date(2024, 1, 1))

        # Contacts
        ct = contacts(state; reference_date=Date(2024, 1, 1))
        @test nrow(ct) == state.cumulative_cases - 3  # minus index cases

        # Chain statistics
        cs = chain_statistics(state)
        @test nrow(cs) == 3  # 3 chains from 3 index cases
        @test sum(cs.size) == state.cumulative_cases
    end

    @testset "Leaky isolation allows more transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        # Perfect isolation
        rng1 = StableRNG(42)
        iso_perfect = Isolation(delay=Exponential(1.0), residual_transmission=0.0)
        results_perfect = simulate_batch(model, 200;
            interventions=[iso_perfect],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng1)

        # Leaky isolation (50% residual)
        rng2 = StableRNG(42)
        iso_leaky = Isolation(delay=Exponential(1.0), residual_transmission=0.5)
        results_leaky = simulate_batch(model, 200;
            interventions=[iso_leaky],
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng2)

        # Leaky isolation should contain fewer outbreaks
        cp_perfect = containment_probability(results_perfect)
        cp_leaky = containment_probability(results_leaky)
        @test cp_perfect >= cp_leaky
    end

    @testset "Asymptomatic R scaling reduces transmission" begin
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))

        # All asymptomatic, full R
        rng1 = StableRNG(42)
        results_full = simulate_batch(model, 200;
            sim_opts=SimOpts(max_cases=200, prob_asymptomatic=1.0,
                             asymptomatic_R_scaling=1.0),
            rng=rng1)

        # All asymptomatic, halved R
        rng2 = StableRNG(42)
        results_half = simulate_batch(model, 200;
            sim_opts=SimOpts(max_cases=200, prob_asymptomatic=1.0,
                             asymptomatic_R_scaling=0.5),
            rng=rng2)

        cp_full = containment_probability(results_full)
        cp_half = containment_probability(results_half)
        @test cp_half >= cp_full
    end

    @testset "Latent period enforces minimum generation time" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate_conditioned(model, 20:500;
            sim_opts=SimOpts(max_cases=500, latent_period=3.0),
            rng=rng)

        n_children = 0
        for ind in state.individuals
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                gt = ind.infection_time - parent.infection_time
                @test gt >= 3.0 - 1e-10
                n_children += 1
            end
        end
        @test n_children > 0
    end

    @testset "ringbp_generation_time convenience" begin
        gt_fn = ringbp_generation_time(presymptomatic_fraction=0.3)
        @test gt_fn isa Function

        # Returns a distribution when called with an incubation period
        d = gt_fn(5.0)
        @test d isa Distribution

        # Works in a model
        rng = StableRNG(42)
        model = BranchingProcess(NegBin(2.5, 0.16), gt_fn)
        state = simulate(model;
            sim_opts=SimOpts(max_cases=50, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)
        @test state.cumulative_cases > 0
    end

    @testset "effective_R output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate_conditioned(model, 20:500;
            sim_opts=SimOpts(max_cases=500), rng=rng)

        df = effective_R(state)
        @test df isa DataFrame
        @test "generation" in names(df)
        @test "R_eff" in names(df)
        @test nrow(df) > 0
        @test df.R_eff[1] > 0
    end

    @testset "containment_probability with max_cases" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        results = simulate_batch(model, 100;
            sim_opts=SimOpts(max_cases=50),
            rng=rng)

        # Without max_cases awareness, capped simulations count as extinct
        cp_naive = containment_probability(results)
        # With max_cases awareness, capped simulations are not extinct
        cp_aware = containment_probability(results; max_cases=50)
        @test cp_aware <= cp_naive
    end

    @testset "max_time terminates simulation" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=10_000, max_time=30.0),
            rng=rng)

        max_inf_time = maximum(ind.infection_time for ind in state.individuals)
        @test max_inf_time <= 35.0  # allow some overshoot from generation
    end

    @testset "Analytical vs simulation extinction" begin
        # Compare analytical extinction probability with simulated containment
        R, k = 1.5, 0.5
        q_analytical = extinction_probability(R, k)

        rng = StableRNG(42)
        model = BranchingProcess(NegBin(R, k), Exponential(5.0))
        results = simulate_batch(model, 5000;
            sim_opts=SimOpts(max_cases=10_000, max_generations=200),
            rng=rng)
        q_simulated = containment_probability(results)

        # Should be within a few percentage points
        @test abs(q_analytical - q_simulated) < 0.05
    end
end
