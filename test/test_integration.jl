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

    @testset "is_extinct classification" begin
        # No kwargs mirrors state.extinct.
        ext = simulate(BranchingProcess(Poisson(0.4), Exponential(5.0)); rng = StableRNG(3))
        @test is_extinct(ext) == ext.extinct

        # A capped run is not extinct under a matching max_cases.
        capped = simulate(BranchingProcess(Poisson(3.0), Exponential(5.0));
            max_cases = 30, rng = StableRNG(5))
        @test !is_extinct(capped; max_cases = 30)

        # by_week bins on onset (with infection fallback) in 7-day blocks from 0.
        m = BranchingProcess(Poisson(1.0), Exponential(5.0))
        st = EpiBranch.new_state(m, EpiBranch.AbstractClinicalTransition[],
            NoAttributes(), StableRNG(1))
        push!(st.individuals,
            Individual(id = 1,
                state = Dict{Symbol, Any}(:infected => true, :onset_time => 6.9)))  # week 1
        push!(st.individuals,
            Individual(id = 2,
                state = Dict{Symbol, Any}(:infected => true, :onset_time => 7.0)))  # week 2
        push!(st.individuals,
            Individual(id = 3, infection_time = 14.0,
                state = Dict{Symbol, Any}(:infected => true)))                      # week 3, fallback
        @test !is_extinct(st; by_week = 1)      # onset 6.9 lands in week 1
        @test !is_extinct(st; by_week = 2)      # onset 7.0 lands in week 2
        @test !is_extinct(st; by_week = 3)      # infection-time fallback lands in week 3
        @test is_extinct(st; by_week = 4)       # nothing in week 4
        @test !is_extinct(st; by_week = 1:2)
    end

    @testset "scenario_sweep" begin
        params = Dict(:offspring => [Poisson(0.8), Poisson(1.2)],
            :generation_time => [Exponential(5.0)])
        res = scenario_sweep(params; n_sim = 50, max_cases = 100, rng = StableRNG(1))
        @test res isa DataFrame
        @test nrow(res) == 2                                   # one row per combination
        @test all(0.0 .<= res.containment_probability .<= 1.0)

        # Reproducible with a fresh RNG of the same seed.
        res2 = scenario_sweep(params; n_sim = 50, max_cases = 100, rng = StableRNG(1))
        @test res.containment_probability == res2.containment_probability

        # An unrecognised key (a simulation control, not a sweep axis) is rejected
        # rather than silently producing a column that does not affect the run.
        @test_throws ArgumentError scenario_sweep(
            Dict(:offspring => [Poisson(0.8)], :max_cases => [50, 100]);
            n_sim = 10, rng = StableRNG(1))
    end

    @testset "susceptible_fraction dispatches" begin
        # NoPopulation → always 1.0.
        m0 = BranchingProcess(Poisson(1.0), Exponential(5.0))
        s0 = EpiBranch.new_state(m0, EpiBranch.AbstractClinicalTransition[],
            NoAttributes(), StableRNG(1))
        @test susceptible_fraction(s0) == 1.0
        @test susceptible_fraction(s0, 1000) == 1.0     # unbounded ignores extra_infected

        # Int population → global depletion, hitting zero at full attack.
        mN = BranchingProcess(Poisson(1.0), Exponential(5.0); population_size = 100)
        sN = EpiBranch.new_state(mN, EpiBranch.AbstractClinicalTransition[],
            NoAttributes(), StableRNG(1))
        @test susceptible_fraction(sN) == 1.0
        sN.cumulative_cases = 40
        @test susceptible_fraction(sN) ≈ 0.6
        @test susceptible_fraction(sN, 60) == 0.0       # extra_infected drives it to 0
        sN.cumulative_cases = 100
        @test susceptible_fraction(sN) == 0.0
    end
end
