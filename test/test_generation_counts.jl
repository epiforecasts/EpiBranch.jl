using Distributions
using StableRNGs
using Statistics

@testset "Simulating a branching process by generation, as counts" begin
    @testset "model validation" begin
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(NegBin(2.0, 0.5), Exponential(2.0))
        )
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(
                Infectiousness(NegBin(2.0, 0.5); until = (:recovered,))
            )
        )
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(Infectiousness(NegBin(2.0, 0.5)), Infectiousness(Poisson(0.5)))
        )
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(
                (rng, ind) -> rand(rng, Poisson(2.0)), EpiBranch.NoGenerationTime()
            )
        )
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(NegBin(2.0, 0.5)); n_initial = [1, 2]
        )
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(NegBin(2.0, 0.5); population_size = 10); n_initial = 20
        )
        @test_throws ArgumentError BranchingProcess(NegBin(2.0, 0.5); multiplier = Gamma(0.5, 2.0))
    end

    @testset "agrees in distribution with per-case simulation" begin
        N = 400
        model = BranchingProcess(NegBin(1.5, 1.0); population_size = N)
        n_reps = 4000

        by_generation = [
            simulate_by_generation(
                model; n_initial = 5, max_generations = 4, rng = StableRNG(s)
            ).cumulative_cases for s in 1:n_reps
        ]
        by_case = [
            simulate(
                model; n_initial = 5, max_generations = 4, max_cases = N,
                rng = StableRNG(10^6 + s)
            ).cumulative_cases for s in 1:n_reps
        ]

        m1, v1 = mean(by_generation), var(by_generation)
        m2, v2 = mean(by_case), var(by_case)
        se = sqrt(v1 / n_reps + v2 / n_reps)
        @test isapprox(m1, m2; atol = 5se)
        @test isapprox(v1, v2; rtol = 0.25)
    end

    @testset "matches the SIR final-size law" begin
        # Same quantitative check as the continuous-time engine's finite
        # population (test_density_dependent.jl): at R0 = 2, major outbreaks'
        # attack rate z solves z = 1 - exp(-R0 z), z ≈ 0.7968.
        N = 3000
        model = BranchingProcess(Poisson(2.0); population_size = N)
        finals = [
            simulate_by_generation(
                model; n_initial = 1, max_generations = 40, rng = StableRNG(s)
            ).cumulative_cases for s in 1:200
        ]
        major = filter(x -> x > 0.2N, finals)
        @test !isempty(major)
        @test isapprox(mean(major) / N, 0.7968; atol = 0.03)
    end

    @testset "multiplier gives a fixed-shape negative-binomial total" begin
        # An unlimited population leaves a generation's draw unthinned, so
        # `state.counts` after one generation is exactly the total this
        # generation's cases drew.
        I, λ, k = 1000, 2.0, 0.5
        model = BranchingProcess(Poisson(λ))
        shared = BranchingProcess(Poisson(λ); multiplier = Gamma(k, 1 / k))
        totals = [
            simulate_by_generation(
                shared; n_initial = I, max_generations = 1, rng = StableRNG(s)
            ).counts for s in 1:4000
        ]
        target = NegBin(I * λ, k)
        se_mean = sqrt(var(target) / length(totals))
        @test isapprox(mean(totals), mean(target); atol = 5se_mean)
        @test isapprox(var(totals), var(target); rtol = 0.2)

        # With no multiplier, summing I iid NegBin(R, k) cases instead gives a
        # total of shape I * k, far narrower (relative to its mean) than the
        # shared-multiplier total above.
        no_multiplier = [
            simulate_by_generation(
                model; n_initial = I, max_generations = 1, rng = StableRNG(s)
            ).counts for s in 1:1000
        ]
        @test var(no_multiplier) < var(totals) / 10
    end

    @testset "multi-type offspring" begin
        M = [1.2 0.4; 0.3 0.9]
        model = BranchingProcess(
            M, R -> NegBin(R, 0.5), EpiBranch.NoGenerationTime();
            population_size = 5000
        )
        @test_throws ArgumentError simulate_by_generation(
            model; n_initial = [5, 5], max_generations = 10
        )
        model = BranchingProcess(
            M, R -> NegBin(R, 0.5), EpiBranch.NoGenerationTime();
            population = Population([2000, 3000])
        )
        state = simulate_by_generation(
            model; n_initial = [5, 5], max_generations = 10, rng = StableRNG(1)
        )
        @test state.counts isa Vector{Int}
        @test length(state.counts) == 2
        @test state.cumulative_cases >= 10
    end

    @testset "a vector of multipliers draws one independently for each type" begin
        # Uncoupled types in an unlimited population: the count of each type
        # after one generation is its own total, so the counts correlate only
        # through a shared multiplier.
        I, k = 1000, 0.5
        model(multiplier) = BranchingProcess(
            [2.0 0.0; 0.0 2.0], R -> Poisson(R), EpiBranch.NoGenerationTime(); multiplier
        )
        draw(multiplier) = reduce(
            hcat, [
                simulate_by_generation(
                    model(multiplier); n_initial = [I, I], max_generations = 1,
                    rng = StableRNG(s)
                ).counts for s in 1:3000
            ]
        )
        shared = draw(Gamma(k, 1 / k))
        per_type = draw([Gamma(k, 1 / k), Gamma(k, 1 / k)])
        @test cor(shared[1, :], shared[2, :]) > 0.9
        @test abs(cor(per_type[1, :], per_type[2, :])) < 0.1
        # The total of each type keeps the marginal law of the shared multiplier.
        target = NegBin(I * 2.0, k)
        for i in 1:2
            @test isapprox(mean(per_type[i, :]), mean(target); rtol = 0.1)
            @test isapprox(var(per_type[i, :]), var(target); rtol = 0.2)
        end
        # A single type has only one draw to make.
        a = simulate_by_generation(
            BranchingProcess(Poisson(2.0); multiplier = Gamma(k, 1 / k));
            n_initial = 10, max_generations = 5, rng = StableRNG(7)
        )
        b = simulate_by_generation(
            BranchingProcess(Poisson(2.0); multiplier = [Gamma(k, 1 / k)]);
            n_initial = 10, max_generations = 5, rng = StableRNG(7)
        )
        @test a.counts == b.counts
        @test_throws ArgumentError model([Gamma(k, 1 / k)])
        @test_throws ArgumentError BranchingProcess(
            (rng, ind) -> rand(rng, Poisson(2.0)), EpiBranch.NoGenerationTime();
            multiplier = Gamma(k, 1 / k)
        )
    end

    @testset "continuing a run matches running straight through" begin
        model = BranchingProcess(NegBin(2.0, 0.5); population_size = 5000)
        straight = simulate_by_generation(
            model; n_initial = 5, max_generations = 20, rng = StableRNG(3)
        )
        split = simulate_by_generation(
            model; n_initial = 5, max_generations = 10, rng = StableRNG(3)
        )
        split = simulate(model, split; max_generations = 20)
        @test split.counts == straight.counts
        @test split.cumulative_cases == straight.cumulative_cases
        @test split.generation == straight.generation
    end

    @testset "extinction stops the run early" begin
        # R = 0.1 from a single index case goes extinct within a handful of
        # generations with overwhelming probability.
        model = BranchingProcess(NegBin(0.1, 0.5))
        state = simulate_by_generation(
            model; n_initial = 1, max_generations = 100, rng = StableRNG(4)
        )
        @test state.counts == 0
        @test state.generation < 100
        @test simulate(model, state; max_generations = 200) === state
    end
end

@testset "A single type allocates nothing per generation" begin
    finite = BranchingProcess(NegBin(2.0, 0.5); population = Population(200_000))
    unlimited = BranchingProcess(Poisson(0.9))
    shared = BranchingProcess(Poisson(0.9); multiplier = Gamma(5, 1 / 5))
    vaccinated = ModelSpec(
        BranchingProcess(Poisson(1.5); population = Population(100_000));
        interventions = [
            CapacityConstrained(
                GroupVaccination(efficacy = 0.9, group_key = :type);
                budget_per_period = 50.0, period = 1.0
            ),
        ]
    )
    continue_to(model, state, n; kw...) = simulate(model, state; max_generations = n, kw...)
    for (model, kw) in (
            (finite, (;)), (unlimited, (;)), (shared, (;)),
            (vaccinated, (; generation_length = 7.0)),
        )
        state = simulate_by_generation(
            model; n_initial = 1000, max_generations = 2, rng = StableRNG(1), kw...
        )
        # Each run uses a fresh stream with the same seed, so the runs draw alike.
        # The checks below guard against an extinct chain passing trivially.
        continue_to(model, state, 4; rng = StableRNG(2))
        short = @allocated continue_to(model, state, 4; rng = StableRNG(2))
        long = @allocated continue_to(model, state, 14; rng = StableRNG(2))
        reached = continue_to(model, state, 14; rng = StableRNG(2))
        @test reached.generation == 14
        @test reached.counts > 0
        @test long == short
    end
end

@testset "An individual run of a model with a multiplier and immune people" begin
    @testset "a multiplier is drawn once per generation for every case" begin
        I, λ, k = 100, 2.0, 0.5
        model = BranchingProcess(
            Poisson(λ), Exponential(1.0); multiplier = Gamma(k, 1 / k)
        )
        @test EpiBranch.count_rule(model) === nothing
        totals = [
            simulate(
                model; n_initial = I, max_generations = 1, rng = StableRNG(s)
            ).cumulative_cases - I for s in 1:2000
        ]
        target = NegBin(I * λ, k)
        @test isapprox(mean(totals), mean(target); atol = 5 * sqrt(var(target) / length(totals)))
        @test isapprox(var(totals), var(target); rtol = 0.2)
    end

    @testset "a vector of multipliers is drawn for each type in each generation" begin
        I, k = 100, 0.5
        model(multiplier) = BranchingProcess(
            [2.0 0.0; 0.0 2.0], R -> Poisson(R), EpiBranch.NoGenerationTime(); multiplier
        )
        draw(multiplier) = reduce(
            hcat, [
                begin
                    state = simulate(
                        model(multiplier); index_types = repeat([1, 2], inner = I),
                        max_generations = 1, rng = StableRNG(s)
                    )
                    [
                        count(i -> i.generation == 1 && individual_type(i) == j, individuals(state))
                            for j in 1:2
                    ]
                end for s in 1:1500
            ]
        )
        shared = draw(Gamma(k, 1 / k))
        per_type = draw([Gamma(k, 1 / k), Gamma(k, 1 / k)])
        @test cor(shared[1, :], shared[2, :]) > 0.9
        @test abs(cor(per_type[1, :], per_type[2, :])) < 0.15
    end

    @testset "immune people are never infected and agree with the counts" begin
        N, n_reps = 400, 3000
        model = BranchingProcess(
            NegBin(2.0, 1.0); population = Population(N; immune = 200),
            multiplier = nothing
        )
        @test EpiBranch.count_rule(model) === nothing
        by_case = [
            simulate(
                model; n_initial = 5, max_generations = 4, max_cases = N,
                rng = StableRNG(s)
            ).cumulative_cases for s in 1:n_reps
        ]
        by_generation = [
            simulate_by_generation(
                model; n_initial = 5, max_generations = 4, rng = StableRNG(10^6 + s)
            ).cumulative_cases for s in 1:n_reps
        ]
        se = sqrt(var(by_case) / n_reps + var(by_generation) / n_reps)
        @test isapprox(mean(by_case), mean(by_generation); atol = 5se)
        @test maximum(by_case) <= N - 200
    end

    @testset "a multiplier and immunity together agree with the counts" begin
        N, n_reps = 600, 3000
        model = BranchingProcess(
            Poisson(1.8); population = Population(N; immune = 150),
            multiplier = Gamma(2.0, 0.5)
        )
        by_case = [
            simulate(
                model; n_initial = 5, max_generations = 5, max_cases = N,
                rng = StableRNG(s)
            ).cumulative_cases for s in 1:n_reps
        ]
        by_generation = [
            simulate_by_generation(
                model; n_initial = 5, max_generations = 5, rng = StableRNG(10^6 + s)
            ).cumulative_cases for s in 1:n_reps
        ]
        se = sqrt(var(by_case) / n_reps + var(by_generation) / n_reps)
        @test isapprox(mean(by_case), mean(by_generation); atol = 5se)
        @test isapprox(var(by_case), var(by_generation); rtol = 0.25)
    end

    @testset "a multiplier keeps tracing able to draw before settling" begin
        model = BranchingProcess(Poisson(2.0), Exponential(2.0); multiplier = Gamma(2.0, 0.5))
        wrapped = EpiBranch._multiplied(model.infectiousness, model.multiplier)
        plain = model.infectiousness
        @test all(EpiBranch._draws_before_settling, plain)
        @test all(EpiBranch._draws_before_settling, wrapped)
    end

    @testset "a model that does not read the immune refuses them" begin
        @test_throws ArgumentError HomogeneousProcess(
            transmission_rate = 0.4, population = Population(100; immune = 10)
        )
    end

    @testset "simulate dispatches on the model's resolution" begin
        pop = Population(5_000)
        plain = BranchingProcess(NegBin(2.0, 0.5); population = pop)
        @test plain.resolution isa Individuals
        @test simulate(plain; n_initial = 5, max_generations = 4, rng = StableRNG(1)) isa
            SimulationState

        model = BranchingProcess(
            NegBin(2.0, 0.5); population = pop,
            resolution = ByGeneration(; generation_length = 7.0)
        )
        state = simulate(model; n_initial = 5, max_generations = 6, rng = StableRNG(1))
        @test state isa GenerationCounts
        @test state.generation_length == 7.0
        wrapper = simulate_by_generation(
            plain; n_initial = 5, max_generations = 6, generation_length = 7.0,
            rng = StableRNG(1)
        )
        @test wrapper.counts == state.counts
        @test wrapper.cumulative_cases == state.cumulative_cases

        kept = simulate_by_generation(
            model; n_initial = 5, max_generations = 6, rng = StableRNG(1)
        )
        @test kept.generation_length == 7.0
        @test kept.counts == state.counts
        overridden = simulate_by_generation(
            model; n_initial = 5, max_generations = 6, generation_length = 3.0,
            rng = StableRNG(1)
        )
        @test overridden.generation_length == 3.0

        spec = ModelSpec(model)
        @test simulate(spec; n_initial = 5, max_generations = 6, rng = StableRNG(1)).counts ==
            state.counts

        multi = BranchingProcess(
            [0.5 0.1; 0.1 0.5], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population = Population([400, 600]), resolution = ByGeneration()
        )
        @test simulate(multi; n_initial = [5, 5], max_generations = 3, rng = StableRNG(1)) isa
            GenerationCounts
    end
end
