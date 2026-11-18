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
            BranchingProcess(
                Infectiousness(NegBin(2.0, 0.5)), Infectiousness(Poisson(0.5))
            )
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
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(NegBin(2.0, 0.5)); multiplier = Gamma(0.5, 2.0)
        )
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
        totals = [
            simulate_by_generation(
                model; n_initial = I, max_generations = 1,
                multiplier = Gamma(k, 1 / k), rng = StableRNG(s)
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

    @testset "multiplier_per_type draws an independent multiplier for each type" begin
        # Uncoupled types in an unlimited population: each type's count after
        # one generation is its own total, so the counts correlate only through
        # a shared multiplier.
        I, k = 1000, 0.5
        model = BranchingProcess(
            [2.0 0.0; 0.0 2.0], R -> Poisson(R), EpiBranch.NoGenerationTime()
        )
        draw(; kwargs...) = reduce(
            hcat, [
                simulate_by_generation(
                    model; n_initial = [I, I], max_generations = 1,
                    multiplier = Gamma(k, 1 / k), rng = StableRNG(s), kwargs...
                ).counts for s in 1:3000
            ]
        )
        shared = draw()
        per_type = draw(multiplier_per_type = true)
        @test cor(shared[1, :], shared[2, :]) > 0.9
        @test abs(cor(per_type[1, :], per_type[2, :])) < 0.1
        # Each type's total keeps the shared multiplier's marginal law.
        target = NegBin(I * 2.0, k)
        for i in 1:2
            @test isapprox(mean(per_type[i, :]), mean(target); rtol = 0.1)
            @test isapprox(var(per_type[i, :]), var(target); rtol = 0.2)
        end
        # A single type has only one draw to make.
        single = BranchingProcess(Poisson(2.0))
        a = simulate_by_generation(
            single; n_initial = 10, max_generations = 5, multiplier = Gamma(k, 1 / k),
            rng = StableRNG(7)
        )
        b = simulate_by_generation(
            single; n_initial = 10, max_generations = 5, multiplier = Gamma(k, 1 / k),
            multiplier_per_type = true, rng = StableRNG(7)
        )
        @test a.counts == b.counts
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
