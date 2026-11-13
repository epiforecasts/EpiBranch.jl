@testset "Generation counts with susceptibles and depletion rules" begin
    @testset "susceptibles and the immune, per type" begin
        model = BranchingProcess(Poisson(0.5); population_size = 1000)
        state = simulate_by_generation(
            model; n_initial = 10, immune = 300, max_generations = 1, rng = StableRNG(1)
        )
        @test state.type_sizes == 1000
        @test state.susceptibles == 1000 - 300 - 10 - state.counts
        @test_throws ArgumentError simulate_by_generation(model; n_initial = 10, immune = 995)
        @test_throws ArgumentError simulate_by_generation(
            model; n_initial = 10, population_by_type = [500, 500]
        )

        multi = BranchingProcess(
            [0.5 0.1; 0.1 0.5], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population_size = 1000
        )
        state = simulate_by_generation(
            multi; n_initial = [5, 5], immune = [100, 0], population_by_type = [400, 600],
            max_generations = 1, rng = StableRNG(1)
        )
        @test state.type_sizes == [400, 600]
        @test state.susceptibles == [400 - 100 - 5, 600 - 5] .- state.counts
        @test_throws ArgumentError simulate_by_generation(
            multi; n_initial = [5, 5], population_by_type = [400, 500]
        )
        @test_throws ArgumentError simulate_by_generation(
            multi; n_initial = [5, 5], population_by_type = [400, 600], immune = [1, 2, 3]
        )

        # A type with many immune is depleted sooner: its contacts are
        # thinned by its own susceptible share, not a shared one. Each type
        # gets the same number of potential contacts, and keeps 150 / 1000
        # and 500 / 1000 of them.
        mixed = BranchingProcess(
            [1.0 1.0; 1.0 1.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population_size = 2000
        )
        kept = [
            simulate_by_generation(
                mixed; n_initial = [50, 500], immune = [800, 0],
                population_by_type = [1000, 1000], max_generations = 1,
                depletion = StartOfGenerationDepletion(), rng = StableRNG(s)
            ).counts for s in 1:200
        ]
        @test isapprox(mean(first, kept) / mean(last, kept), 0.3; atol = 0.03)
    end

    @testset "the binomial rule agrees with the per-contact rule in a large population" begin
        N = 2_000_000
        model = BranchingProcess(NegBin(2.0, 0.5); population_size = N)
        run(dep, s) = simulate_by_generation(
            model; n_initial = 500, max_generations = 4, depletion = dep, rng = StableRNG(s)
        )
        per_contact = [run(PerContactDepletion(), s).cumulative_cases for s in 1:300]
        binomial = [run(StartOfGenerationDepletion(), 10^6 + s).cumulative_cases for s in 1:300]
        se = sqrt(var(per_contact) / 300 + var(binomial) / 300)
        @test isapprox(mean(per_contact), mean(binomial); atol = 5se)
        @test isapprox(var(per_contact), var(binomial); rtol = 0.35)

        # In a small population the rules part: the binomial one keeps more
        # of a generation's contacts, as its share does not fall as they are
        # used.
        small = BranchingProcess(Poisson(3.0); population_size = 200)
        kept(dep) = mean(
            simulate_by_generation(
                small; n_initial = 50, max_generations = 1, depletion = dep, rng = StableRNG(s)
            ).counts for s in 1:500
        )
        @test kept(StartOfGenerationDepletion()) > kept(PerContactDepletion()) + 3

        # Neither keeps more than the susceptibles left
        capped = simulate_by_generation(
            BranchingProcess(Poisson(50.0); population_size = 100); n_initial = 90,
            max_generations = 1, depletion = StartOfGenerationDepletion(), rng = StableRNG(1)
        )
        @test capped.counts <= 10
        @test capped.susceptibles == 10 - capped.counts
    end

    @testset "closed-form sums match drawing every case" begin
        rng = StableRNG(1)
        n, reps = 40, 20_000
        for law in (Poisson(1.3), NegBin(1.5, 0.4), Binomial(5, 0.3), Geometric(0.4))
            closed = [EpiBranch.sum_offspring(rng, law, n) for _ in 1:reps]
            each = [sum(rand(rng, law) for _ in 1:n) for _ in 1:reps]
            se = sqrt(var(closed) / reps + var(each) / reps)
            @test isapprox(mean(closed), mean(each); atol = 5se)
            @test isapprox(var(closed), var(each); rtol = 0.1)
            @test isapprox(mean(closed), n * mean(law); atol = 5se)
        end
    end

    @testset "a generation costs the same whatever the number of cases" begin
        function per_generation(N)
            model = BranchingProcess(Poisson(2.0); population_size = N)
            run() = simulate_by_generation(
                model; n_initial = 10, max_generations = 25, multiplier = Gamma(6.5, 1 / 6.5),
                depletion = StartOfGenerationDepletion(), rng = StableRNG(1)
            )
            state = run()
            t = minimum(@elapsed(run()) for _ in 1:5)
            return t / state.generation, state.cumulative_cases
        end
        small_t, small_cases = per_generation(10_000)
        large_t, large_cases = per_generation(100_000_000)
        @test large_cases > 1000 * small_cases / 10
        # Up to timing noise: drawing every case would take thousands of
        # times longer for the larger outbreak.
        @test large_t < 10 * small_t + 1.0e-4
    end
end
