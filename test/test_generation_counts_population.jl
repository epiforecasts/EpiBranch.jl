using Distributions
using StableRNGs
using Statistics

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
            population = Population([400, 600])
        )
        state = simulate_by_generation(
            multi; n_initial = [5, 5], immune = [100, 0], max_generations = 1, rng = StableRNG(1)
        )
        @test state.type_sizes == [400, 600]
        @test state.population_size == 1000
        @test state.susceptibles == [400 - 100 - 5, 600 - 5] .- state.counts
        @test_throws ArgumentError simulate_by_generation(
            multi; n_initial = [5, 5], population_by_type = [400, 600]
        )
        @test_throws ArgumentError simulate_by_generation(
            multi; n_initial = [5, 5], population_by_type = [400, 600], immune = [1, 2, 3]
        )

        # `population_by_type` is deprecated: it still splits a total between
        # types, and says so.
        total = BranchingProcess(
            [0.5 0.1; 0.1 0.5], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population_size = 1000
        )
        old = @test_logs (:warn, r"population_by_type") match_mode = :any simulate_by_generation(
            total; n_initial = [5, 5], immune = [100, 0], population_by_type = [400, 600],
            max_generations = 1, rng = StableRNG(1)
        )
        @test old.type_sizes == state.type_sizes
        @test old.counts == state.counts
        @test_throws ArgumentError simulate_by_generation(
            total; n_initial = [5, 5], population_by_type = [400, 500]
        )

        # A type with many immune is depleted sooner: its contacts are
        # thinned by its own susceptible share, not a shared one. Each type
        # gets the same number of potential contacts, and keeps 150 / 1000
        # and 500 / 1000 of them.
        mixed = BranchingProcess(
            [1.0 1.0; 1.0 1.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population = Population([1000, 1000])
        )
        kept = [
            simulate_by_generation(
                mixed; n_initial = [50, 500], immune = [800, 0], max_generations = 1,
                rng = StableRNG(s)
            ).counts for s in 1:200
        ]
        @test isapprox(mean(first, kept) / mean(last, kept), 0.3; atol = 0.03)
    end

    @testset "the exact draws match deciding contacts one at a time" begin
        # The rule they must reproduce: each contact is kept with the
        # susceptible share of its type at that moment.
        function by_contact(rng, contacts, s, N)
            kept = 0
            for _ in 1:contacts
                s <= 0 && break
                if rand(rng) <= s / N
                    kept += 1
                    s -= 1
                end
            end
            return kept
        end
        function same_law(a, b)
            se = sqrt(var(a) / length(a) + var(b) / length(b))
            @test isapprox(mean(a), mean(b); atol = 5se)
            @test isapprox(var(a), var(b); rtol = 0.1)
        end
        deplete(rng, potential, s, N) = only(
            EpiBranch._deplete!(rng, PerContactDepletion(), potential, [s], [N])
        )
        reps = 20_000
        rng = StableRNG(1)

        # A given number of contacts: several for each susceptible, a few
        # relative to the susceptibles, and more than there are susceptibles.
        for (contacts, s, N) in ((300, 40, 100), (30, 400, 1000), (5, 50, 60), (0, 10, 20))
            same_law(
                [by_contact(rng, contacts, s, N) for _ in 1:reps],
                [deplete(rng, [contacts], s, N) for _ in 1:reps]
            )
        end

        # Poisson contacts with a given mean
        for (rate, s, N) in ((300.0, 40, 100), (30.0, 400, 1000), (0.5, 50, 60))
            same_law(
                [by_contact(rng, rand(rng, Poisson(rate)), s, N) for _ in 1:reps],
                [deplete(rng, [rate], s, N) for _ in 1:reps]
            )
        end

        # The draws never keep more than the susceptibles left
        @test deplete(rng, [10_000], 7, 100) <= 7
        @test deplete(rng, [10_000.0], 7, 100) <= 7
        @test deplete(rng, [10.0], 0, 100) == 0
    end

    @testset "a generation matches drawing its contacts one at a time" begin
        function by_contact(rng, potential, s, N)
            kept = zeros(Int, length(potential))
            for i in eachindex(potential), _ in 1:potential[i]
                s[i] <= 0 && break
                if rand(rng) <= s[i] / N[i]
                    kept[i] += 1
                    s[i] -= 1
                end
            end
            return kept
        end
        reps = 20_000
        function check(model, n_initial, immune; sizes)
            counts = [
                simulate_by_generation(
                    model; n_initial, immune, max_generations = 1, rng = StableRNG(i)
                ).counts for i in 1:reps
            ]
            rng = StableRNG(10^6)
            offspring = EpiBranch._generation_offspring(model)
            current = n_initial isa Integer ? [n_initial] : n_initial
            susceptibles = sizes .- current .- immune
            reference = map(1:reps) do _
                potential = EpiBranch._generation_potential(rng, offspring, current, nothing, false)
                by_contact(rng, potential, copy(susceptibles), sizes)
            end
            return counts, reference
        end
        function same_law(a, b)
            se = sqrt(var(a) / length(a) + var(b) / length(b))
            @test isapprox(mean(a), mean(b); atol = 5se)
            @test isapprox(var(a), var(b); rtol = 0.1)
        end

        # Mixed Poisson offspring, drawn through the Poisson mean
        for law in (Poisson(3.0), NegBin(3.0, 0.5))
            model = BranchingProcess(law; population_size = 300)
            counts, reference = check(model, 40, 60; sizes = [300])
            same_law(counts, only.(reference))
        end

        # Another offspring law, drawn by skipping over the contacts that miss
        model = BranchingProcess(Binomial(6, 0.5); population_size = 300)
        counts, reference = check(model, 40, 60; sizes = [300])
        same_law(counts, only.(reference))

        # Several types, each thinned by its own share
        multi = BranchingProcess(
            [2.0 1.0; 0.5 2.5], R -> NegBin(R, 0.5), EpiBranch.NoGenerationTime();
            population = Population([200, 400])
        )
        counts, reference = check(multi, [30, 10], [50, 100]; sizes = [200, 400])
        for i in 1:2
            same_law(getindex.(counts, i), getindex.(reference, i))
        end
    end

    @testset "a population with many contacts per susceptible is not overcounted" begin
        # The contacts of the generation go to a few people each. The
        # infections are capped at the susceptibles, and approach them.
        capped = simulate_by_generation(
            BranchingProcess(Poisson(50.0); population_size = 100); n_initial = 90,
            max_generations = 1, rng = StableRNG(1)
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
                rng = StableRNG(1)
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
