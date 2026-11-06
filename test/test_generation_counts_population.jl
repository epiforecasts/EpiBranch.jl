@testset "Generation counts with susceptibles, depletion rules and campaigns" begin
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
            closed = [EpiBranch._sum_offspring(rng, law, n) for _ in 1:reps]
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

    @testset "a vaccination campaign removes susceptibles between generations" begin
        # With few cases in a large population, the susceptibles removed are
        # the doses that reached susceptibles.
        N = 1_000_000
        model = BranchingProcess(Poisson(1.0); population_size = N)
        removed(state) = N - state.cumulative_cases - state.susceptibles
        gv = GroupVaccination(efficacy = 1.0)

        steps = Steps([10.0], [0.0, 100.0])
        campaign = CapacityConstrained(
            gv; budget_per_period = steps, period = 1.0, carry_over = false
        )
        state = simulate_by_generation(
            model; n_initial = 1000, max_generations = 3, generation_length = 7.0,
            interventions = [campaign], rng = StableRNG(1)
        )
        # Periods starting on days 10 to 13, then 14 to 20: 100 doses each
        @test isapprox(removed(state), 1100; rtol = 0.02)

        # Efficacy is the share of doses reaching susceptibles that take
        half = simulate_by_generation(
            model; n_initial = 1000, max_generations = 3, generation_length = 7.0,
            interventions = [
                CapacityConstrained(
                    GroupVaccination(efficacy = 0.5); budget_per_period = steps,
                    period = 1.0, carry_over = false
                ),
            ], rng = StableRNG(1)
        )
        @test isapprox(removed(half), 550; rtol = 0.1)

        # With a single budget, capacity unused before the window opens is
        # carried to the first generation inside it, in either order of
        # wrapping
        for ivs in (
                [Scheduled(CapacityConstrained(gv; budget_per_period = 5000.0); start_time = 14.0)],
                [CapacityConstrained(Scheduled(gv; start_time = 14.0); budget_per_period = 5000.0)],
            )
            early = simulate_by_generation(
                model; n_initial = 1000, max_generations = 2, generation_length = 7.0,
                interventions = ivs, rng = StableRNG(1)
            )
            @test removed(early) == 0
            late = simulate_by_generation(
                model; n_initial = 1000, max_generations = 3, generation_length = 7.0,
                interventions = ivs, rng = StableRNG(1)
            )
            @test isapprox(removed(late), 5000; rtol = 0.01)
        end

        # With no capacity, the coverage of the group is reached at once
        covered = simulate_by_generation(
            model; n_initial = 1000, max_generations = 1, generation_length = 7.0,
            interventions = [GroupVaccination(efficacy = 1.0, coverage = 0.3)],
            rng = StableRNG(1)
        )
        @test isapprox(removed(covered), 0.3N; rtol = 0.01)

        # Continuing a run keeps what a campaign has already done
        straight = simulate_by_generation(
            model; n_initial = 1000, max_generations = 6, generation_length = 7.0,
            interventions = [campaign], rng = StableRNG(2)
        )
        split = simulate_by_generation(
            model; n_initial = 1000, max_generations = 3, generation_length = 7.0,
            interventions = [campaign], rng = StableRNG(2)
        )
        split = simulate(model, split; max_generations = 6)
        @test split.susceptibles == straight.susceptibles
        @test split.counts == straight.counts

        @test_throws ArgumentError simulate_by_generation(model; interventions = [campaign])
        @test_throws ArgumentError simulate_by_generation(
            BranchingProcess(Poisson(1.0)); generation_length = 7.0, interventions = [campaign]
        )
        @test_throws ArgumentError simulate_by_generation(
            model; generation_length = 7.0, interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)]
        )
        @test_throws ArgumentError simulate_by_generation(
            model; generation_length = 7.0,
            interventions = [Scheduled(gv; start_after = ReportedCases(5))]
        )
        @test_throws ArgumentError simulate_by_generation(
            model; generation_length = 7.0,
            interventions = [GroupVaccination(efficacy = 0.5, delay_to_immunity = 3.0)]
        )
    end

    @testset "a campaign splits its doses between types, and reaches susceptibles by their share" begin
        # With no transmission, the susceptibles removed are the doses that
        # reached susceptibles of each type.
        model = BranchingProcess(
            [0.0 0.0; 0.0 0.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population_size = 100_000
        )
        campaign = CapacityConstrained(
            GroupVaccination(efficacy = 1.0); budget_per_period = 10_000.0
        )
        runs = [
            simulate_by_generation(
                model; n_initial = [1, 1], population_by_type = [80_000, 20_000],
                immune = [40_000, 0], max_generations = 1, generation_length = 7.0,
                interventions = [campaign], rng = StableRNG(s)
            ) for s in 1:50
        ]
        # 8,000 and 2,000 doses; half of the first type's members are immune
        @test isapprox(mean(s.susceptibles[1] for s in runs), 80_000 - 40_000 - 1 - 4000; atol = 100)
        @test isapprox(mean(s.susceptibles[2] for s in runs), 20_000 - 1 - 2000; atol = 100)
    end

    @testset "a campaign agrees with individual-based simulation of the same model" begin
        # A well-mixed population with discrete generations of 7 days (a
        # latent period of 7 days and a short infectious one), and a
        # campaign of 40 doses a day from day 14, reaching members
        # whatever their status.
        N, R0, L, B, T, D = 1000, 2.5, 7.0, 40.0, 14.0, 21.0
        group = (rng, ind) -> (ind.state[:group] = 1)
        clinical = clinical_presentation(
            incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
        )
        progression = [
            Transition(:infectious; from = :infection, delay = Dirac(L)),
            Transition(:recovered; from = :infectious, delay = Dirac(0.05), terminal = true),
        ]
        rate = (from, to) -> R0 / (0.05 * N)
        # Each member is offered a dose on a day drawn between T and T + D,
        # and served up to the day's budget.
        vaccination = GroupVaccination(
            efficacy = 0.8, mode = AllOrNothingMode(), eligibility = OnSymptomOnset(),
            dose_delay = Uniform(T, T + D)
        )
        campaign = CapacityConstrained(
            vaccination; budget_per_period = B, period = 1.0, carry_over = false
        )
        individual(ivs, s) = simulate(
            ModelSpec(
                MixingProcess(; population_size = N, mixing_by = (:group,), rate);
                progression, attributes = [group, clinical], interventions = ivs
            );
            n_initial = 10, rng = StableRNG(s)
        ).cumulative_cases
        by_counts(ivs, s) = simulate_by_generation(
            BranchingProcess(Poisson(R0); population_size = N);
            n_initial = 10, max_generations = 60, generation_length = L,
            interventions = ivs, rng = StableRNG(10^6 + s)
        ).cumulative_cases

        n = 60
        sizes = Dict{Bool, Float64}()
        for with_campaign in (false, true)
            a = [individual(with_campaign ? [campaign] : [], s) for s in 1:n]
            b = [
                by_counts(with_campaign ? [Scheduled(campaign; start_time = T)] : [], s)
                    for s in 1:n
            ]
            se = sqrt(var(a) / n + var(b) / n)
            @test isapprox(mean(a), mean(b); atol = 5se)
            sizes[with_campaign] = mean(b)
        end
        @test sizes[true] < 0.8 * sizes[false]
    end
end
