@testset "Generation intervals" begin
    @testset "Per-individual accessor" begin
        rng = StableRNG(1)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; max_cases = 200, rng = rng)

        # Index cases have no parent.
        for ind in state.individuals
            if ind.parent_id == 0
                @test isnan(realised_generation_interval(ind, state))
            end
        end

        # Infected non-index cases: interval equals the infection-time gap.
        for ind in state.individuals
            if is_infected(ind) && ind.parent_id != 0
                parent = state.individuals[ind.parent_id]
                @test realised_generation_interval(ind, state) ≈
                      ind.infection_time - parent.infection_time
                @test realised_generation_interval(ind, state) >= 0
            end
        end
    end

    @testset "Batch reducer collects all infected non-index cases" begin
        rng = StableRNG(2)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; max_cases = 200, rng = rng)

        gts = realised_generation_intervals(state)
        expected = count(ind -> is_infected(ind) && ind.parent_id != 0,
            state.individuals)
        @test length(gts) == expected
        @test all(>=(0), gts)

        # Vector-of-states method flattens.
        states = simulate(model, 5; max_cases = 200,
            rng = StableRNG(3))
        all_gts = realised_generation_intervals(states)
        @test length(all_gts) == sum(length(realised_generation_intervals(s))
        for s in states)
    end

    @testset "Intervention-free run recovers the intrinsic mean (no depletion)" begin
        # With NoPopulation and no interventions nothing is blocked, so the
        # realised intervals are the raw generation-time draws.
        rng = StableRNG(4)
        gt_mean = 5.0
        model = BranchingProcess(Poisson(1.5), Exponential(gt_mean))
        states = simulate(model, 50; max_cases = 500,
            rng = rng)
        gts = realised_generation_intervals(states)
        @test length(gts) > 5000
        @test isapprox(sum(gts) / length(gts), gt_mean; rtol = 0.05)
    end

    @testset "Isolation shortens the realised generation interval" begin
        # Isolation blocks late transmissions, so realised intervals among
        # those that still occur are shorter.
        rng_a = StableRNG(7)
        rng_b = StableRNG(7)
        attrs = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        opts = (; max_cases = 1000)

        free = realised_generation_intervals(
            simulate(BranchingProcess(Poisson(2.5), Exponential(6.0); attributes = attrs),
            30; opts..., rng = rng_a))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        isolated = realised_generation_intervals(
            simulate(
            BranchingProcess(Poisson(2.5), Exponential(6.0); interventions = [iso], attributes = attrs),
            30;
            opts...,
            rng = rng_b))

        @test !isempty(free)
        @test !isempty(isolated)
        @test sum(isolated) / length(isolated) < sum(free) / length(free)
    end

    @testset "Works for NetworkProcess" begin
        n = 150
        adj = [sort(unique([mod1(i - 1, n), mod1(i + 1, n),
                   mod1(i + 5, n), mod1(i - 5, n)])) for i in 1:n]
        model = NetworkProcess(adj, 0.5, LogNormal(1.6, 0.5))
        state = simulate(model;
            n_initial = 1,
            stopping_rules = [Extinction(), MaxGenerations(100)],
            rng = StableRNG(9))
        gts = realised_generation_intervals(state)
        @test all(>=(0), gts)
        @test length(gts) ==
              count(ind -> is_infected(ind) && ind.parent_id != 0,
            state.individuals)
    end
end
