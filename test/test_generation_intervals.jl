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
        expected = count(
            ind -> is_infected(ind) && ind.parent_id != 0,
            state.individuals
        )
        @test length(gts) == expected
        @test all(>=(0), gts)

        # Vector-of-states method flattens.
        states = simulate(
            model, 5; max_cases = 200,
            rng = StableRNG(3)
        )
        all_gts = realised_generation_intervals(states)
        @test length(all_gts) == sum(
            length(realised_generation_intervals(s))
                for s in states
        )
    end

    @testset "Intervention-free run recovers the intrinsic mean (no depletion)" begin
        # With NoPopulation and no interventions nothing is blocked, and
        # subcritical outbreaks run to extinction rather than being cut off
        # partway through an interval, so the realised intervals are the raw
        # generation-time draws.
        rng = StableRNG(4)
        gt_mean = 5.0
        model = BranchingProcess(Poisson(0.9), Exponential(gt_mean))
        states = simulate(model, 1000; max_cases = nothing, rng = rng)
        @test all(s -> s.extinct, states)
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
            simulate(
                ModelSpec(BranchingProcess(Poisson(2.5), Exponential(6.0)); attributes = attrs),
                30; opts..., rng = rng_a
            )
        )
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        isolated = realised_generation_intervals(
            simulate(
                ModelSpec(
                    BranchingProcess(Poisson(2.5), Exponential(6.0));
                    interventions = [iso], attributes = attrs
                ),
                30;
                opts...,
                rng = rng_b
            )
        )

        @test !isempty(free)
        @test !isempty(isolated)
        @test sum(isolated) / length(isolated) < sum(free) / length(free)
    end

    @testset "Realised serial intervals" begin
        attrs = clinical_presentation(
            incubation_period = LogNormal(1.0, 0.3), prob_asymptomatic = 0.3
        )
        model = ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0)); attributes = attrs
        )
        state = simulate(model; max_cases = 300, rng = StableRNG(11))
        sis = realised_serial_intervals(state)

        pairs = [
            (ind, state.individuals[ind.parent_id])
                for ind in state.individuals if is_infected(ind) && ind.parent_id != 0
        ]
        both = [
            onset_time(c) - onset_time(p) for (c, p) in pairs
                if !isnan(onset_time(c)) && !isnan(onset_time(p))
        ]
        @test !isempty(sis)
        @test sis == both
        @test length(sis) < length(pairs)

        for (c, _) in pairs
            @test isnan(realised_serial_interval(c, state)) ==
                (isnan(onset_time(c)) || isnan(onset_time(state.individuals[c.parent_id])))
        end
        idx = first(i for i in state.individuals if i.parent_id == 0)
        @test isnan(realised_serial_interval(idx, state))

        states = simulate(model, 4; max_cases = 100, rng = StableRNG(12))
        @test realised_serial_intervals(states) ==
            reduce(vcat, realised_serial_intervals.(states))
    end
end
