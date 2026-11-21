using EpiBranch, Distributions, Random, StableRNGs, Statistics, DataFrames

@testset "Count-level simulation" begin
    gt = DiscreteNonParametric(1:3, [0.3, 0.5, 0.2])
    count_run(model; n_initial = 4, max_time, rng) = simulate(
        model; n_initial, max_time, max_cases = nothing, max_generations = nothing, rng
    )
    individual_run(model; n_initial = 4, max_time, rng) = simulate(
        model; n_initial, stopping_rules = [MaxTime(Float64(max_time))], rng
    )

    # Daily infections, padded to `T + 1` days.
    function daily(state, T)
        counts = zeros(Int, T + 1)
        for ind in individuals(state)
            get(ind.state, :infected, false) || continue
            counts[Int(ind.infection_time) + 1] += 1
        end
        return counts
    end

    # The offspring of each infector by lag, for infectors whose offspring are
    # complete by `T`.
    function offspring_by_lag(state, T, L)
        people = individuals(state)
        by_lag = zeros(Int, length(people), L)
        for ind in people
            ind.parent_id == 0 && continue
            lag = Int(ind.infection_time - people[ind.parent_id].infection_time)
            by_lag[ind.parent_id, lag] += 1
        end
        return by_lag[[i for i in eachindex(people) if people[i].infection_time <= T - L], :]
    end

    function compare(model, T, L, n)
        counts = zeros(Int, n, T + 1, 2)
        offspring = (zeros(Int, 0, L), zeros(Int, 0, L))
        for i in 1:n, (j, run) in enumerate((count_run, individual_run))
            state = run(model; max_time = T, rng = StableRNG(1000 * j + i))
            counts[i, :, j] = daily(state, T)
            offspring = Base.setindex(offspring, vcat(offspring[j], offspring_by_lag(state, T, L)), j)
        end
        return counts, offspring
    end

    # Correlation across lags of one infector's offspring when the offspring
    # number has mean `R` and variance `V` and is split over the lags by
    # `weights`. Each simulation is compared with this value rather than with
    # the other simulation, whose sampling error is large with a heavy tail.
    function split_correlation(R, V, weights)
        v = [w * R + w^2 * (V - R) for w in weights]
        return [
            i == j ? 1.0 : weights[i] * weights[j] * (V - R) / sqrt(v[i] * v[j])
                for i in eachindex(weights), j in eachindex(weights)
        ]
    end

    function check_agreement(model, T, L, n; reference = nothing, atol = 0.05)
        counts, offspring = compare(model, T, L, n)
        a, b = counts[:, :, 1], counts[:, :, 2]
        # Daily means, within the sampling error of the difference.
        for t in 2:(T + 1)
            se = sqrt(var(a[:, t]) / n + var(b[:, t]) / n)
            @test abs(mean(a[:, t]) - mean(b[:, t])) < 4.5 * se
        end
        # Daily variances.
        for t in 3:(T + 1)
            @test var(a[:, t]) ≈ var(b[:, t]) rtol = 0.25
        end
        # Lag-one autocorrelation of daily counts. The estimate for each day
        # is noisy, so days are compared loosely one by one. Their average has
        # far less sampling error and is compared tightly.
        diffs = [cor(a[:, t], a[:, t + 1]) - cor(b[:, t], b[:, t + 1]) for t in 2:T]
        @test all(abs.(diffs) .< 0.1)
        @test abs(mean(diffs)) < 0.03
        # Correlation across days of one infector's offspring.
        if reference === nothing
            @test cor(offspring[1]) ≈ cor(offspring[2]) atol = atol
        else
            @test cor(offspring[1]) ≈ reference atol = 0.03
            @test cor(offspring[2]) ≈ reference atol = 0.03
        end
        @test vec(mean(offspring[1]; dims = 1)) ≈ vec(mean(offspring[2]; dims = 1)) rtol = 0.06
        return cor(offspring[1])
    end

    @testset "agrees with the individual simulation" begin
        weights = [0.3, 0.5, 0.2]
        c = check_agreement(
            BranchingProcess(Poisson(1.3), gt), 8, 3, 1000;
            reference = split_correlation(1.3, 1.3, weights)
        )
        @test all(abs(c[i, j]) < 0.03 for i in 1:3, j in 1:3 if i != j)

        # A negative binomial offspring number is correlated across the days
        # of one infector.
        c = check_agreement(
            BranchingProcess(NegBin(1.3, 0.8), gt), 8, 3, 1000;
            reference = split_correlation(1.3, 1.3 + 1.3^2 / 0.8, weights)
        )
        @test c[1, 2] > 0.25

        R(t) = t < 5 ? 1.6 : 0.8
        renewal = BranchingProcess(
            Infectiousness(RenewalOffspring(R, [0.3, 0.5, 0.2]; k = 1.0))
        )
        c = check_agreement(renewal, 8, 3, 1000)
        @test c[1, 2] > 0.25
    end

    @testset "Poisson mixtures agree with the individual simulation" begin
        # Offspring numbers are correlated across the days of one infector,
        # through a rate that is neither gamma nor constant.
        lognormal = PoissonLognormal(log(1.3) - 0.245, 0.7)
        weibull = PoissonWeibull(1.4, 1.3 / mean(Weibull(1.4, 1.0)))
        for offspring in (lognormal, weibull)
            rule = EpiBranch.count_rule(BranchingProcess(offspring, gt))
            @test mean(rule.ν) ≈ 1 atol = 1.0e-9
            @test rule.R(1) ≈ mean(offspring)
            c = check_agreement(BranchingProcess(offspring, gt), 8, 3, 4000)
            @test c[1, 2] > 0.1
        end
    end

    @testset "other offspring laws agree with the individual simulation" begin
        # Binomial offspring are less variable than Poisson, so the offspring
        # of one infector are negatively correlated across the days.
        weights = [0.3, 0.5, 0.2]
        c = check_agreement(
            BranchingProcess(Binomial(5, 0.3), gt), 8, 3, 1000;
            reference = split_correlation(1.5, 1.05, weights)
        )
        @test c[1, 2] < -0.1
        geometric = Geometric(0.4)
        c = check_agreement(
            BranchingProcess(geometric, gt), 8, 3, 2000;
            reference = split_correlation(mean(geometric), var(geometric), weights)
        )
        @test c[1, 2] > 0.1
    end

    @testset "the members of a cohort keep their own offspring counts" begin
        model = BranchingProcess(Binomial(5, 0.3), gt)
        @test EpiBranch.count_rule(model) isa EpiBranch.OffspringCountRule
        state = count_run(model; max_time = 12, rng = StableRNG(4))
        members = state.cohort_members
        @test all(length(m[1]) == n for (m, n) in zip(members, state.counts))
        people = individuals(state)
        children = zeros(Int, length(people))
        for ind in people
            ind.parent_id > 0 && (children[ind.parent_id] += 1)
        end
        # Once every infection of a cohort has fallen by the last day, each
        # member has had all of its offspring.
        offset = 0
        for d in 0:state.day
            n = state.counts[d + 1]
            if d + length(gt.support) <= state.day
                @test children[(offset + 1):(offset + n)] == Int.(members[d + 1][1])
            end
            offset += n
        end
        @test sum(children) == state.cumulative_cases - state.counts[1]
        # A law with a point mass at zero ends the run.
        dead = count_run(BranchingProcess(Binomial(3, 0.0), gt); max_time = 6, rng = StableRNG(1))
        @test dead.extinct
        @test dead.cumulative_cases == 4
    end

    @testset "the members of a cohort keep their infectiousness" begin
        # With a mixed Poisson law the split of the infections of a cohort
        # among its members depends on the total, so the members are kept.
        model = BranchingProcess(PoissonLognormal(-0.2, 0.9), gt)
        state = count_run(model; max_time = 6, rng = StableRNG(4))
        @test length(state.cohort_members) == state.day + 1
        @test all(
            length(m) == n && sum(m) ≈ total for
                (m, n, total) in zip(
                    first.(state.cohort_members), state.counts, first.(state.cohort_infectiousness)
                )
        )
        @test length(individuals(state)) == state.cumulative_cases
        negbin = count_run(BranchingProcess(NegBin(1.3, 0.8), gt); max_time = 6, rng = StableRNG(4))
        @test all(isempty ∘ only, negbin.cohort_members)
    end

    @testset "the people of a state" begin
        model = BranchingProcess(NegBin(1.5, 0.6), gt)
        state = count_run(model; n_initial = 3, max_time = 14, rng = StableRNG(7))
        @test state isa CountSimulationState
        @test state isa AbstractSimulationState
        @test state.built === nothing
        people = individuals(state)
        @test state.built !== nothing
        @test length(people) == state.cumulative_cases == sum(state.counts)
        @test individuals(state) === people
        @test count(ind -> ind.parent_id == 0, people) == 3
        @test all(ind -> ind.parent_id < ind.id, people)
        @test all(people) do ind
            ind.parent_id == 0 ||
                ind.infection_time - people[ind.parent_id].infection_time in 1:3
        end
        @test all(ind -> ind.generation == (ind.parent_id == 0 ? 0 : people[ind.parent_id].generation + 1), people)
        @test [Int(ind.infection_time) for ind in people] == sort([Int(ind.infection_time) for ind in people])

        # The same seed gives the same line list whenever it is asked for.
        again = count_run(model; n_initial = 3, max_time = 14, rng = StableRNG(7))
        @test isequal(linelist(state), linelist(again))
        @test isequal(linelist(state), linelist(again))
        @test nrow(contacts(state)) == nrow(linelist(state)) - 3
        @test sum(chain_statistics(state).size) == state.cumulative_cases
        @test sum(weekly_incidence(state).cases) == state.cumulative_cases

        # The columns of the individual simulation's line list.
        reference = linelist(individual_run(model; max_time = 14, rng = StableRNG(1)))
        @test names(linelist(state)) == names(reference)
    end

    @testset "natural history and observation are drawn with the line list" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(1.4), gt);
            attributes = clinical_presentation(incubation_period = Gamma(4.0, 1.0))
        )
        state = count_run(spec; max_time = 12, rng = StableRNG(3))
        @test state isa CountSimulationState
        ll = linelist(state)
        @test "date_onset" in names(ll)
        @test "incubation_period" in names(ll)
        @test all(>(0), ll.incubation_period)
        @test names(ll) == names(linelist(individual_run(spec; max_time = 12, rng = StableRNG(3))))
    end

    @testset "continuation" begin
        model = BranchingProcess(NegBin(1.4, 0.7), gt)
        straight = count_run(model; max_time = 20, rng = StableRNG(11))
        state = count_run(model; max_time = 8, rng = StableRNG(11))
        individuals(state)
        @test simulate(model, state; max_time = 20) === state
        @test state.built === nothing
        @test state.counts == straight.counts
        @test state.cumulative_cases == straight.cumulative_cases
        @test isequal(linelist(state), linelist(straight))
        @test_throws ArgumentError simulate(model, state; max_time = 5)
        # A different RNG type can take over the continuation.
        other = count_run(model; max_time = 8, rng = StableRNG(11))
        @test simulate(model, other; max_time = 12, rng = Xoshiro(2)) === other
        @test other.rng isa Xoshiro
        @test_throws ArgumentError simulate(
            BranchingProcess(NegBin(1.4, 0.7), Gamma(2.0, 1.0)), state; max_time = 30
        )
    end

    @testset "independent copies" begin
        model = BranchingProcess(PoissonLognormal(-0.2, 0.9), gt)
        for realised in (false, true)
            state = count_run(model; max_time = 4, rng = StableRNG(11))
            realised && individuals(state)
            other = copy(state)
            @test other !== state
            @test isequal(linelist(other), linelist(state))
            @test individuals(other) !== individuals(state)
            expected = deepcopy(state)
            simulate(model, other; max_time = 8)
            simulate(model, expected; max_time = 8)
            @test other.counts == expected.counts
            @test isequal(linelist(other), linelist(expected))
            @test state.day == 4
            @test length(state.counts) == 5
        end

        Random.seed!(1234)
        state = simulate(
            model; n_initial = 4, max_time = 4, max_cases = nothing,
            max_generations = nothing
        )
        other = copy(state)
        simulate(model, other; max_time = 8)
        simulate(model, state; max_time = 8)
        @test other.counts == state.counts
        @test isequal(linelist(other), linelist(state))
    end

    @testset "extinction" begin
        state = count_run(
            BranchingProcess(Poisson(0.2), gt); n_initial = 1, max_time = 200, rng = StableRNG(2)
        )
        @test state.extinct
        @test state.day < 200
        @test length(individuals(state)) == state.cumulative_cases
        @test is_extinct(state)
    end

    @testset "max_cases ends the run at the end of a day" begin
        model = BranchingProcess(Poisson(2.0), gt)
        state = simulate(
            model; n_initial = 1, max_cases = 50, max_generations = nothing,
            rng = StableRNG(3)
        )
        @test state isa CountSimulationState
        @test state.cumulative_cases >= 50
        @test sum(state.counts[1:(end - 1)]) < 50
        # Continuing runs past the cap.
        simulate(model, state; max_time = state.day + 3)
        @test state.day > 0 && length(state.counts) == state.day + 1
        # A cap met by the index cases ends the run on day 0.
        @test simulate(
            model; n_initial = 60, max_cases = 50, max_generations = nothing,
            rng = StableRNG(3)
        ).day == 0
    end

    @testset "initial cases are checked against the population" begin
        model = BranchingProcess(Poisson(2.0), gt; population = Population(5))
        @test_throws ArgumentError simulate(
            model; n_initial = 10, max_time = 5, max_cases = nothing,
            max_generations = nothing, rng = StableRNG(1)
        )
    end

    @testset "containment_probability on counts" begin
        # The same event as on individuals: extinct with fewer than `max_cases` cases.
        for offspring in (Poisson(1.3), NegativeBinomial(0.5, 0.3))
            model = BranchingProcess(offspring, gt)
            kw = (; max_cases = 60, n_initial = 1)
            on_counts = containment_probability(
                model, 4000; kw..., max_generations = nothing, rng = StableRNG(5)
            )
            on_individuals = containment_probability(
                model, 4000; kw..., stopping_rules = [MaxCases(60), MaxGenerations(1000)],
                rng = StableRNG(6)
            )
            @test abs(on_counts.proportion - on_individuals.proportion) <
                5 * hypot(on_counts.mcse, on_individuals.mcse)
        end
        # A model that qualifies runs on counts and still returns `n` outcomes.
        model = BranchingProcess(Poisson(1.3), gt)
        @test containment_probability(
            model, 50; max_cases = 60, max_generations = nothing, rng = StableRNG(1)
        ).n == 50
        spec = ModelSpec(model)
        @test containment_probability(
            spec, 50; max_cases = 60, max_generations = nothing, rng = StableRNG(1)
        ).n == 50
    end

    @testset "a generation cap" begin
        model = BranchingProcess(NegativeBinomial(0.8, 0.3), gt)
        # The people built from the counts have the generations the counts hold.
        state = simulate(
            model; n_initial = 3, max_time = 30, max_cases = nothing, max_generations = 4,
            rng = StableRNG(1)
        )
        gens = [i.generation for i in individuals(state)]
        @test maximum(gens) <= 4
        padded = [vcat(vec(c), zeros(Int, 5 - length(c))) for c in state.generation_counts]
        @test [count(==(k), gens) for k in 0:4] == sum(padded)
        @test state.counts == sum.(state.generation_counts)
        # A cap beyond what the run can reach costs no more than the days run.
        loose = simulate(
            model; n_initial = 3, max_time = 30, max_cases = nothing,
            max_generations = 10_000_000, rng = StableRNG(1)
        )
        @test maximum(length, loose.generation_counts) <= 31
        @test sum(loose.counts) == loose.cumulative_cases
        # A run that reaches the cap is not extinct; one that dies out first is.
        capped = simulate(
            BranchingProcess(Poisson(4.0), gt); max_time = 40, max_cases = nothing,
            max_generations = 2, rng = StableRNG(2)
        )
        @test !capped.extinct && length(individuals(capped)) == capped.cumulative_cases
        @test simulate(
            BranchingProcess(Poisson(0.1), gt); max_time = 100, max_cases = nothing,
            max_generations = 50, rng = StableRNG(3)
        ).extinct
        alone = simulate(
            BranchingProcess(Poisson(2.0), gt); n_initial = 2, max_cases = nothing,
            max_generations = 3, rng = StableRNG(4)
        )
        @test alone isa CountSimulationState
        @test !alone.extinct && alone.day < 100
        @test maximum(i.generation for i in individuals(alone)) <= 3
        # The same distribution as the individual simulation under the cap.
        for (offspring, G) in ((Poisson(1.4), 3), (NegativeBinomial(0.5, 0.3), 5))
            m = BranchingProcess(offspring, gt)
            on_counts = [
                simulate(
                    m; max_cases = 2000, max_generations = G, rng = StableRNG(i)
                ).cumulative_cases for i in 1:1000
            ]
            on_individuals = [
                simulate(
                    m; max_cases = 2000, stopping_rules = [MaxCases(2000), MaxGenerations(G)],
                    rng = StableRNG(10_000 + i)
                ).cumulative_cases for i in 1:1000
            ]
            @test abs(mean(on_counts) - mean(on_individuals)) <
                5 * hypot(std(on_counts), std(on_individuals)) / sqrt(1000)
            @test abs(mean(on_counts .== 1) - mean(on_individuals .== 1)) < 0.05
        end
        # Containment under a cap agrees with individuals. The default cap is applied.
        m = BranchingProcess(Poisson(1.3), gt)
        kw = (; max_cases = 200, n_initial = 1)
        a = containment_probability(m, 4000; kw..., max_generations = 4, rng = StableRNG(5))
        b = containment_probability(
            m, 4000; kw..., stopping_rules = [MaxCases(200), MaxGenerations(4)], rng = StableRNG(6)
        )
        @test abs(a.proportion - b.proportion) < 5 * hypot(a.mcse, b.mcse)
        @test a.proportion < containment_probability(
            m, 4000; kw..., max_generations = nothing, rng = StableRNG(7)
        ).proportion
        # Continuing keeps the cap.
        state = simulate(
            m; max_time = 3, max_cases = nothing, max_generations = 4, rng = StableRNG(8)
        )
        simulate(m, state; max_time = 12)
        @test maximum(i.generation for i in individuals(state)) <= 4
    end

    @testset "several windows agree with the individual simulation" begin
        gt2 = DiscreteNonParametric(2:4, [0.2, 0.5, 0.3])
        model = BranchingProcess(
            Infectiousness(NegBin(0.8, 0.5); kernel = gt, name = :a),
            Infectiousness(Poisson(0.6); kernel = gt2, name = :b)
        )
        rule = EpiBranch.count_rule(model)
        @test rule isa EpiBranch.CountRules
        @test size(rule.channels) == (2, 1)
        @test count_run(model; max_time = 5, rng = StableRNG(1)) isa CountSimulationState
        # Offspring numbers are correlated across the days of one infector
        # through the first window only.
        c = check_agreement(model, 8, 4, 3000; atol = 0.06)
        @test c[1, 2] > 0.25 && abs(c[1, 4]) < 0.05
        # Windows of different laws do not share infectiousness.
        state = count_run(model; max_time = 6, rng = StableRNG(2))
        @test size(state.cohort_infectiousness[end]) == (2, 1, 1)
        @test length(individuals(state)) == state.cumulative_cases
        @test nrow(linelist(state)) == state.cumulative_cases
        # A window that ends keeps the run on individuals.
        @test EpiBranch.count_rule(
            BranchingProcess(
                Infectiousness(Poisson(0.6); kernel = gt),
                Infectiousness(Poisson(0.6); kernel = gt, until = (:recovered,), name = :b)
            )
        ) === nothing
    end

    @testset "several types agree with the individual simulation" begin
        # Per-type daily counts, padded to `T + 1` days.
        function daily_by_type(state, T)
            counts = zeros(Int, 2, T + 1)
            for ind in individuals(state)
                get(ind.state, :infected, false) || continue
                counts[individual_type(ind), Int(ind.infection_time) + 1] += 1
            end
            return counts
        end
        M = [0.9 0.5; 0.4 0.8]
        T, n = 8, 1500
        for dist_fn in (R -> NegBin(R, 0.8), [NegBin(1.4, 0.5), Poisson(1.2)])
            model = BranchingProcess(M, dist_fn, gt)
            rule = EpiBranch.count_rule(model)
            @test rule isa EpiBranch.CountRules
            @test size(rule.channels) == (1, 2)
            @test rule.allocation[1] ≈ model.infectiousness[1].offspring.alloc_probs
            a = zeros(Int, n, 2, T + 1)
            b = zeros(Int, n, 2, T + 1)
            for i in 1:n
                kw = (; index_types = [1, 2, 2], max_time = T)
                x = simulate(
                    model; kw..., max_cases = nothing, max_generations = nothing,
                    rng = StableRNG(i)
                )
                @test x isa CountSimulationState
                a[i, :, :] = daily_by_type(x, T)
                b[i, :, :] = daily_by_type(
                    simulate(
                        model; kw..., stopping_rules = [MaxTime(Float64(T))],
                        rng = StableRNG(5000 + i)
                    ), T
                )
            end
            for type in 1:2, t in 2:(T + 1)
                se = sqrt(var(a[:, type, t]) / n + var(b[:, type, t]) / n)
                @test abs(mean(a[:, type, t]) - mean(b[:, type, t])) < 4.5 * se
                @test var(a[:, type, t]) ≈ var(b[:, type, t]) rtol = 0.3
            end
            # Counts of the two types are correlated through the shared parents.
            for t in 3:(T + 1)
                @test cor(a[:, 1, t], a[:, 2, t]) ≈ cor(b[:, 1, t], b[:, 2, t]) atol = 0.1
            end
        end
    end

    @testset "the people of a multi-type state have their types" begin
        M = [0.9 0.5; 0.4 0.8]
        model = BranchingProcess(M, R -> NegBin(R, 0.8), gt)
        state = simulate(
            model; index_types = [2, 1, 2], max_time = 10, max_cases = nothing,
            max_generations = nothing, rng = StableRNG(4)
        )
        people = individuals(state)
        @test length(people) == state.cumulative_cases
        @test sum(state.generation_counts[1]; dims = 2) == reshape([1, 2], 2, 1)
        @test individual_type.(people[1:3]) == [1, 2, 2]
        for d in 0:state.day, type in 1:2
            @test count(
                ind -> ind.infection_time == d && individual_type(ind) == type, people
            ) == sum(state.generation_counts[d + 1][type, :])
        end
        @test all(people) do ind
            ind.parent_id == 0 ||
                ind.infection_time - people[ind.parent_id].infection_time in 1:3
        end
        # Types are drawn when none are given, and from a distribution.
        drawn = simulate(
            model; n_initial = 200, max_time = 0, max_cases = nothing,
            max_generations = nothing, rng = StableRNG(1)
        )
        @test 60 < drawn.generation_counts[1][1, 1] < 140
        skewed = simulate(
            model; n_initial = 200, index_types = Categorical([0.9, 0.1]), max_time = 0,
            max_cases = nothing, max_generations = nothing, rng = StableRNG(1)
        )
        @test skewed.generation_counts[1][1, 1] > 150
        # A generation cap counts each type's generations.
        capped = simulate(
            model; index_types = [1, 2], max_time = 30, max_cases = nothing,
            max_generations = 3, rng = StableRNG(2)
        )
        @test maximum(i.generation for i in individuals(capped)) <= 3
        @test capped.counts == [sum(c) for c in capped.generation_counts]
        # A type that infects no one.
        sink = BranchingProcess([0.4 0.0; 0.5 0.0], R -> Poisson(R), gt)
        state = simulate(
            sink; index_types = [1, 1], max_time = 60, max_cases = nothing,
            max_generations = nothing, rng = StableRNG(3)
        )
        @test state isa CountSimulationState
        @test all(
            ind -> individual_type(ind) == 1 ||
                individual_type(individuals(state)[ind.parent_id]) == 1,
            individuals(state)
        )
        @test state.extinct
        # Each of several windows has its own matrix.
        two = BranchingProcess(
            (
                Infectiousness(EpiBranch.MultiTypeOffspring(M, R -> NegBin(R, 0.8)); kernel = gt, name = :a),
                Infectiousness(EpiBranch.MultiTypeOffspring(0.3 .* M, R -> Poisson(R)); kernel = gt, name = :b),
            ); n_types = 2
        )
        @test size(EpiBranch.count_rule(two).channels) == (2, 2)
        # A population with a size for each type stays on counts; a continuous
        # generation time does not.
        @test EpiBranch.count_rule(
            BranchingProcess(M, R -> NegBin(R, 0.8), gt; population = Population([100, 100]))
        ) !== nothing
        @test EpiBranch.count_rule(BranchingProcess(M, R -> NegBin(R, 0.8), Gamma(2.0, 1.0))) === nothing
    end

    @testset "several types and a finite population" begin
        M = [1.2 0.4; 0.3 0.9]
        T = 8
        by_type(state) = begin
            out = zeros(Int, T + 1, 2)
            for ind in individuals(state)
                get(ind.state, :infected, false) || continue
                out[Int(ind.infection_time) + 1, individual_type(ind)] += 1
            end
            out
        end
        # Mean daily cases by type of the count path and the individual
        # simulation, as a z-score of each cell.
        function max_z(model; n = 600, init)
            rng = StableRNG(3)
            counts = [
                by_type(
                    simulate(
                        model; init..., max_time = T, max_cases = nothing,
                        max_generations = nothing, rng
                    )
                ) for _ in 1:n
            ]
            people = [
                by_type(simulate(model; init..., stopping_rules = [MaxTime(Float64(T))], rng))
                    for _ in 1:n
            ]
            m1, m2 = mean(counts), mean(people)
            se = sqrt.((var(counts) .+ var(people)) ./ n) .+ 1.0e-9
            return maximum(abs.((m1 .- m2) ./ se))
        end
        init = (; index_types = [1, 1, 2])
        @test max_z(BranchingProcess(M, R -> NegBin(R, 0.8), gt); init) < 4
        pop = Population([60, 40])
        @test max_z(BranchingProcess(M, R -> Poisson(R), gt; population = pop); init) < 4
        @test max_z(
            BranchingProcess(1.8 .* M, R -> NegBin(R, 0.8), gt; population = Population([80, 50]));
            init
        ) < 4
        @test max_z(
            BranchingProcess(NegBin(2.0, 0.8), gt; population = Population(120));
            init = (; n_initial = 3)
        ) < 4

        # The people of a state have the types and the susceptibles of its counts.
        model = BranchingProcess(1.8 .* M, R -> NegBin(R, 0.8), gt; population = Population([80, 50]))
        state = simulate(
            model; index_types = [1, 1, 2], max_time = 12, max_cases = nothing,
            max_generations = nothing, rng = StableRNG(5)
        )
        @test state isa CountSimulationState
        cases(state) = [
            count(i -> individual_type(i) == j, individuals(state)) for j in 1:2
        ]
        @test cases(state) == [80, 50] .- state.susceptibles
        @test all(state.susceptibles .>= 0)
        @test sum(cases(state)) == state.cumulative_cases
        @test [individual_type(i) for i in individuals(state)[1:3]] == [1, 1, 2]
        @test all(
            i -> individual_type(i) in 1:2 && (i.parent_id == 0 || i.infection_time > individuals(state)[i.parent_id].infection_time),
            individuals(state)
        )
        # A continuation keeps depleting the same susceptibles.
        before = copy(state.susceptibles)
        copied = copy(state)
        simulate(model, state; max_time = 16)
        @test all(state.susceptibles .<= before)
        @test copied.susceptibles == before
        @test cases(state) == [80, 50] .- state.susceptibles

        # A type that infects no one, and a generation cap.
        sink = BranchingProcess([1.5 0.0; 0.8 0.0], R -> Poisson(R), gt)
        state = simulate(
            sink; index_types = [1, 1], max_time = 6, max_cases = nothing,
            max_generations = 3, rng = StableRNG(2)
        )
        @test state isa CountSimulationState
        @test all(i -> individual_type(i) == 1 || i.parent_id != 0, individuals(state))
        @test maximum(i.generation for i in individuals(state)) <= 3
        @test state.counts == sum.(state.generation_counts)
    end

    @testset "which runs take the count-level path" begin
        model = BranchingProcess(Poisson(1.2), gt)
        @test count_run(model; max_time = 5, rng = StableRNG(1)) isa CountSimulationState
        kw = (; max_time = 5, max_cases = nothing, max_generations = nothing, rng = StableRNG(1))
        # A generation limit or a stopping rule needs individuals.
        @test simulate(model; max_time = 5, rng = StableRNG(1)) isa SimulationState
        @test simulate(model; kw..., max_cases = 100) isa CountSimulationState
        @test simulate(model; kw..., max_time = nothing, max_cases = 100) isa CountSimulationState
        @test simulate(model; kw..., max_generations = 4) isa CountSimulationState
        @test simulate(model; kw..., stopping_rules = [MaxTime(5.0)]) isa SimulationState
        @test simulate(model; kw..., condition = 1:1000) isa SimulationState
        @test simulate(model; kw..., rng = CommonRandomNumbers(1)) isa SimulationState
        # A continuous generation time, a function-valued offspring, a
        # population, a window that ends and interventions are individual.
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2), Gamma(2.0, 1.0))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2), gt; population = Population(1000))) !== nothing
        M = [1.2 0.4; 0.3 0.9]
        two = BranchingProcess(M, R -> NegBin(R, 0.8), gt)
        @test EpiBranch.count_rule(two) isa EpiBranch.CountRules
        @test EpiBranch.count_rule(two).allocation[1] ≈ [0.8 0.4 / 1.3; 0.2 0.9 / 1.3] atol = 1.0e-12
        @test EpiBranch.count_rule(BranchingProcess(M, R -> Poisson(R), gt; population = Population([50, 30]))) !== nothing
        @test EpiBranch.count_rule(BranchingProcess(M, R -> Poisson(R), gt; population = Population(80))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(M, R -> Binomial(5, R / 5), gt)) === nothing
        @test simulate(two; kw..., max_time = 5, index_types = [1, 2]) isa CountSimulationState
        @test simulate(two; kw..., max_time = 5, index_types = DiscreteUniform(1, 2)) isa CountSimulationState
        @test EpiBranch.count_rule(BranchingProcess(Binomial(5, 0.3), gt)) isa EpiBranch.OffspringCountRule
        @test EpiBranch.count_rule(BranchingProcess(Binomial(5, 0.3), gt; population = Population(1000))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(DiscreteUniform(-1, 2), gt)) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2), DiscreteNonParametric(0:2, [0.2, 0.5, 0.3]))) ===
            nothing
        @test EpiBranch.count_rule(
            BranchingProcess(Infectiousness(Poisson(1.2); until = (:recovered,), kernel = gt))
        ) === nothing
        @test EpiBranch.count_rule(
            BranchingProcess(Infectiousness(RenewalOffspring(1.3, Gamma(4.0, 1.0); R_max = 1.3)))
        ) === nothing
        @test EpiBranch.count_rule(ModelSpec(model; interventions = [Isolation(delay = 2.0, duration = Inf)])) === nothing
        @test EpiBranch.count_rule(ModelSpec(model)) !== nothing
        @test EpiBranch.count_rule(model).g ≈ [0.3, 0.5, 0.2]
    end

    @testset "repeated runs and containment" begin
        model = BranchingProcess(NegBin(1.5, 0.5), gt)
        kw = (; n_initial = 2, max_time = 40, max_cases = nothing, max_generations = nothing)
        states = simulate(model, 20; kw..., rng = StableRNG(3))
        @test all(s -> s isa CountSimulationState, states)
        @test length(states) == 20
        again = simulate(model, 20; kw..., rng = StableRNG(3))
        @test [s.counts for s in states] == [s.counts for s in again]
        @test length(unique(s.counts for s in states)) > 1
        @test containment_probability(states).n_contained == count(s -> s.extinct, states)
        # `parallel` gives count states too.
        par = simulate(model, 20; kw..., rng = StableRNG(3), parallel = true)
        @test all(s -> s isa CountSimulationState, par)
        # A qualifying spec also gives count states, for runs and for containment.
        spec = ModelSpec(model)
        spec_states = simulate(spec, 5; kw..., rng = StableRNG(3))
        @test all(s -> s isa CountSimulationState, spec_states)
        @test [s.counts for s in spec_states] == [s.counts for s in states[1:5]]
        @test 0 <= containment_probability(spec, 50; kw..., rng = StableRNG(3)).proportion <= 1
        # A model that is not on the count path still gives individual runs.
        slow = BranchingProcess(NegBin(1.5, 0.5), Gamma(2.0, 1.0))
        @test simulate(slow, 3; kw..., rng = StableRNG(3)) isa Vector{<:SimulationState}

        n = 3000
        # A short horizon keeps the individual runs small for surviving lineages.
        by_counts = containment_probability(
            model, n; kw..., max_time = 15, rng = StableRNG(4)
        )
        by_people = containment_probability(
            model, n; n_initial = 2, stopping_rules = [MaxTime(15.0)], rng = StableRNG(5)
        )
        @test 0.1 < by_counts.proportion < 0.9
        @test abs(by_counts.proportion - by_people.proportion) <
            4 * hypot(by_counts.mcse, by_people.mcse)
    end
end
