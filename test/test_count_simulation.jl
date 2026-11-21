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

    @testset "the members of a cohort keep their infectiousness" begin
        # With a mixed Poisson law the split of the infections of a cohort
        # among its members depends on the total, so the members are kept.
        model = BranchingProcess(PoissonLognormal(-0.2, 0.9), gt)
        state = count_run(model; max_time = 6, rng = StableRNG(4))
        @test length(state.cohort_members) == state.day + 1
        @test all(
            length(m) == n && sum(m) ≈ total for
                (m, n, total) in zip(state.cohort_members, state.counts, state.cohort_infectiousness)
        )
        @test length(individuals(state)) == state.cumulative_cases
        negbin = count_run(BranchingProcess(NegBin(1.3, 0.8), gt); max_time = 6, rng = StableRNG(4))
        @test all(isempty, negbin.cohort_members)
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

    @testset "containment_probability on counts" begin
        # The same event as on individuals: extinct with fewer than `max_cases` cases.
        for offspring in (Poisson(1.3), NegativeBinomial(0.5, 0.3))
            model = BranchingProcess(offspring, gt)
            kw = (; max_cases = 60, n_initial = 1)
            on_counts = containment_probability(
                model, 4000; kw..., max_generations = nothing, rng = StableRNG(5)
            )
            on_individuals = containment_probability(
                model, 4000; kw..., max_generations = 1000, rng = StableRNG(6)
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

    @testset "which runs take the count-level path" begin
        model = BranchingProcess(Poisson(1.2), gt)
        @test count_run(model; max_time = 5, rng = StableRNG(1)) isa CountSimulationState
        kw = (; max_time = 5, max_cases = nothing, max_generations = nothing, rng = StableRNG(1))
        # A generation limit or a stopping rule needs individuals.
        @test simulate(model; max_time = 5, rng = StableRNG(1)) isa SimulationState
        @test simulate(model; kw..., max_cases = 100) isa CountSimulationState
        @test simulate(model; kw..., max_time = nothing, max_cases = 100) isa CountSimulationState
        @test simulate(model; kw..., max_generations = 4) isa SimulationState
        @test simulate(model; kw..., stopping_rules = [MaxTime(5.0)]) isa SimulationState
        @test simulate(model; kw..., condition = 1:1000) isa SimulationState
        @test simulate(model; kw..., rng = CommonRandomNumbers(1)) isa SimulationState
        # A continuous generation time, a function-valued offspring, a
        # population, a window that ends and interventions are individual.
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2), Gamma(2.0, 1.0))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Poisson(1.2), gt; population = Population(1000))) === nothing
        @test EpiBranch.count_rule(BranchingProcess(Binomial(5, 0.3), gt)) === nothing
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
        # A short horizon keeps the individual runs of the surviving lineages small.
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
