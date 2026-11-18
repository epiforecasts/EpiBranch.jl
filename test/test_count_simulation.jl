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

    function check_agreement(model, T, L, n)
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
        # Autocorrelation of daily counts at lag one.
        for t in 2:T
            @test cor(a[:, t], a[:, t + 1]) ≈ cor(b[:, t], b[:, t + 1]) atol = 0.05
        end
        # Correlation across days of one infector's offspring.
        @test cor(offspring[1]) ≈ cor(offspring[2]) atol = 0.03
        @test vec(mean(offspring[1]; dims = 1)) ≈ vec(mean(offspring[2]; dims = 1)) rtol = 0.06
        return cor(offspring[1])
    end

    @testset "agrees with the individual simulation" begin
        c = check_agreement(BranchingProcess(Poisson(1.3), gt), 8, 3, 2000)
        @test all(abs(c[i, j]) < 0.03 for i in 1:3, j in 1:3 if i != j)

        # A negative binomial offspring number is correlated across the days
        # of one infector.
        c = check_agreement(BranchingProcess(NegBin(1.3, 0.8), gt), 8, 3, 2000)
        @test c[1, 2] > 0.25

        R(t) = t < 5 ? 1.6 : 0.8
        renewal = BranchingProcess(
            Infectiousness(RenewalOffspring(R, [0.3, 0.5, 0.2]; k = 1.0))
        )
        c = check_agreement(renewal, 8, 3, 2000)
        @test c[1, 2] > 0.25
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
        @test_throws ArgumentError simulate(
            BranchingProcess(NegBin(1.4, 0.7), Gamma(2.0, 1.0)), state; max_time = 30
        )
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

    @testset "which runs take the count-level path" begin
        model = BranchingProcess(Poisson(1.2), gt)
        @test count_run(model; max_time = 5, rng = StableRNG(1)) isa CountSimulationState
        kw = (; max_time = 5, max_cases = nothing, max_generations = nothing, rng = StableRNG(1))
        # Anything that stops a run other than its time is decided per person.
        @test simulate(model; max_time = 5, rng = StableRNG(1)) isa SimulationState
        @test simulate(model; kw..., max_cases = 100) isa SimulationState
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
end
