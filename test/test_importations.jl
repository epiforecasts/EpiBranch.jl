using Distributions
using DataFrames
using StableRNGs

@testset "BranchingProcess importations" begin
    times = [3.0, 10.0, 25.0]

    @testset "Imports start at their own times, as external seeds" begin
        # Offspring of zero: only the index case and the imports are infected.
        model = BranchingProcess(Poisson(0.0), Exponential(5.0); importations = times)
        state = simulate(model; rng = StableRNG(1))
        inds = individuals(state)
        @test length(inds) == 4
        @test [i.infection_time for i in inds] == [0.0, times...]
        @test all(i -> i.state[:index], inds)
        @test [get(i.state, :infection_route, nothing) for i in inds] ==
            [nothing, :external, :external, :external]
        @test state.cumulative_cases == 4
    end

    @testset "Imports add to n_initial" begin
        model = BranchingProcess(Poisson(0.0), Exponential(5.0); importations = times)
        state = simulate(model; n_initial = 2, rng = StableRNG(1))
        @test [i.infection_time for i in individuals(state)] == [0.0, 0.0, times...]
    end

    @testset "Offspring of an import come after it" begin
        model = BranchingProcess(Poisson(2.0), Exponential(5.0); importations = [50.0])
        state = simulate(model; max_cases = 200, rng = StableRNG(2))
        imported = filter(i -> get(i.state, :infection_route, nothing) === :external, individuals(state))
        @test length(imported) == 1
        @test all(i -> i.infection_time >= 50.0, individuals(state)[imported[1].secondary_case_ids])
    end

    @testset "Imports share capacity limits and the population" begin
        # Only the import has offspring. It arrives after the index case has
        # used up a population of two, so none of its offspring can be infected.
        model = BranchingProcess(
            (rng, ind) -> ind.infection_time > 0 ? 10 : 0, Exponential(5.0);
            importations = [1.0], population_size = 2
        )
        state = simulate(model; rng = StableRNG(3))
        @test state.cumulative_cases == 2
        capped = simulate(
            BranchingProcess(Poisson(0.0), Exponential(5.0); importations = times);
            max_cases = 2, rng = StableRNG(3)
        )
        @test capped.cumulative_cases == 2
    end

    @testset "An import after max_time stays uninfected" begin
        model = BranchingProcess(Poisson(0.0), Exponential(5.0); importations = times)
        state = simulate(model; max_time = 12.0, rng = StableRNG(4))
        @test count(is_infected, individuals(state)) == 3
    end

    @testset "Invalid importations are refused" begin
        @test_throws ArgumentError BranchingProcess(Poisson(1.0); importations = [-1.0])
        @test_throws ArgumentError BranchingProcess(Poisson(1.0); importations = "daily")
    end

    @testset "Imports in a multi-type model draw their type uniformly" begin
        model = BranchingProcess(
            (rng, ind) -> zeros(Int, 2), Exponential(5.0);
            n_types = 2, importations = fill(1.0, 200)
        )
        state = simulate(model; index_types = [1], rng = StableRNG(5))
        inds = individuals(state)
        @test individual_type(inds[1]) == 1
        n2 = count(i -> individual_type(i) == 2, inds[2:end])
        @test 60 < n2 < 140
    end
end

@testset "BranchingProcess importations at a rate" begin
    quiet(imp) = BranchingProcess(Poisson(0.0), Exponential(5.0); importations = imp)
    imported(state) = [i.infection_time for i in individuals(state)[2:end]]

    @testset "A constant rate gives Poisson arrivals within max_time" begin
        counts = [
            length(imported(simulate(quiet(0.5); max_time = 40.0, rng = StableRNG(s))))
                for s in 1:300
        ]
        @test isapprox(sum(counts) / 300, 20.0; atol = 0.7)
        state = simulate(quiet(0.5); max_time = 40.0, rng = StableRNG(1))
        @test all(i -> get(i.state, :infection_route, nothing) === :external, individuals(state)[2:end])
        @test all(<(40.0), imported(state))
    end

    @testset "A schedule or function of time sets the rate on each day" begin
        steps = Steps([10.0], [0.0, 2.0])
        times = reduce(
            vcat,
            [imported(simulate(quiet(steps); max_time = 20.0, rng = StableRNG(s))) for s in 1:100]
        )
        @test all(>=(10.0), times)
        @test isapprox(length(times) / 100, 20.0; atol = 1.5)
        fn = t -> t < 10 ? 0.0 : 2.0
        ftimes = reduce(
            vcat,
            [imported(simulate(quiet(fn); max_time = 20.0, rng = StableRNG(s))) for s in 1:100]
        )
        @test all(>=(10.0), ftimes)
        @test isapprox(length(ftimes) / 100, 20.0; atol = 1.5)
        seasonal = Seasonal(amplitude = 1.0, peak_day = 0.0, period = 20.0)
        scaled = t -> 3 * EpiBranch.calendar_multiplier(seasonal, t)
        stimes = reduce(
            vcat,
            [imported(simulate(quiet(scaled); max_time = 20.0, rng = StableRNG(s))) for s in 1:200]
        )
        # The mean rate is three a day. Few arrive near the trough on day 10.
        @test isapprox(length(stimes) / 200, 60.0; atol = 3.0)
        @test count(t -> 8 < t < 12, stimes) < count(t -> t < 2 || t > 18, stimes) / 2
    end

    @testset "Rate-based imports share capacity limits" begin
        state = simulate(quiet(5.0); max_cases = 10, max_time = 100.0, rng = StableRNG(2))
        @test state.cumulative_cases == 10
    end

    @testset "A rate needs max_time, and invalid rates are refused" begin
        @test_throws ArgumentError simulate(quiet(0.5); rng = StableRNG(1))
        @test_throws ArgumentError quiet(-1.0)
        @test_throws ArgumentError quiet("daily")
    end

    @testset "A function rate above its grid bound is refused, not under-drawn" begin
        pulse = t -> 50.02 <= t < 50.08 ? 50.0 : 1.0
        threw = [
            try
                simulate(quiet(pulse); max_time = 100.0, rng = StableRNG(s))
                false
            catch e
                e isa ArgumentError
            end
                for s in 1:300
        ]
        @test any(threw)
    end
end

@testset "A count-eligible model with importations runs as individuals" begin
    gt = DiscreteNonParametric([1.0, 2.0], [0.5, 0.5])
    plain = BranchingProcess(Poisson(0.0), gt)
    @test EpiBranch.count_rule(plain) !== nothing
    model = BranchingProcess(Poisson(0.0), gt; importations = [7.0])
    @test EpiBranch.count_rule(model) === nothing
    state = simulate(model; max_time = 20.0, rng = StableRNG(6))
    @test [i.infection_time for i in individuals(state)] == [0.0, 7.0]
end

@testset "Entry points that cannot apply importations refuse them" begin
    model = BranchingProcess(Poisson(1.0); importations = [7.0])
    @test_throws ArgumentError simulate_by_generation(model; rng = StableRNG(7))
    @test_throws ArgumentError simulate_by_generation(
        ModelSpec(model); rng = StableRNG(7)
    )
    cases = DataFrame(
        id = [1, 2], parent_id = [missing, 1], infection_time = [0.0, 2.0]
    )
    obs_model = BranchingProcess(Poisson(1.0), Gamma(2.0, 2.0); importations = [7.0])
    @test_throws ArgumentError state_from_cases(obs_model, cases; rng = StableRNG(7))
end
