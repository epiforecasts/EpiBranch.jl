# A tiny observation model that only counts how often it ran, to check that
# a continued run never applies it (see "no observation on continuation"
# below).
struct CountingObservation <: EpiBranch.ObservationModel
    counter::Vector{Int}
end
CountingObservation() = CountingObservation([0])
function EpiBranch.apply_observation!(o::CountingObservation, state, rng)
    o.counter[1] += 1
    return state
end

@testset "simulate(model, state; max_time) continuation" begin
    @testset "continuing a closing window agrees bit for bit with one run" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(1))
        @test !state.extinct
        continued = simulate(model, state; max_time = 30.0)
        direct = simulate(model; max_time = 30.0, n_initial = 5, rng = StableRNG(1))
        @test continued === state
        @test continued.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(continued.individuals, direct.individuals)
        )
    end

    @testset "chained continuations agree with one run straight through" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        chained = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(7))
        chained = simulate(model, chained; max_time = 15.0)
        chained = simulate(model, chained; max_time = 30.0)
        direct = simulate(model; max_time = 30.0, n_initial = 5, rng = StableRNG(7))
        @test chained.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(chained.individuals, direct.individuals)
        )
    end

    @testset "a bare process continues the same way" begin
        # No progression at all, so (as for the never-closing window below)
        # mass action never finishes sampling a window in one call; agreement
        # is in distribution rather than bit for bit.
        process = HomogeneousProcess(transmission_rate = 0.4, population_size = 1500)
        final_size(seed; split) = begin
            if split === nothing
                simulate(process; max_time = 15.0, n_initial = 5, rng = StableRNG(seed)).cumulative_cases
            else
                state = simulate(process; max_time = split, n_initial = 5, rng = StableRNG(seed))
                simulate(process, state; max_time = 15.0).cumulative_cases
            end
        end
        direct = [final_size(s; split = nothing) for s in 1:150]
        continued = [final_size(s; split = 3.0) for s in 1:150]
        @test isapprox(mean(direct), mean(continued); rtol = 0.1)
    end

    @testset "mass action with a window that never closes agrees in distribution" begin
        # No `until` ever reached: the infectious window never closes, so
        # mass action samples contacts only up to the race's time limit and
        # must redraw the cut pairs on continuation.
        process = HomogeneousProcess(transmission_rate = 0.3, population_size = 1500)
        final_size(seed; split) = begin
            if split === nothing
                simulate(process; max_time = 20.0, n_initial = 5, rng = StableRNG(seed)).cumulative_cases
            else
                state = simulate(process; max_time = split, n_initial = 5, rng = StableRNG(seed))
                simulate(process, state; max_time = 20.0).cumulative_cases
            end
        end
        direct = [final_size(s; split = nothing) for s in 1:150]
        continued = [final_size(s; split = 4.0) for s in 1:150]
        @test isapprox(mean(direct), mean(continued); rtol = 0.1)
    end

    @testset "a branching process continues the same way" begin
        model = ModelSpec(BranchingProcess(NegBin(3.0, 0.15), Gamma(2.5, 2.0)))
        state = simulate(model; max_time = 5.0, max_cases = 10_000, rng = StableRNG(6))
        continued = simulate(model, state; max_time = 20.0)
        direct = simulate(model; max_time = 20.0, max_cases = 10_000, rng = StableRNG(6))
        @test length(continued.individuals) == length(direct.individuals)
        @test continued.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(continued.individuals, direct.individuals)
        )
    end

    @testset "continuing a state that already reached extinction is a no-op" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 20);
            progression = [Transition(:recovered; rate = 5.0, terminal = true)]
        )
        state = simulate(model; max_time = 50.0, rng = StableRNG(3))
        @test state.extinct
        continued = simulate(model, state; max_time = 100.0)
        @test continued === state
    end

    @testset "max_time must move past the state's own" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 500);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(4))
        @test !state.extinct
        @test_throws ArgumentError simulate(model, state; max_time = 5.0)
        @test_throws ArgumentError simulate(model, state; max_time = 2.0)
    end

    @testset "a continuation that changes may_block is refused" begin
        # The first call has no intervention that can block a proposal; the
        # second adds a leaky `Isolation`, which defines `competing_risk` and
        # so turns on the blocking that sizes the race's buffers. Resuming
        # with it must be refused rather than index buffers sized for the
        # first call.
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        blocking_model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            interventions = [
                Isolation(
                    onset_to_isolation_delay = 1.0, duration = Inf,
                    post_isolation_transmission = 0.5
                ),
            ]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(1))
        @test !state.extinct
        @test_throws ArgumentError simulate(blocking_model, state; max_time = 30.0)
    end

    @testset "no observation on continuation" begin
        counter = CountingObservation()
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 500);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            observation = counter
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(5))
        @test !state.extinct
        @test counter.counter[1] == 1
        simulate(model, state; max_time = 30.0)
        @test counter.counter[1] == 1
    end
end
