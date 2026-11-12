using Distributions
using StableRNGs
import SSMProblems

@testset "outbreak_state_space (SSMProblems)" begin
    model = ModelSpec(
        HomogeneousProcess(transmission_rate = 0.4, population_size = 500);
        progression = [Transition(:recovered; rate = 0.2, terminal = true)]
    )
    new_cases = TransitionCount(:infection)
    observe(t, state) = Poisson(new_cases(state, (t - 1.0, Float64(t))) + 0.1)
    ssm = outbreak_state_space(model, observe; n_initial = 5)

    @testset "returns a state-space model over simulation states" begin
        @test ssm isa SSMProblems.StateSpaceModel
        x0 = SSMProblems.simulate(StableRNG(1), SSMProblems.prior(ssm))
        @test x0 isa SimulationState
        @test x0.cumulative_cases == 5
        @test x0.continuation.max_time == 0.0
    end

    @testset "a chain of steps agrees with one run straight through" begin
        x0, xs, ys = SSMProblems.simulate(StableRNG(5), ssm, 6)
        direct = EpiBranch.simulate(model; max_time = 6.0, n_initial = 5, rng = StableRNG(5))
        @test length(xs) == length(ys) == 6
        @test xs[end].cumulative_cases == direct.cumulative_cases
        @test issorted(x.cumulative_cases for x in xs)
    end

    @testset "a step leaves the previous state untouched" begin
        rng = StableRNG(2)
        x0 = SSMProblems.simulate(rng, SSMProblems.prior(ssm))
        before = x0.cumulative_cases
        x1 = SSMProblems.simulate(rng, SSMProblems.dyn(ssm), 1, x0)
        x1b = SSMProblems.simulate(rng, SSMProblems.dyn(ssm), 1, x0)
        @test x0.cumulative_cases == before
        @test x1 !== x0 && x1b !== x1
    end

    @testset "step length scales time" begin
        slow = outbreak_state_space(model, observe; dt = 2.5, n_initial = 5)
        x0 = SSMProblems.simulate(StableRNG(3), SSMProblems.prior(slow))
        x2 = SSMProblems.simulate(StableRNG(3), SSMProblems.dyn(slow), 2, x0)
        @test x2.continuation.max_time == 5.0
    end

    @testset "a bare process is accepted" begin
        process = HomogeneousProcess(transmission_rate = 0.4, population_size = 500)
        bare = outbreak_state_space(process, observe; n_initial = 5)
        x0 = SSMProblems.simulate(StableRNG(1), SSMProblems.prior(bare))
        @test x0 isa SimulationState
    end

    @testset "observation is the user's distribution" begin
        x0 = SSMProblems.simulate(StableRNG(1), SSMProblems.prior(ssm))
        x1 = SSMProblems.simulate(StableRNG(1), SSMProblems.dyn(ssm), 1, x0)
        @test SSMProblems.logdensity(SSMProblems.obs(ssm), 1, x1, 3) ==
            logpdf(observe(1, x1), 3)
    end

    @testset "arguments the filter or the steps own are refused" begin
        @test_throws ArgumentError outbreak_state_space(model, observe; dt = 0.0)
        @test_throws ArgumentError outbreak_state_space(model, observe; rng = StableRNG(1))
        @test_throws ArgumentError outbreak_state_space(model, observe; max_time = 5.0)
    end
end
