# Custom stopping rule used by the "user extension" test below. Defined
# at module scope because struct definitions can't live inside @testset.
struct MaxChainLengthRule <: AbstractStoppingRule
    n::Int
end
function EpiBranch.should_stop(r::MaxChainLengthRule, state::SimulationState)
    isempty(state.individuals) && return false
    return maximum(ind.generation for ind in state.individuals) >= r.n
end

@testset "Stopping rules" begin
    @testset "MaxCases stops once cumulative cases reach the cap" begin
        # The engine processes a full generation per step, so cumulative
        # cases can overshoot the cap by one generation's worth of new
        # infections — we check the rule fired, not that the count is
        # tight against the cap.
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model;
            stopping_rules = [Extinction(), MaxCases(50)],
            rng = rng)
        @test state.cumulative_cases >= 50
        @test !state.extinct  # extinction would mean the cap rule wasn't what stopped us
    end

    @testset "MaxGenerations caps generation depth" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model;
            stopping_rules = [Extinction(), MaxGenerations(3)],
            rng = rng)
        @test state.current_generation <= 3
    end

    @testset "MaxTime stops once max_infection_time crosses the cap" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model;
            stopping_rules = [Extinction(), MaxTime(20.0)],
            rng = rng)
        @test state.max_infection_time >= 20.0
        @test !state.extinct
    end

    @testset "Ergonomic kwargs build rules" begin
        opts = SimOpts(max_cases = 100, max_generations = 5)
        @test any(r isa MaxCases && r.n == 100 for r in opts.stopping_rules)
        @test any(r isa MaxGenerations && r.n == 5 for r in opts.stopping_rules)
        @test any(r isa Extinction for r in opts.stopping_rules)
    end

    @testset "Custom user-defined stopping rule" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        # Cap chain depth to 3 generations via a user rule.
        state = simulate(model;
            stopping_rules = [Extinction(), MaxChainLengthRule(3)],
            rng = rng)
        @test maximum(ind.generation for ind in state.individuals) <= 3
    end

    @testset "Extinction is prepended to user-supplied rules" begin
        # Per its docstring, Extinction is always included unless the user
        # supplies their own — otherwise a rule set with no extinction check
        # loops forever on an outbreak that goes extinct below the cap.
        opts = SimOpts(stopping_rules = [MaxCases(1000)])
        @test any(r isa Extinction for r in opts.stopping_rules)

        # Not doubled when the user includes one.
        opts2 = SimOpts(stopping_rules = [Extinction(), MaxCases(1000)])
        @test count(r -> r isa Extinction, opts2.stopping_rules) == 1

        # The caller's vector is left untouched.
        user = AbstractStoppingRule[MaxCases(1000)]
        SimOpts(stopping_rules = user)
        @test length(user) == 1

        # A subcritical outbreak with only a (never-reached) MaxCases rule still
        # terminates on extinction rather than hanging.
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; stopping_rules = [MaxCases(1_000_000)],
            rng = StableRNG(1))
        @test is_extinct(state)
    end
end
