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

@testset "Positional simulation options" begin
    rules = [MaxCases(10)]
    opts = SimOpts(1, rules)
    @test opts.n_initial == 1
    @test opts.initial_cases === nothing
    @test opts.stopping_rules == rules
    @test opts.stopping_rules isa Vector{AbstractStoppingRule}
end

@testset "Initial-case control validation and population mapping" begin
    ids = [3, 9]
    opts = SimOpts(; initial_cases = ids)
    @test opts.n_initial == 2
    push!(ids, 12)
    @test opts.initial_cases == [3, 9]
    @test_throws ArgumentError SimOpts(; initial_cases = [3, 3])
    @test_throws ArgumentError SimOpts(; initial_cases = [0])
    @test_throws ArgumentError SimOpts(; initial_cases = [-1])
    @test_throws ArgumentError SimOpts(; initial_cases = [3], n_initial = 1)
    @test EpiBranch._validate_initial_case_ids(opts, 9, 0.0) === nothing
    @test_throws ArgumentError EpiBranch._validate_initial_case_ids(opts, 8, 0.0)
    @test_throws ArgumentError EpiBranch._validate_initial_case_ids(opts, 9, 0.1)
    @test EpiBranch._validate_initial_case_ids(SimOpts(), 9, 0.1) === nothing
    @test_throws ArgumentError simulate(BranchingProcess(Poisson(0.0)); initial_cases = [3])
    # Race positions differ from population IDs, as they do within households.
    best = fill(Inf, 3)
    EpiBranch._seed_initial_cases!(best, [9, 3, 6], opts.initial_cases)
    @test best == [0.0, 0.0, Inf]
    empty_opts = SimOpts(; initial_cases = Int[], stopping_rules = [MaxCases(10)])
    @test empty_opts.n_initial == 0
    @test empty_opts.initial_cases == Int[]
end

# Many small races must reuse the population lookup without copying it.
function seed_household_population!(best, chosen, n_households)
    for household in 1:n_households
        fill!(best, Inf)
        EpiBranch._seed_initial_cases!(best, (2household - 1, 2household), chosen)
    end
    return nothing
end

@testset "Population seed lookup reuse" begin
    chosen = Set(1:2:60000)
    best = fill(Inf, 2)
    seed_household_population!(best, chosen, 1)
    allocated = @allocated seed_household_population!(best, chosen, 30000)
    @test best == [0.0, Inf]
    @test length(chosen) == 30000
    @test allocated < 1_000_000
end
