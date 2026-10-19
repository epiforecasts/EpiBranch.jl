# Custom stopping rule used by the "user extension" test below. Defined
# at module scope because struct definitions can't live inside @testset.
struct MaxChainLengthRule <: AbstractStoppingRule
    n::Int
end
function EpiBranch.should_stop(r::MaxChainLengthRule, state::SimulationState)
    isempty(state.individuals) && return false
    return maximum(ind.generation for ind in state.individuals) >= r.n
end

# A user-defined time-bounding rule, used below to check that `time_bound`
# lets a continuous-time (Sellke) run honour it without core naming the type.
struct CustomTimeBoundRule <: AbstractStoppingRule
    t::Float64
end
EpiBranch.should_stop(r::CustomTimeBoundRule, state::SimulationState) = state.max_infection_time >= r.t
EpiBranch.time_bound(r::CustomTimeBoundRule) = r.t
EpiBranch.honoured_without_should_stop(::CustomTimeBoundRule) = true

# A rule that bounds time and tests a case count as well: a run that ends at
# the bound applies half of it, so it keeps the default and is reported.
struct TimeOrCasesRule <: AbstractStoppingRule
    t::Float64
    n::Int
end
function EpiBranch.should_stop(r::TimeOrCasesRule, state::SimulationState)
    return state.max_infection_time >= r.t || state.cumulative_cases >= r.n
end
EpiBranch.time_bound(r::TimeOrCasesRule) = r.t

# The built-in models all run on the continuous-time race, which enforces
# `MaxTime` by processing cases in time order; this model runs on the
# generation engine instead (see `generate_offspring`), used below to check
# that engine's own per-branch enforcement.
struct _OffspringTree <: EpiBranch.TransmissionModel
    offspring::Distribution
    generation_time::Gamma{Float64}
end
EpiBranch.population_size(::_OffspringTree) = EpiBranch.NoPopulation()
EpiBranch.generate_offspring(m::_OffspringTree, parent, state) = rand(state.rng, m.offspring)

@testset "Stopping rules" begin
    @testset "MaxCases stops once cumulative cases reach the cap" begin
        # The engine processes a full generation per step, so cumulative
        # cases can overshoot the cap by one generation's worth of new
        # infections — we check the rule stopped the run, not that the count is
        # tight against the cap.
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(
            model;
            stopping_rules = [Extinction(), MaxCases(50)],
            rng = rng
        )
        @test state.cumulative_cases >= 50
        @test !state.extinct  # extinction would mean the cap rule wasn't what stopped us
    end

    @testset "MaxGenerations caps generation depth" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(
            model;
            stopping_rules = [Extinction(), MaxGenerations(3)],
            rng = rng
        )
        @test state.current_generation <= 3
    end

    @testset "MaxTime keeps only infections up to the cap" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(
            model;
            stopping_rules = [Extinction(), MaxTime(20.0)],
            rng = rng
        )
        @test 15.0 < state.max_infection_time <= 20.0
        @test all(ind -> ind.infection_time <= 20.0, filter(is_infected, state.individuals))
        @test !state.extinct
    end

    @testset "MaxTime keeps cases before the limit despite a spread-out generation time" begin
        # A later generation can hold infections well before the limit even
        # once an earlier one has crossed it, so stopping the whole run at the
        # first case past the limit (the bug) loses most of them. We compare
        # case counts up to the limit between a run stopped at it and one left
        # to run on (capped on cases instead, well above the few dozen this
        # mildly supercritical setting reaches); the two should land close.
        # Under the bug, the time-limited run came out at under half the
        # other.
        gt = Gamma(2.0, 2.5)
        model = BranchingProcess(NegBin(1.5, 2.0), gt)
        by_day30(s) = count(i -> is_infected(i) && i.infection_time <= 30, s.individuals)
        n = 150
        with_limit = sum(
            by_day30(
                simulate(
                    model; max_time = 30.0, max_cases = nothing,
                    max_generations = nothing, rng = StableRNG(s)
                )
            )
                for s in 1:n
        ) / n
        without = sum(
            by_day30(
                simulate(
                    model; max_cases = 20_000, max_generations = nothing,
                    rng = StableRNG(s)
                )
            )
                for s in 1:n
        ) / n
        @test with_limit > 0.7 * without
    end

    @testset "MaxTime stops each case's own branch on the generation engine" begin
        model = _OffspringTree(NegBin(1.5, 2.0), Gamma(2.0, 2.5))
        state = simulate(
            model; max_time = 30.0, max_cases = nothing, max_generations = nothing,
            rng = StableRNG(8)
        )
        infected = filter(is_infected, state.individuals)
        @test all(
            case -> case.infection_time < 30.0 || isempty(case.secondary_case_ids),
            infected
        )
        @test !state.extinct
    end

    @testset "A user-defined time-bounding rule keeps cases before the limit on the generation engine too" begin
        # The generation engine must give `CustomTimeBoundRule` the same
        # per-branch treatment as `MaxTime`: it is written exactly as the
        # Extending EpiBranch guide's "Stopping rules" section tells a user
        # to, with `should_stop` plus `time_bound` and
        # `honoured_without_should_stop`. Dispatching on the literal
        # `MaxTime` type instead of that trait would end the whole run at
        # the first case past the bound. That would lose cases infected
        # before the bound but in a later generation, the same bug `MaxTime`
        # itself was fixed for. The per-case count comparison below, as in
        # the `MaxTime` test above, would catch it.
        model = _OffspringTree(NegBin(1.5, 2.0), Gamma(2.0, 2.5))
        by_day30(s) = count(i -> is_infected(i) && i.infection_time <= 30, s.individuals)
        n = 150
        with_limit = sum(
            by_day30(
                simulate(
                    model;
                    stopping_rules = [Extinction(), CustomTimeBoundRule(30.0)],
                    rng = StableRNG(s)
                )
            )
                for s in 1:n
        ) / n
        without = sum(
            by_day30(
                simulate(
                    model; stopping_rules = [Extinction(), MaxCases(20_000)],
                    rng = StableRNG(s)
                )
            )
                for s in 1:n
        ) / n
        @test with_limit > 0.7 * without
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
        state = simulate(
            model;
            stopping_rules = [Extinction(), MaxChainLengthRule(3)],
            rng = rng
        )
        @test maximum(ind.generation for ind in state.individuals) <= 3
    end

    @testset "time_bound lets a custom rule end a Sellke run" begin
        # `_max_time` (consulted by the structure-driven Sellke loops) reads
        # `time_bound` on every rule rather than matching `MaxTime` by type, so
        # a user's own time-bounding rule ends such a run exactly as `MaxTime`
        # does, with no warning that it was ignored.
        @test EpiBranch.time_bound(MaxCases(10)) == Inf
        @test EpiBranch.time_bound(MaxTime(5.0)) == 5.0
        @test EpiBranch.time_bound(CustomTimeBoundRule(5.0)) == 5.0

        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog
        )
        full = simulate(spec; n_initial = 3, rng = StableRNG(4))
        cut = @test_logs simulate(
            spec; n_initial = 3,
            stopping_rules = [Extinction(), CustomTimeBoundRule(2.0)], rng = StableRNG(4)
        )
        @test count(is_infected, cut.individuals) < count(is_infected, full.individuals)
        @test !is_extinct(cut)
    end

    @testset "a rule not honoured by time or generation alone is polled via should_stop" begin
        @test !EpiBranch.honoured_without_should_stop(TimeOrCasesRule(2.0, 5))
        @test EpiBranch.honoured_without_should_stop(Extinction())
        @test EpiBranch.honoured_without_should_stop(MaxTime(5.0))
        @test EpiBranch.honoured_without_should_stop(MaxGenerations(5))
        @test EpiBranch.honoured_without_should_stop(CustomTimeBoundRule(5.0))
        @test !EpiBranch.honoured_without_should_stop(MaxCases(10))

        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog
        )
        full = simulate(spec; n_initial = 3, rng = StableRNG(4))
        # The time bound applies as before, and the race now also polls
        # `should_stop` for the case count, so the rule is fully honoured
        # rather than only half of it.
        cut = @test_logs simulate(
            spec; n_initial = 3,
            stopping_rules = [Extinction(), TimeOrCasesRule(2.0, 5)], rng = StableRNG(4)
        )
        @test count(is_infected, cut.individuals) < count(is_infected, full.individuals)
        @test !is_extinct(cut)
        # An explicit `MaxCases`, with no time bound at all, now stops the
        # race exactly at the cap instead of being ignored.
        capped = @test_logs simulate(
            spec; n_initial = 3,
            stopping_rules = [Extinction(), MaxCases(5)], rng = StableRNG(4)
        )
        @test capped.cumulative_cases == 5
        @test !is_extinct(capped)
    end

    @testset "generation_bound drops a case's own proposals past the cap" begin
        @test EpiBranch.generation_bound(MaxCases(10)) == typemax(Int)
        @test EpiBranch.generation_bound(MaxGenerations(3)) == 3

        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 3.0, population_size = 500);
            progression = prog
        )
        capped = simulate(
            spec; n_initial = 3,
            stopping_rules = [Extinction(), MaxGenerations(3)], rng = StableRNG(4)
        )
        @test maximum(ind.generation for ind in capped.individuals if is_infected(ind)) <= 3
        # Capping depth is not the same as stopping the race the instant that
        # depth is seen: cases below the cap, settling later in time than a
        # capped one, still arise and are counted.
        @test capped.cumulative_cases > 4
        @test !is_extinct(capped)
    end

    @testset "the max_cases/max_generations defaults stay off where the race does not honour them" begin
        homog = HomogeneousProcess(; transmission_rate = 2.0, population_size = 50)
        default_opts = EpiBranch._build_sim_opts(
            homog, nothing, nothing, EpiBranch._DEFAULT_MAX_CASES,
            EpiBranch._DEFAULT_MAX_GENERATIONS, nothing, nothing
        )
        @test !any(r -> r isa MaxCases, default_opts.stopping_rules)
        @test !any(r -> r isa MaxGenerations, default_opts.stopping_rules)

        # A non-default value still gets through, as does `stopping_rules`
        # given directly.
        explicit_opts = EpiBranch._build_sim_opts(
            homog, nothing, nothing, 50, 10, nothing, nothing
        )
        @test any(r -> r isa MaxCases && r.n == 50, explicit_opts.stopping_rules)
        @test any(r -> r isa MaxGenerations && r.n == 10, explicit_opts.stopping_rules)
        given_opts = EpiBranch._build_sim_opts(
            homog, nothing, nothing, EpiBranch._DEFAULT_MAX_CASES,
            EpiBranch._DEFAULT_MAX_GENERATIONS, nothing, [MaxCases(10_000)]
        )
        @test any(r -> r isa MaxCases, given_opts.stopping_rules)

        # The generation-based engine is untouched either way.
        branching = BranchingProcess(Poisson(1.5), Exponential(2.0))
        engine_opts = EpiBranch._build_sim_opts(
            branching, nothing, nothing, EpiBranch._DEFAULT_MAX_CASES,
            EpiBranch._DEFAULT_MAX_GENERATIONS, nothing, nothing
        )
        @test any(r -> r isa MaxCases, engine_opts.stopping_rules)
        @test any(r -> r isa MaxGenerations, engine_opts.stopping_rules)
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
        state = simulate(
            model; stopping_rules = [MaxCases(1_000_000)],
            rng = StableRNG(1)
        )
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
    @test EpiBranch._validate_initial_case_ids(opts, 9) === nothing
    @test_throws ArgumentError EpiBranch._validate_initial_case_ids(opts, 8)
    @test EpiBranch._validate_initial_case_ids(SimOpts(), 9) === nothing
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
