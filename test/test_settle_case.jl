# `resolve_individual!` has one meaning on both engines: it runs once for each
# infection, after the case's clinical transitions, with the clock at the
# case's infection time and the case count up to and including it.

struct _RecordSettle <: AbstractIntervention end
function EpiBranch.resolve_individual!(::_RecordSettle, ind, state)
    push!(
        get!(ind.state, :_settle_calls, NamedTuple[]),
        (
            infected = is_infected(ind), clock = state.max_infection_time,
            count = state.cumulative_cases, generation = state.current_generation,
            outcome_set = haskey(ind.state, :outcome),
        )
    )
    return nothing
end

struct _RecordTransitionClock <: EpiBranch.AbstractClinicalTransition end
function EpiBranch.resolve_individual!(::_RecordTransitionClock, ind, state)
    ind.state[:_transition_clock] = state.max_infection_time
    ind.state[:_transition_count] = state.cumulative_cases
    return nothing
end

_settle_calls(ind) = get(ind.state, :_settle_calls, NamedTuple[])

# Each case's own settlement record, checked against its infection time and
# its place among the cases settled before it.
function _check_settled_once(state)
    cases = filter(is_infected, state.individuals)
    @test all(ind -> length(_settle_calls(ind)) == 1, cases)
    @test all(
        ind -> isempty(_settle_calls(ind)),
        filter(!is_infected, state.individuals)
    )
    calls = [only(_settle_calls(ind)) for ind in cases]
    @test all(c -> c.infected && c.outcome_set, calls)
    @test all(i -> calls[i].clock == cases[i].infection_time, eachindex(cases))
    @test all(i -> calls[i].generation == cases[i].generation, eachindex(cases))
    return nothing
end

# On the generation engine a case counts the earlier generations' cases and
# those of its own generation infected no later than it.
function _check_generation_counts(state)
    cases = filter(is_infected, state.individuals)
    for g in unique(c.generation for c in cases)
        earlier = count(c -> c.generation < g, cases)
        gen = sort(filter(c -> c.generation == g, cases); by = c -> c.infection_time)
        @test [only(_settle_calls(c)).count for c in gen] == earlier .+ (1:length(gen))
    end
    return nothing
end

@testset "resolve_individual! runs once per case on both engines" begin
    progression = [Transition(:recovered; delay = 1.0, terminal = true)]

    @testset "generation engine, including the last generation" begin
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        model = ModelSpec(bp; interventions = [_RecordSettle()], progression)
        for seed in 1:20
            state = simulate(model; rng = StableRNG(seed), max_generations = 3)
            _check_settled_once(state)
            _check_generation_counts(state)
        end
    end

    @testset "generation engine leaves uninfected ring members alone" begin
        ct = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.1), depth = 3,
            action = Quarantine(duration = 14.0)
        )
        iso = Isolation(onset_to_isolation_delay = Dirac(1.0), duration = Inf)
        model = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(5.0));
            interventions = [iso, ct, _RecordSettle()], progression,
            attributes = clinical_presentation(incubation_period = Exponential(3.0))
        )
        for seed in 1:10
            state = simulate(model; rng = StableRNG(seed), max_generations = 5)
            _check_settled_once(state)
            _check_generation_counts(state)
        end
    end

    @testset "continuous-time race" begin
        hp = HomogeneousProcess(; transmission_rate = 2.0, population_size = 50)
        model = ModelSpec(hp; interventions = [_RecordSettle()], progression)
        state = simulate(model; rng = StableRNG(2))
        _check_settled_once(state)
        cases = sort(filter(is_infected, state.individuals); by = c -> c.infection_time)
        @test [only(_settle_calls(c)).count for c in cases] == 1:length(cases)
    end

    @testset "transitions see the case's own clock" begin
        prog = [_RecordTransitionClock(), progression...]
        hp = HomogeneousProcess(; transmission_rate = 2.0, population_size = 50)
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        for (model, kw) in (
                (ModelSpec(hp; progression = prog), (;)),
                (ModelSpec(bp; progression = prog), (; max_generations = 3)),
            )
            state = simulate(model; rng = StableRNG(3), kw...)
            cases = filter(is_infected, state.individuals)
            @test length(cases) > 1
            @test all(c -> c.state[:_transition_clock] == c.infection_time, cases)
            @test all(c -> c.state[:_transition_count] >= 1, cases)
        end
    end
end

@testset "Scheduled start_time gates on the case's infection time" begin
    sch = Scheduled(
        Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf);
        start_time = 10.0
    )
    attributes = clinical_presentation(incubation_period = Dirac(0.5))
    bp = BranchingProcess(Poisson(1.5), Exponential(8.0))
    hp = HomogeneousProcess(; transmission_rate = 1.5, population_size = 200)
    progression = [Transition(:recovered; delay = Exponential(1.0), terminal = true)]
    for (model, kw) in (
            (ModelSpec(bp; interventions = [sch], attributes), (; max_generations = 6)),
            (ModelSpec(hp; interventions = [sch], attributes, progression), (;)),
        )
        before, after = 0, 0
        for seed in 1:100
            state = simulate(model; rng = StableRNG(seed), kw...)
            for ind in filter(is_infected, state.individuals)
                isolated = isfinite(get(ind.state, :isolation_time, Inf))
                if ind.infection_time < 10.0
                    before += isolated
                else
                    after += !isolated
                end
            end
        end
        @test before == 0
        @test after == 0
    end
end
