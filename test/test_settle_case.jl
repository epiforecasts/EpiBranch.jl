# `resolve_individual!` runs once for each infection, after the case's clinical transitions, with the clock at the
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

# A case counts every case infected no later than it, whatever its generation.
function _check_time_order_counts(state)
    cases = sort(filter(is_infected, state.individuals); by = c -> c.infection_time)
    @test [only(_settle_calls(c)).count for c in cases] == 1:length(cases)
    return nothing
end

@testset "resolve_individual! runs once per case" begin
    progression = [Transition(:recovered; delay = 1.0, terminal = true)]

    @testset "branching process, including the last generation" begin
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        model = ModelSpec(bp; interventions = [_RecordSettle()], progression)
        for seed in 1:20
            state = simulate(model; rng = StableRNG(seed), max_generations = 3)
            _check_settled_once(state)
            _check_time_order_counts(state)
        end
    end

    @testset "branching process leaves uninfected ring members alone" begin
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
            _check_time_order_counts(state)
        end
    end

    @testset "continuous-time race" begin
        hp = HomogeneousProcess(; transmission_rate = 2.0, population_size = 50)
        model = ModelSpec(hp; interventions = [_RecordSettle()], progression)
        state = simulate(model; rng = StableRNG(2))
        _check_settled_once(state)
        _check_time_order_counts(state)
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

@testset "Scheduled start_time gates on the isolation's own time, not the case's infection time" begin
    sch = Scheduled(
        Isolation(onset_to_isolation_delay = Dirac(0.0), duration = Inf);
        start_time = 10.0
    )
    attributes = clinical_presentation(incubation_period = Dirac(0.5))
    bp = BranchingProcess(Poisson(1.5), Exponential(8.0))
    hp = HomogeneousProcess(; transmission_rate = 1.5, population_size = 200)
    progression = [Transition(:recovered; delay = Exponential(1.0), terminal = true)]
    reached_before_infection = false
    for (model, kw) in (
            (ModelSpec(bp; interventions = [sch], attributes), (; max_generations = 6)),
            (ModelSpec(hp; interventions = [sch], attributes, progression), (;)),
        )
        for seed in 1:100
            state = simulate(model; rng = StableRNG(seed), kw...)
            for ind in filter(is_infected, state.individuals)
                isolated = isfinite(get(ind.state, :isolation_time, Inf))
                if isolated
                    @test ind.state[:isolation_time] >= 10.0
                    ind.infection_time < 10.0 && (reached_before_infection = true)
                elseif !is_asymptomatic(ind)
                    # Eligible but not isolated: its own isolation, half a
                    # day after infection, must itself fall before the start.
                    @test ind.infection_time + 0.5 < 10.0
                end
            end
        end
    end
    # A case infected just before day 10 is still isolated, once its own
    # isolation, half a day later, falls inside the window.
    @test reached_before_infection
end
