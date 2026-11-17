using Distributions
using StableRNGs

# Triggers the doses of a group at a fixed calendar day.
struct _PerGroupEligibleAt <: EpiBranch.TraceEligibility
    time::Float64
end
EpiBranch.trigger_time(e::_PerGroupEligibleAt, infector, state) = e.time

const _PG_MATRIX = [1.2 0.3; 0.3 1.2]
const _PG_NAMES = ["under5", "over5"]

_pg_gv(key; eligibility = _PerGroupEligibleAt(30.0)) = GroupVaccination(;
    efficacy = 0.9, group_key = key, eligibility
)

# Individual-based simulation over a population of types.
function _pg_types(budget; names = _PG_NAMES, kwargs...)
    spec = ModelSpec(
        BranchingProcess(
            _PG_MATRIX, R -> Poisson(R), Dirac(14.0);
            population = Population([30_000, 30_000]; groups = names)
        );
        attributes = clinical_presentation(
            incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
        ),
        interventions = [CapacityConstrained(_pg_gv(:type); budget_per_period = budget, kwargs...)]
    )
    return simulate(
        spec; index_types = [fill(1, 120); fill(2, 120)], max_generations = 3,
        rng = StableRNG(1), max_cases = 10^5
    )
end

_pg_vaccinated(state, type) = count(
    ind -> is_vaccinated(ind) && individual_type(ind) == type, state.individuals
)

# A population of mixing groups, run by the individual-based simulation.
function _pg_mixing_spec(ivs; n = 2_000, names = _PG_NAMES)
    clinical = clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0)
    return ModelSpec(
        MixingProcess(;
            population = Population([n, n]; groups = names), mixing_by = (:age,),
            mixing = FrequencyDependent([0.5 0.1; 0.1 0.5])
        );
        progression = [Transition(:recovered; rate = 0.2, terminal = true)],
        attributes = clinical, interventions = ivs
    )
end

_pg_mixing_vaccinated(state, name) = count(
    ind -> is_vaccinated(ind) && ind.state[:age] == name, state.individuals
)

@testset "A budget per group" begin
    @testset "PerGroup is one specification" begin
        pg = PerGroup(:under5 => 10.0, "over5" => Steps([7.0], [0.0, 50.0]))
        @test pg.groups == ["under5", "over5"]
        @test PerGroup(Dict("a" => 1.0)).groups == ["a"]
        @test_throws ArgumentError PerGroup()
        @test_throws ArgumentError PerGroup(:a => 1.0, "a" => 2.0)
        @test_throws ArgumentError CapacityConstrained(
            _pg_gv(:type); budget_per_period = PerGroup(:a => -1.0)
        )
        # A schedule per group needs a finite period, as one for everyone does.
        @test_throws ArgumentError CapacityConstrained(
            _pg_gv(:type); budget_per_period = PerGroup(:a => Steps([1.0], [0.0, 1.0]))
        )
        @test_throws ArgumentError HeldCapacity(
            ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), Quarantine(duration = 5.0));
            capacity = PerGroup(:a => -1.0)
        )
    end

    @testset "individual-based simulation over types" begin
        state = _pg_types(PerGroup(:under5 => 40.0, :over5 => 70.0))
        @test _pg_vaccinated(state, 1) == 40
        @test _pg_vaccinated(state, 2) == 70
        # A group missing from the budget is an error.
        @test_throws ArgumentError _pg_types(PerGroup(:under5 => 40.0))
        # A name that is no group of the population is an error too.
        @test_throws ArgumentError _pg_types(
            PerGroup(:under5 => 40.0, :over5 => 70.0, :over50 => 1.0)
        )
        # Groups the population leaves unnamed are numbered from one.
        state = _pg_types(PerGroup(1 => 10.0, 2 => 20.0); names = NoTypeLabels())
        @test (_pg_vaccinated(state, 1), _pg_vaccinated(state, 2)) == (10, 20)
        # The usage of each group is read by its name.
        spec = ModelSpec(
            BranchingProcess(
                _PG_MATRIX, R -> Poisson(R), Dirac(14.0);
                population = Population([30_000, 30_000]; groups = _PG_NAMES)
            );
            attributes = clinical_presentation(
                incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
            ),
            interventions = [
                CapacityConstrained(
                    _pg_gv(:type); budget_per_period = PerGroup(:under5 => 40.0, :over5 => 70.0)
                ),
            ]
        )
        state = simulate(
            spec; index_types = [fill(1, 120); fill(2, 120)], max_generations = 3,
            rng = StableRNG(1), max_cases = 10^5
        )
        cc = only(EpiBranch.interventions(spec))
        @test capacity_usage(cc, state, :under5) == (used = 40, available = 40.0)
        @test capacity_usage(cc, state, "over5") == (used = 70, available = 70.0)
    end

    @testset "a population without groups has none to name" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(1.5), Dirac(14.0); population = Population(5_000));
            interventions = [
                CapacityConstrained(
                    _pg_gv(:g); budget_per_period = PerGroup(:a => 1.0)
                ),
            ],
            attributes = (rng, ind) -> (ind.state[:g] = 1)
        )
        @test_throws ArgumentError simulate(spec; n_initial = 5, max_generations = 3)
        err = try
            simulate(spec; n_initial = 5, max_generations = 3)
        catch e
            e
        end
        @test occursin("needs a population with groups", err.msg)
    end

    @testset "individual-based simulation over mixing groups" begin
        budget = PerGroup(:under5 => 40.0, :over5 => 70.0)
        iv = CapacityConstrained(
            _pg_gv(:age; eligibility = OnSymptomOnset()); budget_per_period = budget
        )
        state = simulate(
            _pg_mixing_spec([iv]); n_initial = 20, max_time = 30.0, rng = StableRNG(2)
        )
        @test _pg_mixing_vaccinated(state, "under5") == 40
        @test _pg_mixing_vaccinated(state, "over5") == 70
        @test_throws ArgumentError simulate(
            _pg_mixing_spec(
                [
                    CapacityConstrained(
                        _pg_gv(:age); budget_per_period = PerGroup(:under5 => 1.0)
                    ),
                ]
            ); n_initial = 20, max_time = 30.0, rng = StableRNG(2)
        )
    end

    @testset "HeldCapacity takes places per group" begin
        tracing = ContactTracing(
            TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = 14.0)
        )
        spec(places) = ModelSpec(
            BranchingProcess(
                _PG_MATRIX, R -> Poisson(R), Dirac(5.0);
                population = Population([30_000, 30_000]; groups = _PG_NAMES)
            );
            attributes = clinical_presentation(
                incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
            ),
            interventions = [HeldCapacity(tracing; capacity = places, hold = 14.0)]
        )
        run(places) = simulate(
            spec(places); index_types = [fill(1, 10); fill(2, 10)], max_generations = 3,
            rng = StableRNG(3), max_cases = 10^4
        )
        state = run(PerGroup(:under5 => 1, :over5 => 2))
        hc = only(EpiBranch.interventions(spec(PerGroup(:under5 => 1, :over5 => 2))))
        @test held_usage(hc, state, 0.0, :under5).capacity == 1
        @test held_usage(hc, state, 0.0, "over5").capacity == 2
        for t in 0.0:2.0:20.0
            @test held_usage(hc, state, t, :under5).held <= 1
            @test held_usage(hc, state, t, :over5).held <= 2
        end
        @test_throws ArgumentError run(PerGroup(:under5 => 1))
    end

    @testset "generation counts read each type's own budget" begin
        spec(budget) = ModelSpec(
            BranchingProcess(
                _PG_MATRIX, R -> Poisson(R), EpiBranch.NoGenerationTime();
                population = Population([40_000, 160_000]; groups = _PG_NAMES)
            );
            interventions = [
                CapacityConstrained(_pg_gv(:type); budget_per_period = budget),
            ]
        )
        run(budget) = simulate_by_generation(
            spec(budget); n_initial = [10, 2], max_generations = 3,
            generation_length = 14.0, rng = StableRNG(1)
        )
        campaigns = EpiBranch._generation_campaigns(
            spec(PerGroup(:under5 => 100.0, :over5 => 30.0)), 14.0, true
        )
        campaign = only(campaigns)
        doses = EpiBranch._doses_by_type(StableRNG(1), campaign, 0.0, 14.0, [40_000, 160_000])
        @test doses == [100, 30]
        s = [40_000, 160_000]
        EpiBranch._vaccinate!(StableRNG(1), campaign, 0.0, 14.0, s, [40_000, 160_000])
        @test 40_000 - s[1] <= 100 && 160_000 - s[2] <= 30
        @test run(PerGroup(:under5 => 100.0, :over5 => 30.0)) isa GenerationCounts
        @test_throws ArgumentError run(PerGroup(:under5 => 100.0))
        @test_throws ArgumentError run(PerGroup(:under5 => 100.0, :other => 30.0, :over5 => 1.0))
    end

    @testset "the mixing approximations read each group's own budget" begin
        iv(budget; kwargs...) = CapacityConstrained(
            _pg_gv(:age; eligibility = OnSymptomOnset()); budget_per_period = budget, kwargs...
        )
        run(budget; allocation = TriggeredInPriority(), kwargs...) = simulate(
            MixingMeanFieldProcess(
                _pg_mixing_spec([iv(budget; kwargs...)]); removal_rate = 0.2, allocation
            ); n_initial = 20, max_time = 30.0
        )
        state = run(PerGroup(:under5 => 40.0, :over5 => 70.0))
        @test state.D[1] ≈ 40
        @test state.D[2] ≈ 70
        @test_throws ArgumentError run(PerGroup(:under5 => 40.0))

        # A budget per period and group is a daily rate to each group.
        state = run(
            PerGroup(:under5 => 5.0, :over5 => Steps([5.0], [0.0, 10.0])); period = 1.0,
            allocation = SplitByGroupSize()
        )
        @test state.D[1] > 0 && state.D[2] > 0
        @test state.D[2] > state.D[1]
    end
end
