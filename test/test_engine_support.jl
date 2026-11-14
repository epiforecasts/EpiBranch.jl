using Distributions
using StableRNGs

# What each engine reads of the intervention specification, checked against one
# table. A case is an intervention built for an engine's own groups; an engine
# either applies it or refuses it with the shared wording of
# `EpiBranch._CAMPAIGN_SUPPORT`. The same table is in the documentation
# (the intervention support page of the tutorials).

# Triggers a group's doses at a fixed calendar day, so the individual-based
# engine dates a dose without a delay.
struct _SupportEligibleAt <: EpiBranch.TraceEligibility
    time::Float64
end
EpiBranch.trigger_time(e::_SupportEligibleAt, infector, state) = e.time

const _SUPPORT_MATRIX = [1.2 0.3; 0.3 1.2]

# The engines, each with the key and values of the groups it splits doses by,
# the eligibility its own clock reads, and a function that runs a spec on it.
function _support_branching(spec; generations = 3, seed = 1)
    return simulate(
        spec; index_types = [fill(1, 120); fill(2, 120)], max_generations = generations,
        rng = StableRNG(seed), max_cases = 10^5
    )
end

function _support_counts(spec; seed = 1)
    return simulate_by_generation(
        spec; n_initial = [10, 2], max_generations = 3,
        population_by_type = [40_000, 160_000], depletion = StartOfGenerationDepletion(),
        generation_length = 14.0, rng = StableRNG(seed)
    )
end

function _support_mixing_spec(ivs; n = 2_000)
    band = (rng, ind) -> (ind.state[:age_band] = ind.id % 2 + 1)
    clinical = clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0)
    return ModelSpec(
        MixingProcess(;
            population_size = 2n, mixing_by = (:age_band,),
            rate = (from, to) -> (from == to ? 0.5 : 0.1) / n
        );
        progression = [Transition(:recovered; rate = 0.2, terminal = true)],
        attributes = [band, clinical], interventions = ivs
    )
end

_support_mean_field(spec) = simulate(
    MixingMeanFieldProcess(spec; removal_rate = 0.2); n_initial = 20, max_time = 10.0
)
_support_hybrid(spec) = simulate(
    HybridProcess(
        spec, MixingMeanFieldProcess(spec; removal_rate = 0.2);
        threshold = 30, check_interval = 0.5
    );
    n_initial = 20, max_time = 10.0, rng = StableRNG(1)
)

const _SUPPORT_ENGINES = (
    individual = (
        key = :type, groups = (1, 2), eligibility = _SupportEligibleAt(30.0),
        spec = ivs -> ModelSpec(
            BranchingProcess(_SUPPORT_MATRIX, R -> Poisson(R), Dirac(14.0));
            attributes = clinical_presentation(
                incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
            ), interventions = ivs
        ),
        run = _support_branching,
    ),
    generation_counts = (
        key = :type, groups = (1, 2), eligibility = _SupportEligibleAt(30.0),
        spec = ivs -> ModelSpec(
            BranchingProcess(
                _SUPPORT_MATRIX, R -> Poisson(R), EpiBranch.NoGenerationTime();
                population_size = 200_000
            ); interventions = ivs
        ),
        run = _support_counts,
    ),
    mean_field = (
        key = :age_band, groups = (1, 2), eligibility = OnSymptomOnset(),
        spec = _support_mixing_spec, run = _support_mean_field,
    ),
    hybrid = (
        key = :age_band, groups = (1, 2), eligibility = OnSymptomOnset(),
        spec = _support_mixing_spec, run = _support_hybrid,
    ),
)

_support_gv(e, key = e.key) = GroupVaccination(;
    efficacy = 0.9, group_key = key, eligibility = e.eligibility
)

# The specification: each case is the intervention an engine is asked to
# apply, and `applies` the engines that apply it. Any other engine refuses it
# (`:refused`) in the shared wording, apart from the cases in `invalid`, which
# every engine rejects for what is wrong with the specification itself.
const _SUPPORT_CASES = [
    (
        name = "GroupVaccination, shared budget",
        build = e -> CapacityConstrained(_support_gv(e); budget_per_period = 100.0),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination, PerGroup budget",
        build = e -> CapacityConstrained(
            _support_gv(e); budget_per_period = PerGroup(e.groups[1] => 60.0, e.groups[2] => 40.0)
        ),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination, PerGroup budget with a schedule",
        build = e -> CapacityConstrained(
            _support_gv(e);
            budget_per_period = PerGroup(
                e.groups[1] => Steps([1.0, 20.0], [0.0, 10.0, 0.0]), e.groups[2] => 40.0
            ), period = 1.0
        ),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination under Scheduled with a window",
        build = e -> Scheduled(
            CapacityConstrained(_support_gv(e); budget_per_period = 100.0);
            start_time = 5.0, end_time = 60.0
        ),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination under Scheduled after cases",
        build = e -> Scheduled(
            CapacityConstrained(_support_gv(e); budget_per_period = 100.0); start_after_cases = 5
        ),
        applies = (:individual,),
    ),
    (
        name = "GroupVaccination with DeferRefused",
        build = e -> CapacityConstrained(
            _support_gv(e); budget_per_period = 100.0, period = 1.0, on_refusal = DeferRefused()
        ),
        applies = (:individual,),
    ),
    (
        name = "RingVaccination under CapacityConstrained",
        build = e -> [
            ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf)),
            CapacityConstrained(RingVaccination(efficacy = 0.9); budget_per_period = 10.0),
        ],
        applies = (:individual,),
    ),
    (
        name = "Isolation under CapacityConstrained, shared budget",
        build = e -> CapacityConstrained(
            Isolation(onset_to_isolation_delay = 1.0, duration = 7.0); budget_per_period = 10.0
        ),
        applies = (:individual,),
    ),
    (
        name = "Isolation under CapacityConstrained, PerGroup budget",
        build = e -> CapacityConstrained(
            Isolation(onset_to_isolation_delay = 1.0, duration = 7.0);
            budget_per_period = PerGroup(e.groups[1] => 10.0, e.groups[2] => 5.0; key = e.key)
        ),
        applies = (:individual,),
    ),
    (
        name = "ContactTracing under CapacityConstrained, PerGroup budget",
        build = e -> CapacityConstrained(
            ContactTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf));
            budget_per_period = PerGroup(e.groups[1] => 10.0, e.groups[2] => 5.0; key = e.key)
        ),
        applies = (:individual,),
    ),
]

# Rejected by every engine: a group the PerGroup budget does not name.
const _SUPPORT_INVALID = [
    (
        name = "PerGroup budget missing a group",
        build = e -> CapacityConstrained(
            _support_gv(e); budget_per_period = PerGroup(e.groups[1] => 10.0)
        ),
    ),
]

_support_list(iv) = iv isa AbstractVector ? iv : [iv]

_support_outcome(f) = try
    f()
    :applied
catch err
    err isa ArgumentError || rethrow()
    err
end

@testset "Each engine reads the intervention specification the same way" begin
    @testset "$(case.name)" for case in _SUPPORT_CASES
        for (engine, e) in pairs(_SUPPORT_ENGINES)
            outcome = _support_outcome(() -> e.run(e.spec(_support_list(case.build(e)))))
            if engine in case.applies
                @test outcome === :applied
            else
                # The same words from every engine that refuses.
                @test outcome isa ArgumentError
                outcome isa ArgumentError &&
                    @test occursin(EpiBranch._CAMPAIGN_SUPPORT, outcome.msg)
            end
        end
    end

    @testset "$(case.name)" for case in _SUPPORT_INVALID
        for (engine, e) in pairs(_SUPPORT_ENGINES)
            outcome = _support_outcome(() -> e.run(e.spec(_support_list(case.build(e)))))
            @test outcome isa ArgumentError
            outcome isa ArgumentError && @test occursin("no budget for group", outcome.msg)
        end
    end

    @testset "a shared budget is one pool in every engine" begin
        # 100 doses between two groups of hundreds of people: one pool gives
        # 100 doses in all, where a budget per group would give each its own.
        shared(e) = CapacityConstrained(_support_gv(e); budget_per_period = 100.0)
        e = _SUPPORT_ENGINES.individual
        state = _support_branching(e.spec([shared(e)]))
        @test count(is_vaccinated, state.individuals) == 100

        e = _SUPPORT_ENGINES.mean_field
        state = _support_mean_field(e.spec([shared(e)]))
        @test sum(state.D) ≈ 100

        e = _SUPPORT_ENGINES.generation_counts
        campaign = EpiBranch._campaign(shared(e), (0.0, Inf), nothing)
        @test EpiBranch._interval_doses(campaign, 0.0, 14.0) == 100
        s = [40_000, 160_000]
        EpiBranch._vaccinate!(StableRNG(1), campaign, 0.0, 14.0, s, [40_000, 160_000])
        @test sum([40_000, 160_000] .- s) <= 100
    end

    @testset "a PerGroup budget gives each group its own doses in every engine" begin
        per_group(e) = CapacityConstrained(
            _support_gv(e); budget_per_period = PerGroup(e.groups[1] => 0.0, e.groups[2] => 70.0)
        )
        e = _SUPPORT_ENGINES.individual
        state = _support_branching(e.spec([per_group(e)]))
        vaccinated(type) = count(
            i -> is_vaccinated(i) && i.state[:type] == type, state.individuals
        )
        @test vaccinated(1) == 0
        @test vaccinated(2) == 70

        e = _SUPPORT_ENGINES.mean_field
        spec = e.spec([per_group(e)])
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        state = _support_mean_field(spec)
        first_group = findfirst(g -> g[1] == 1, model.groups)
        second_group = findfirst(g -> g[1] == 2, model.groups)
        @test state.D[first_group] == 0
        @test state.D[second_group] ≈ 70

        e = _SUPPORT_ENGINES.generation_counts
        campaign = EpiBranch._campaign(per_group(e), (0.0, Inf), nothing)
        @test EpiBranch._interval_doses(campaign, 0.0, 14.0, 1) == 0
        @test EpiBranch._interval_doses(campaign, 0.0, 14.0, 2) == 70
    end
end

@testset "a PerGroup budget shared out as a rate stays within each group" begin
    e = _SUPPORT_ENGINES.mean_field
    budget = PerGroup(1 => Steps([10.0], [30.0, 0.0]), 2 => Steps([10.0], [70.0, 0.0]))
    campaign = CapacityConstrained(_support_gv(e); budget_per_period = budget, period = 1.0)
    spec = e.spec([campaign])
    for allocation in (SplitByGroupSize(), SplitBySusceptibles())
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2, allocation)
        state = simulate(model; n_initial = 20, max_time = 15.0)
        by_value(v) = state.D[findfirst(g -> g[1] == v, model.groups)]
        @test by_value(1) ≈ 300.0 rtol = 1.0e-3
        @test by_value(2) ≈ 700.0 rtol = 1.0e-3
    end
end

@testset "a PerGroup budget handed from an exact run keeps the doses given to each group" begin
    e = _SUPPORT_ENGINES.mean_field
    budget = PerGroup(1 => 0.0, 2 => 50.0)
    campaign = CapacityConstrained(_support_gv(e); budget_per_period = budget)
    spec = e.spec([campaign])
    approx = MixingMeanFieldProcess(spec; removal_rate = 0.2)
    exact = simulate(spec; n_initial = 20, max_time = 8.0, rng = StableRNG(2))
    handed = EpiBranch._to_approx_state(approx, exact, 8.0)
    group = findfirst(g -> g[1] == 2, approx.groups)
    @test handed.D[group] == count(is_vaccinated, exact.individuals)
    @test all(
        entry -> entry[3] == findfirst(==(2), approx.campaign.values), handed.campaign.log
    )
end
