using Distributions
using StableRNGs

# Support for the intervention specification in each model, checked against one
# table. A case is an intervention built for the groups of a model, which
# either applies it or refuses it with the wording of
# `EpiBranch._CAMPAIGN_SUPPORT`.

# Triggers the doses of a group at a fixed calendar day, so the
# individual-based engine dates a dose without a delay.
struct _SupportEligibleAt <: EpiBranch.TraceEligibility
    time::Float64
end
EpiBranch.trigger_time(e::_SupportEligibleAt, infector, state) = e.time

const _SUPPORT_MATRIX = [1.2 0.3; 0.3 1.2]

# The engines, each with the key and values of the groups it splits doses by,
# the eligibility its clock reads, and a function that runs a spec on it.
function _support_branching(spec; generations = 3, seed = 1)
    return simulate(
        spec; index_types = [fill(1, 120); fill(2, 120)], max_generations = generations,
        rng = StableRNG(seed), max_cases = 10^5,
        stopping_rules = [Extinction(), MaxGenerations(generations)]
    )
end

function _support_counts(spec; seed = 1)
    return simulate_by_generation(
        spec; n_initial = [10, 2], max_generations = 3,
        generation_length = 14.0, rng = StableRNG(seed)
    )
end

function _support_mixing_spec(ivs; n = 2_000)
    clinical = clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0)
    return ModelSpec(
        MixingProcess(;
            population = Population([n, n]), mixing_by = (:age_band,),
            mixing = FrequencyDependent([0.5 0.1; 0.1 0.5])
        );
        progression = [Transition(:recovered; rate = 0.2, terminal = true)],
        attributes = clinical, interventions = ivs
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
            BranchingProcess(
                _SUPPORT_MATRIX, R -> Poisson(R), Dirac(14.0);
                population = Population([60_000, 60_000])
            );
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
                population = Population([40_000, 160_000])
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

# Each case is an intervention that a model is asked to apply. `applies` lists
# the models that apply it; every other model refuses it in the shared wording.
const _SUPPORT_CASES = [
    (
        name = "GroupVaccination, shared budget",
        build = e -> CapacityConstrained(_support_gv(e); budget_per_period = 100.0),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination, budget per group",
        build = e -> CapacityConstrained(
            _support_gv(e); budget_per_period = PerGroup(1 => 100.0, 2 => 50.0)
        ),
        applies = (:individual, :generation_counts, :mean_field, :hybrid),
    ),
    (
        name = "GroupVaccination, schedule per group",
        build = e -> CapacityConstrained(
            _support_gv(e);
            budget_per_period = PerGroup(1 => Steps([10.0], [0.0, 20.0]), 2 => 5.0), period = 1.0
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
        name = "GroupVaccination with DeferRefused, budget per group",
        build = e -> CapacityConstrained(
            _support_gv(e); budget_per_period = PerGroup(1 => 100.0, 2 => 50.0), period = 1.0,
            on_refusal = DeferRefused()
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
            Isolation(delay = 1.0, duration = 7.0); budget_per_period = 10.0
        ),
        applies = (:individual,),
    ),
    (
        name = "Isolation under CapacityConstrained, budget per group",
        build = e -> CapacityConstrained(
            Isolation(delay = 1.0, duration = 7.0);
            budget_per_period = PerGroup(1 => 10.0, 2 => 5.0)
        ),
        applies = (:individual,),
    ),
    (
        name = "Isolation under HeldCapacity, places per group",
        build = e -> HeldCapacity(
            Isolation(delay = 1.0, duration = 7.0);
            capacity = PerGroup(1 => 10, 2 => 5)
        ),
        applies = (:individual,),
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

@testset "Each model reads the intervention specification the same way" begin
    @testset "$(case.name)" for case in _SUPPORT_CASES
        for (engine, e) in pairs(_SUPPORT_ENGINES)
            outcome = _support_outcome(() -> e.run(e.spec(_support_list(case.build(e)))))
            if engine in case.applies
                @test outcome === :applied
            else
                @test outcome isa ArgumentError
                if outcome isa ArgumentError
                    @test occursin(EpiBranch._CAMPAIGN_SUPPORT, outcome.msg)
                    @test occursin("`simulate` on a model of individuals", outcome.msg)
                end
            end
        end
    end

    @testset "a budget per group is checked against the population in every model" begin
        # The same words in every model, whichever way the budget is wrong.
        for (engine, e) in pairs(_SUPPORT_ENGINES)
            missing_group = CapacityConstrained(
                _support_gv(e); budget_per_period = PerGroup(1 => 100.0)
            )
            outcome = _support_outcome(() -> e.run(e.spec([missing_group])))
            @test outcome isa ArgumentError
            outcome isa ArgumentError &&
                @test occursin("PerGroup has no budget for the group(s) 2", outcome.msg)
            unknown_group = CapacityConstrained(
                _support_gv(e); budget_per_period = PerGroup(1 => 100.0, 2 => 1.0, 3 => 1.0)
            )
            outcome = _support_outcome(() -> e.run(e.spec([unknown_group])))
            @test outcome isa ArgumentError
            outcome isa ArgumentError &&
                @test occursin("PerGroup names 3, which is not among", outcome.msg)
        end
    end

    @testset "only an intervention that describes its doses is applied to counts" begin
        gv = _support_gv(_SUPPORT_ENGINES.individual)
        @test EpiBranch.gives_group_doses(gv)
        @test EpiBranch._group_dosing(gv) isa EpiBranch.GroupDosing
        @test !EpiBranch.gives_group_doses(RingVaccination(efficacy = 0.9))
        @test !EpiBranch.gives_group_doses(Isolation(delay = 1.0, duration = 7.0))
        @test EpiBranch._group_dosing(Isolation(delay = 1.0, duration = 7.0)) === nothing
    end

    @testset "an intervention of one's own is applied to counts through the dose traits" begin
        struct _SupportPrEP <: AbstractIntervention
            key::Symbol
        end
        EpiBranch.gives_group_doses(::_SupportPrEP) = true
        EpiBranch.dose_group_key(p::_SupportPrEP) = p.key
        EpiBranch.dose_efficacy(::_SupportPrEP) = 0.6
        dosing = EpiBranch._group_dosing(_SupportPrEP(:type))
        @test dosing.group_key === :type
        @test dosing.efficacy == 0.6
        @test dosing.mode isa LeakyMode
        @test dosing.coverage == 1.0
        @test dosing.dose_delay == 0.0
        cc = CapacityConstrained(_SupportPrEP(:type); budget_per_period = 50.0)
        campaign = EpiBranch._campaign(cc, (0.0, Inf), nothing)
        @test campaign.efficacy == 0.6
    end

    @testset "a shared budget is one pool in every model" begin
        # 100 doses between two groups of hundreds of people: one pool gives
        # 100 doses in all.
        shared(e) = CapacityConstrained(_support_gv(e); budget_per_period = 100.0)
        e = _SUPPORT_ENGINES.individual
        state = _support_branching(e.spec([shared(e)]))
        @test count(is_vaccinated, individuals(state)) == 100

        e = _SUPPORT_ENGINES.mean_field
        state = _support_mean_field(e.spec([shared(e)]))
        @test sum(state.D) ≈ 100

        e = _SUPPORT_ENGINES.generation_counts
        campaign = EpiBranch._campaign(shared(e), (0.0, Inf), nothing)
        @test EpiBranch._interval_doses(campaign, 0.0, 14.0) == 100
        doses = EpiBranch._generation_doses([campaign], [[0]], 14.0, 0)[1]
        @test only(doses)[1] == 100
    end

end
