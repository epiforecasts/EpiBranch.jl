using Random
using Distributions
using StableRNGs
using DataFrames

function _toy_case(;
        id = 1, infection = 0.0, onset = NaN, isolation = Inf, trace = NaN,
        test = NaN, positive = false
    )
    state = Dict{Symbol, Any}(:onset_time => onset, :test_result => positive)
    isfinite(isolation) && (state[:isolation_time] = isolation)
    isnan(trace) || (state[:trace_time] = trace)
    isnan(test) || (state[:test_time] = test)
    return Individual(; id, infection_time = infection, state)
end

function _toy_sim_state(individuals)
    return SimulationState(
        individuals, Int[], 0, Xoshiro(1), length(individuals), true,
        NoPopulation(), 100.0, NoAttributes(), EpiBranch.AbstractClinicalTransition[]
    )
end

# Load (log10 copies) rising to a peak five days after infection, then falling.
_acute_load(rng, ind) = t -> max(0.0, 8.0 - abs(t - 5.0) * 1.6)

@testset "Case definitions" begin
    @testset "history only shows events up to its time" begin
        ind = _toy_case(; onset = 3.0, trace = 5.0)
        early = CaseHistory(ind, 4.0, 0.5)
        @test has_happened(early, :onset)
        @test !has_happened(early, :trace)
        @test event_time(early, :trace) == Inf
        @test event_time(early, :onset) == 3.0
        @test has_happened(CaseHistory(ind, 5.0, 0.5), :trace)
    end

    @testset "ClinicalCaseDefinition changes over time" begin
        def = ClinicalCaseDefinition()
        # Symptoms at 3, traced at 5, positive result at 6.
        ind = _toy_case(; onset = 3.0, trace = 5.0, test = 6.0, isolation = 6.0, positive = true)
        state = _toy_sim_state([ind])
        EpiBranch._classify_cases!(state, def)
        @test EpiBranch._CLASSIFICATION_CHANGES_KEY in keys(ind.state)
        changes = case_classification_changes(state)
        @test changes.time == [3.0, 5.0, 6.0]
        @test changes.classification == [:suspected, :probable, :confirmed]
        @test case_classification(ind) == :confirmed

        # A contact without symptoms is not a case; negative test never confirms.
        contact = _toy_case(; id = 2, trace = 1.0, isolation = 2.0)
        asymptomatic = _toy_case(; id = 3, onset = NaN, positive = true)
        state = _toy_sim_state([contact, asymptomatic])
        EpiBranch._classify_cases!(state, def)
        @test ismissing(case_classification(contact))
        @test ismissing(case_classification(asymptomatic))
        @test nrow(case_classification_changes(state)) == 0
    end

    @testset "FixedProportionConfirmed" begin
        inds = [_toy_case(; id = i, onset = 1.0) for i in 1:2000]
        state = _toy_sim_state(inds)
        EpiBranch._classify_cases!(state, FixedProportionConfirmed(0.3))
        share = count(==(:confirmed), case_classification.(inds)) / length(inds)
        @test 0.25 < share < 0.35
        @test all(in((:confirmed, :probable)), case_classification.(inds))
        # Re-evaluating keeps each case's classification.
        before = case_classification.(inds)
        EpiBranch._classify_cases!(state, FixedProportionConfirmed(0.3))
        @test case_classification.(inds) == before
        @test_throws ArgumentError FixedProportionConfirmed(1.5)
    end

    @testset "a definition written outside the package" begin
        struct HospitalisedIsSevere <: EpiBranch.AbstractCaseDefinition end
        function EpiBranch.classify_case(::HospitalisedIsSevere, h)
            has_happened(h, :onset) || return missing
            return has_happened(h, :isolation) ? "severe" : "mild"
        end
        ind = _toy_case(; onset = 2.0, isolation = 4.0)
        state = _toy_sim_state([ind])
        EpiBranch._classify_cases!(state, HospitalisedIsSevere())
        @test case_classification_changes(state).classification == ["mild", "severe"]

        by_function = h -> has_happened(h, :onset) ? :symptomatic : missing
        EpiBranch._classify_cases!(state, by_function)
        @test case_classification(ind) == :symptomatic
    end

    @testset "in a ModelSpec" begin
        process = BranchingProcess(NegBin(2.5, 0.5), Gamma(2.0, 2.0))
        clinical = clinical_presentation(incubation_period = LogNormal(1.2, 0.4))
        assay = Assay(value = load -> 40.0 - 3.0 * load, detect = 2.0)
        interventions = [
            ContactTracing(
                OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 10.0)
            ),
        ]
        spec = ModelSpec(
            process; attributes = [clinical, within_host(_acute_load)],
            progression = [Testing(assay = assay, delay = Exponential(2.0))],
            interventions,
            case_definition = ClinicalCaseDefinition()
        )
        @test EpiBranch.case_definition(spec) isa ClinicalCaseDefinition
        @test EpiBranch.case_definition(ModelSpec(spec; interventions = [])) isa ClinicalCaseDefinition
        @test EpiBranch.case_definition(process) === nothing

        state = simulate(spec; max_cases = 300, condition = 100:300, rng = StableRNG(11))
        changes = case_classification_changes(state)
        @test nrow(changes) > 0
        @test issorted(changes.time)
        @test Set(skipmissing(changes.classification)) ⊆
            Set([:suspected, :probable, :confirmed])
        # Confirmed only from a recorded positive test, and never before the
        # case had symptoms.
        for ind in state.individuals
            is_infected(ind) || continue
            if case_classification(ind) === :confirmed
                @test ind.state[:test_result]
                @test ind.state[:test_time] >= onset_time(ind)
            end
        end
        @test any(==(:confirmed), skipmissing(changes.classification))
        # Some case was reclassified after onset (by tracing or a result).
        @test any(c -> c > 1, combine(groupby(changes, :id), nrow => :n).n)

        ll = linelist(state)
        @test "case_classification" in names(ll)
        # The classification does not alter the simulation.
        plain = simulate(
            ModelSpec(spec; case_definition = nothing);
            max_cases = 300, condition = 100:300, rng = StableRNG(11)
        )
        @test isequal(linelist(plain), ll[!, Not(:case_classification)])
        @test nrow(case_classification_changes(plain)) == 0

        states = simulate(spec, 3; max_cases = 50, rng = StableRNG(2))
        @test all(s -> nrow(case_classification_changes(s)) > 0, states)
    end

    @testset "no confirmed cases without a Testing step" begin
        spec = ModelSpec(
            BranchingProcess(NegBin(2.5, 0.5), Gamma(2.0, 2.0));
            attributes = clinical_presentation(incubation_period = LogNormal(1.2, 0.4)),
            interventions = [Isolation(onset_to_isolation_delay = 2.0, duration = 14.0)],
            case_definition = ClinicalCaseDefinition()
        )
        state = simulate(spec; max_cases = 100, rng = StableRNG(3))
        labels = Set(skipmissing(case_classification_changes(state).classification))
        @test !isempty(labels)
        @test :confirmed ∉ labels
    end

    @testset "continued run is classified again" begin
        spec = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2_000);
            progression = [
                Transition(:onset; delay = Exponential(2.0)),
                Transition(:recovered; from = :onset, delay = Exponential(4.0), terminal = true),
            ],
            case_definition = FixedProportionConfirmed(0.5),
        )
        state = simulate(spec; max_time = 10.0, rng = StableRNG(4))
        n1 = nrow(case_classification_changes(state))
        first_labels = Dict(i.id => case_classification(i) for i in state.individuals if is_infected(i) && !ismissing(case_classification(i)))
        simulate(spec, state; max_time = 14.0)
        @test nrow(case_classification_changes(state)) >= n1
        @test all(first_labels) do (id, label)
            case_classification(state.individuals[id]) == label
        end
    end
end
