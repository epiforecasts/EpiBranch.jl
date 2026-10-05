# Custom IsolationEligibility used by the user-extension test below.
struct OnlyOlder <: EpiBranch.IsolationEligibility
    age_threshold::Int
end
function EpiBranch.is_eligible_for_isolation(e::OnlyOlder, ind, state)
    return !is_asymptomatic(ind) && get(ind.state, :age, 0) >= e.age_threshold
end
EpiBranch._required_for_eligibility(::OnlyOlder) = [:onset_time, :asymptomatic, :age]

@testset "Isolation trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default keyword constructor reproduces previous behaviour" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), isolation_duration = Inf)
        @test iso.eligibility isa SymptomaticOnly
        @test iso.test_sensitivity == 1.0
        @test iso.post_isolation_transmission == 0.0
    end

    @testset "AllCases eligibility isolates asymptomatic individuals too" begin
        # Make the population partially asymptomatic, then with AllCases
        # eligibility (and full sensitivity) every case should get
        # :test_positive = true.
        clin_mixed = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.5
        )
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases(), isolation_duration = Inf)
        rng = StableRNG(42)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = clin_mixed
            );
            max_cases = 100,
            rng = rng
        )
        # AllCases + sensitivity = 1.0 means every individual tests
        # positive, including asymptomatic ones.
        @test all(get(ind.state, :test_positive, false) for ind in state.individuals)
    end

    @testset "test_sensitivity accepts a function" begin
        # Age-conditional sensitivity: 0+ → 0%, 50+ → 100%.
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(
            onset_to_isolation_delay = Exponential(0.1),
            test_sensitivity = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0, isolation_duration = Inf
        )
        rng = StableRNG(13)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        for ind in state.individuals
            expected = !is_asymptomatic(ind) && ind.state[:age] >= 50
            @test get(ind.state, :test_positive, false) == expected
        end
    end

    @testset "onset_to_isolation_delay accepts a function" begin
        # Age-conditional delay stands in for a delay that depends on
        # per-individual state recorded by another intervention, e.g. a
        # group's own event time.
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(
            onset_to_isolation_delay = (rng, ind) -> ind.state[:age] >= 50 ? 0.1 : 5.0, isolation_duration = Inf
        )
        rng = StableRNG(21)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        # Isolation time is only set once `resolve_individual!` has run on an
        # individual, which does not happen for every case before the run
        # stops at `max_cases`; restrict the check to those it did reach.
        checked = 0
        for ind in state.individuals
            is_test_positive(ind) || continue
            isfinite(isolation_time(ind)) || continue
            checked += 1
            expected_delay = ind.state[:age] >= 50 ? 0.1 : 5.0
            @test isolation_time(ind) - onset_time(ind) ≈ expected_delay
        end
        @test checked > 0
    end

    @testset "required_fields dispatches on eligibility" begin
        # Default SymptomaticOnly requires :asymptomatic.
        @test :asymptomatic in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), isolation_duration = Inf)
        )
        # AllCases doesn't.
        @test :asymptomatic ∉ EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases(), isolation_duration = Inf)
        )
        # Custom eligibility declares its own required fields.
        @test :age in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = OnlyOlder(50), isolation_duration = Inf)
        )
    end

    @testset "Custom IsolationEligibility integrates end-to-end" begin
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(onset_to_isolation_delay = Exponential(0.1), eligibility = OnlyOlder(50), isolation_duration = Inf)
        rng = StableRNG(17)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        for ind in state.individuals
            ind.state[:test_positive] && @test ind.state[:age] >= 50
        end
    end
end
