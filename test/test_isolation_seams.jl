# Custom IsolationEligibility used by the user-extension test below.
struct OnlyOlder <: EpiBranch.IsolationEligibility
    age_threshold::Int
end
function EpiBranch.is_eligible_for_isolation(e::OnlyOlder, ind, state)
    !is_asymptomatic(ind) && get(ind.state, :age, 0) >= e.age_threshold
end
EpiBranch._required_for_eligibility(::OnlyOlder) = [:onset_time, :asymptomatic, :age]

@testset "Isolation trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default keyword constructor reproduces previous behaviour" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        @test iso.eligibility isa SymptomaticOnly
        @test iso.test_sensitivity == 1.0
        @test iso.post_isolation_transmission == 0.0
    end

    @testset "AllCases eligibility isolates asymptomatic individuals too" begin
        # Make the population partially asymptomatic, then with AllCases
        # eligibility (and full sensitivity) every case should get
        # :test_positive = true.
        clin_mixed = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.5)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases())
        rng = StableRNG(42)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(2.0), Exponential(5.0)), [iso]),
                clin_mixed);
            max_cases = 100,
            rng = rng)
        # AllCases + sensitivity = 1.0 means every individual tests
        # positive, including asymptomatic ones.
        @test all(get(ind.state, :test_positive, false) for ind in state.individuals)
    end

    @testset "test_sensitivity accepts a function" begin
        # Age-conditional sensitivity: 0+ → 0%, 50+ → 100%.
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        iso = Isolation(
            onset_to_isolation_delay = Exponential(0.1),
            test_sensitivity = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0
        )
        rng = StableRNG(13)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(2.0), Exponential(5.0)), [iso]),
                attrs);
            max_cases = 100,
            rng = rng)
        for ind in state.individuals
            expected = !is_asymptomatic(ind) && ind.state[:age] >= 50
            @test get(ind.state, :test_positive, false) == expected
        end
    end

    @testset "required_fields dispatches on eligibility" begin
        # Default SymptomaticOnly requires :asymptomatic.
        @test :asymptomatic in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0)))
        # AllCases doesn't.
        @test :asymptomatic ∉ EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases()))
        # Custom eligibility declares its own required fields.
        @test :age in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = OnlyOlder(50)))
    end

    @testset "Custom IsolationEligibility integrates end-to-end" begin
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        iso = Isolation(onset_to_isolation_delay = Exponential(0.1), eligibility = OnlyOlder(50))
        rng = StableRNG(17)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(2.0), Exponential(5.0)), [iso]),
                attrs);
            max_cases = 100,
            rng = rng)
        for ind in state.individuals
            ind.state[:test_positive] && @test ind.state[:age] >= 50
        end
    end
end
