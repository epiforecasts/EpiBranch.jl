# Custom IsolationEligibility used by the user-extension test below.
struct OnlyOlder <: EpiInterventions.IsolationEligibility
    age_threshold::Int
end
function EpiInterventions.is_eligible_for_isolation(e::OnlyOlder, ind, state)
    !is_asymptomatic(ind) && get(ind.state, :age, 0) >= e.age_threshold
end
EpiInterventions._required_for_eligibility(::OnlyOlder) = [:onset_time, :asymptomatic, :age]

@testset "Isolation trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default keyword constructor reproduces previous behaviour" begin
        iso = Isolation(delay = Exponential(1.0))
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
        iso = Isolation(delay = Exponential(1.0), eligibility = AllCases())
        rng = StableRNG(42)
        state = simulate(BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso], attributes = clin_mixed,
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        # AllCases + sensitivity = 1.0 means every individual tests
        # positive, including asymptomatic ones.
        @test all(get(ind.state, :test_positive, false) for ind in state.individuals)
    end

    @testset "test_sensitivity accepts a function" begin
        # Age-conditional sensitivity: 0+ → 0%, 50+ → 100%.
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        iso = Isolation(
            delay = Exponential(0.1),
            test_sensitivity = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0
        )
        rng = StableRNG(13)
        state = simulate(BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso], attributes = attrs,
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        for ind in state.individuals
            expected = !is_asymptomatic(ind) && ind.state[:age] >= 50
            @test get(ind.state, :test_positive, false) == expected
        end
    end

    @testset "required_fields dispatches on eligibility" begin
        # Default SymptomaticOnly requires :asymptomatic.
        @test :asymptomatic in EpiBranchCore.required_fields(
            Isolation(delay = Exponential(1.0)))
        # AllCases doesn't.
        @test :asymptomatic ∉ EpiBranchCore.required_fields(
            Isolation(delay = Exponential(1.0), eligibility = AllCases()))
        # Custom eligibility declares its own required fields.
        @test :age in EpiBranchCore.required_fields(
            Isolation(delay = Exponential(1.0), eligibility = OnlyOlder(50)))
    end

    @testset "Custom IsolationEligibility integrates end-to-end" begin
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        iso = Isolation(delay = Exponential(0.1), eligibility = OnlyOlder(50))
        rng = StableRNG(17)
        state = simulate(BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso], attributes = attrs,
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        for ind in state.individuals
            ind.state[:test_positive] && @test ind.state[:age] >= 50
        end
    end
end
