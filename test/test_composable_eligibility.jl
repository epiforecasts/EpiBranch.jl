using Test
using EpiBranch
using EpiBranch: OnSymptomOnset, OnLabConfirmation, OnIsolation, TraceEveryone, TraceNobody,
                 Any, All, Unless, is_eligible, required_fields

# Mock individual and state for testing
struct MockIndividual
    state::Dict{Symbol, Any}
end

struct MockState
    rng::Any
end

# Helper functions
is_asymptomatic(ind) = get(ind.state, :asymptomatic, false)
is_isolated(ind) = get(ind.state, :isolated, false)

@testset "Basic Eligibility Types" begin
    state = MockState(nothing)

    @testset "OnSymptomOnset" begin
        policy = OnSymptomOnset()

        # Symptomatic parent should be eligible
        symptomatic_parent = MockIndividual(Dict(:asymptomatic => false, :isolated => false))
        contact = MockIndividual(Dict())
        @test is_eligible(policy, symptomatic_parent, contact, state) == true

        # Asymptomatic parent should not be eligible
        asymptomatic_parent = MockIndividual(Dict(:asymptomatic => true, :isolated => false))
        @test is_eligible(policy, asymptomatic_parent, contact, state) == false

        # Isolation status should not matter
        symptomatic_isolated = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        @test is_eligible(policy, symptomatic_isolated, contact, state) == true
    end


    @testset "NoTracing" begin
        policy = NoTracing()
        parent = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        contact = MockIndividual(Dict())

        @test is_eligible(policy, parent, contact, state) == false
    end
end

@testset "Composition Operators" begin
    state = MockState(nothing)
    contact = MockIndividual(Dict())

    @testset "Any (OR composition)" begin
        # Should be eligible if ANY condition is met
        policy = Any(OnSymptomOnset(), SymptomaticParent())

        # Symptomatic but not isolated - should be eligible via OnSymptomOnset
        parent1 = MockIndividual(Dict(:asymptomatic => false, :isolated => false))
        @test is_eligible(policy, parent1, contact, state) == true

        # Symptomatic and isolated - should be eligible via both
        parent2 = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        @test is_eligible(policy, parent2, contact, state) == true

        # Asymptomatic - should not be eligible via either
        parent3 = MockIndividual(Dict(:asymptomatic => true, :isolated => false))
        @test is_eligible(policy, parent3, contact, state) == false
    end

    @testset "All (AND composition)" begin
        # Should be eligible only if ALL conditions are met
        # This is equivalent to SymptomaticParent
        policy = All(OnSymptomOnset(), SymptomaticParent())

        # Symptomatic but not isolated - should not be eligible
        parent1 = MockIndividual(Dict(:asymptomatic => false, :isolated => false))
        @test is_eligible(policy, parent1, contact, state) == false

        # Symptomatic and isolated - should be eligible
        parent2 = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        @test is_eligible(policy, parent2, contact, state) == true

        # Asymptomatic - should not be eligible
        parent3 = MockIndividual(Dict(:asymptomatic => true, :isolated => false))
        @test is_eligible(policy, parent3, contact, state) == false
    end

    @testset "Unless (conditional exclusion)" begin
        # Trace symptomatic cases unless they are isolated
        policy = Unless(OnSymptomOnset(), SymptomaticParent())

        # Symptomatic but not isolated - should be eligible
        parent1 = MockIndividual(Dict(:asymptomatic => false, :isolated => false))
        @test is_eligible(policy, parent1, contact, state) == true

        # Symptomatic and isolated - should be excluded
        parent2 = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        @test is_eligible(policy, parent2, contact, state) == false

        # Asymptomatic - primary condition fails, should not be eligible
        parent3 = MockIndividual(Dict(:asymptomatic => true, :isolated => false))
        @test is_eligible(policy, parent3, contact, state) == false
    end
end

@testset "Nested Composition" begin
    state = MockState(nothing)
    contact = MockIndividual(Dict())

    # Complex policy: (OnSymptomOnset OR OnCaseDetection) AND NOT NoTracing
    complex_policy = All(
        Any(OnSymptomOnset(), OnCaseDetection()),
        Unless(OnCaseDetection(), NoTracing())
    )

    # Should work for symptomatic cases
    symptomatic = MockIndividual(Dict(:asymptomatic => false, :isolated => false))
    @test is_eligible(complex_policy, symptomatic, contact, state) == true

    # Should work for asymptomatic cases (via OnCaseDetection)
    asymptomatic = MockIndividual(Dict(:asymptomatic => true, :isolated => false))
    @test is_eligible(complex_policy, asymptomatic, contact, state) == true
end

@testset "Additional Eligibility Types" begin
    state = MockState(nothing)
    contact = MockIndividual(Dict())

    @testset "OnLabConfirmation" begin
        policy = OnLabConfirmation()

        # Symptomatic and test positive
        positive_parent = MockIndividual(Dict(:asymptomatic => false, :test_positive => true))
        @test is_eligible(policy, positive_parent, contact, state) == true

        # Symptomatic but test negative
        negative_parent = MockIndividual(Dict(:asymptomatic => false, :test_positive => false))
        @test is_eligible(policy, negative_parent, contact, state) == false

        # Asymptomatic
        asymptomatic_parent = MockIndividual(Dict(:asymptomatic => true, :test_positive => true))
        @test is_eligible(policy, asymptomatic_parent, contact, state) == false
    end

    @testset "TraceEveryone" begin
        policy = TraceEveryone()
        any_parent = MockIndividual(Dict(:asymptomatic => true))
        @test is_eligible(policy, any_parent, contact, state) == true
    end

    @testset "TraceNobody" begin
        policy = TraceNobody()
        any_parent = MockIndividual(Dict(:asymptomatic => false, :isolated => true))
        @test is_eligible(policy, any_parent, contact, state) == false
    end
end

@testset "Required Fields Validation" begin
    @test required_fields(OnSymptomOnset()) == [:asymptomatic]
    @test required_fields(OnLabConfirmation()) == [:asymptomatic, :test_positive]
    @test required_fields(OnIsolation()) == [:asymptomatic, :isolated]
    @test required_fields(TraceEveryone()) == Symbol[]
    @test required_fields(TraceNobody()) == Symbol[]

    # Composition should union requirements
    any_policy = Any(OnSymptomOnset(), OnIsolation())
    expected = union([:asymptomatic], [:asymptomatic, :isolated])
    @test Set(required_fields(any_policy)) == Set(expected)

    all_policy = All(OnSymptomOnset(), TraceEveryone())
    @test required_fields(all_policy) == [:asymptomatic]
end

@testset "Integration with ContactTracing" begin
    using Distributions

    # Test that new eligibility types work with ContactTracing constructor
    ct1 = ContactTracing(
        OnSymptomOnset(),
        ConstantRate(0.7),
        ConstantDelay(Exponential(1.5)),
        Quarantine()
    )
    @test ct1.eligibility isa OnSymptomOnset

    # Test with composition
    ct2 = ContactTracing(
        Any(OnSymptomOnset(), SymptomaticParent()),
        ConstantRate(0.5),
        ConstantDelay(Exponential(2.0)),
        FlagOnly()
    )
    @test ct2.eligibility isa Any

    # Test convenience constructor still works
    ct3 = ContactTracing(
        probability = 0.6,
        isolation_to_trace_delay = Exponential(2.0),
        eligibility = OnSymptomOnset()
    )
    @test ct3.eligibility isa OnSymptomOnset
end

@testset "Custom Eligibility Example" begin
    # Example of how users can extend the system
    struct AgeBasedEligibility <: TraceEligibility
        min_age::Int
    end

    function is_eligible(e::AgeBasedEligibility, parent, contact, state)
        !is_asymptomatic(parent) && get(parent.state, :age, 0) >= e.min_age
    end

    state = MockState(nothing)
    contact = MockIndividual(Dict())
    policy = AgeBasedEligibility(65)

    # Symptomatic elderly parent should be eligible
    elderly_parent = MockIndividual(Dict(:asymptomatic => false, :age => 70))
    @test is_eligible(policy, elderly_parent, contact, state) == true

    # Symptomatic young parent should not be eligible
    young_parent = MockIndividual(Dict(:asymptomatic => false, :age => 30))
    @test is_eligible(policy, young_parent, contact, state) == false

    # Asymptomatic elderly parent should not be eligible
    asymptomatic_elderly = MockIndividual(Dict(:asymptomatic => true, :age => 70))
    @test is_eligible(policy, asymptomatic_elderly, contact, state) == false
end