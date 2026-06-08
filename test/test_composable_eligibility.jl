using EpiBranch: is_eligible, required_fields

# Custom eligibility used by the extension test below. Structs must be
# defined at top level, so it lives here rather than inside the testset.
struct SymptomaticOver65 <: EpiBranch.TraceEligibility end
function EpiBranch.is_eligible(::SymptomaticOver65, infector, contact, state)
    !EpiBranch.is_asymptomatic(infector) && get(infector.state, :age, 0) >= 65
end

# Build an infector Individual carrying the given state keys.
infector_with(; kwargs...) = Individual(id = 1, state = Dict{Symbol, Any}(kwargs...))

# None of the built-in policies read the contact or the state argument,
# so a bare contact and `nothing` state suffice.
const _CONTACT = Individual(id = 2, parent_id = 1)
elig(policy, infector) = is_eligible(policy, infector, _CONTACT, nothing)

@testset "Composable eligibility" begin
    @testset "Atomic predicates read one key each" begin
        symptomatic = infector_with(asymptomatic = false)
        asymptomatic = infector_with(asymptomatic = true)
        @test elig(OnSymptomOnset(), symptomatic)
        @test !elig(OnSymptomOnset(), asymptomatic)

        @test elig(OnLabConfirmation(), infector_with(test_positive = true))
        @test !elig(OnLabConfirmation(), infector_with(test_positive = false))
        # Atomic: lab confirmation does not also require symptoms.
        @test elig(OnLabConfirmation(), infector_with(test_positive = true, asymptomatic = true))

        @test elig(OnIsolation(), infector_with(isolated = true))
        @test !elig(OnIsolation(), infector_with(isolated = false))

        @test elig(TraceEveryone(), asymptomatic)
        @test !elig(TraceNobody(), symptomatic)
    end

    @testset "Boolean operators compose policies" begin
        infector = infector_with(asymptomatic = false, isolated = false, test_positive = true)

        @test elig(OnSymptomOnset() | OnLabConfirmation(), infector)   # OR
        @test !elig(OnIsolation() | TraceNobody(), infector)

        @test elig(OnSymptomOnset() & OnLabConfirmation(), infector)   # AND
        @test !elig(OnSymptomOnset() & OnIsolation(), infector)

        @test elig(!OnIsolation(), infector)                           # NOT
        @test !elig(!OnSymptomOnset(), infector)

        # "symptomatic, not yet isolated" — the old `Unless` use case.
        @test elig(OnSymptomOnset() & !OnIsolation(), infector)
        @test !elig(OnSymptomOnset() & !OnIsolation(),
            infector_with(asymptomatic = false, isolated = true))
    end

    @testset "Operators build the wrapper types" begin
        @test (OnSymptomOnset() | OnLabConfirmation()) isa AnyOf
        @test (OnSymptomOnset() & OnIsolation()) isa AllOf
        @test (OnSymptomOnset() & !OnIsolation()) isa AllOf
    end

    @testset "SymptomaticParent reproduces the original gate" begin
        @test elig(SymptomaticParent(), infector_with(asymptomatic = false, isolated = true))
        @test !elig(SymptomaticParent(), infector_with(asymptomatic = false, isolated = false))
        @test !elig(SymptomaticParent(), infector_with(asymptomatic = true, isolated = true))
        # Equivalent to the composed atomic form.
        composed = OnSymptomOnset() & OnIsolation()
        for p in (infector_with(asymptomatic = false, isolated = true),
            infector_with(asymptomatic = false, isolated = false),
            infector_with(asymptomatic = true, isolated = true))
            @test elig(SymptomaticParent(), p) == elig(composed, p)
        end
    end

    @testset "required_fields" begin
        @test required_fields(OnSymptomOnset()) == [:asymptomatic]
        @test required_fields(OnLabConfirmation()) == [:test_positive]
        @test required_fields(OnIsolation()) == [:isolated]
        @test required_fields(TraceEveryone()) == Symbol[]
        @test required_fields(TraceNobody()) == Symbol[]
        @test required_fields(SymptomaticParent()) == [:asymptomatic, :isolated]

        @test Set(required_fields(OnSymptomOnset() | OnIsolation())) ==
              Set([:asymptomatic, :isolated])
        @test required_fields(!OnLabConfirmation()) == [:test_positive]
        @test Set(required_fields(OnSymptomOnset() & !OnIsolation())) ==
              Set([:asymptomatic, :isolated])
    end

    @testset "Integration with ContactTracing constructors" begin
        # Terse positional form wraps probability/delay automatically.
        ct = ContactTracing(OnSymptomOnset(), 0.7, Exponential(1.5))
        @test ct.eligibility isa OnSymptomOnset
        @test ct.trace_rate isa ConstantRate
        @test ct.isolation_to_trace_delay isa ConstantDelay
        @test ct.action isa Quarantine

        ct2 = ContactTracing(OnSymptomOnset() | OnLabConfirmation(), 0.5, Exponential(2.0), FlagOnly())
        @test ct2.eligibility isa AnyOf
        @test ct2.action isa FlagOnly
        @test Set(required_fields(ct2)) == Set([:asymptomatic, :test_positive])

        # Keyword form keeps the original default eligibility.
        ct3 = ContactTracing(probability = 0.6, isolation_to_trace_delay = Exponential(1.0))
        @test ct3.eligibility isa SymptomaticParent
    end

    @testset "Custom eligibility still slots in by type" begin
        @test elig(SymptomaticOver65(), infector_with(asymptomatic = false, age = 70))
        @test !elig(SymptomaticOver65(), infector_with(asymptomatic = false, age = 30))
        # Custom policies compose with the operators too.
        @test elig(SymptomaticOver65() | OnIsolation(),
            infector_with(asymptomatic = true, isolated = true, age = 30))
    end
end
