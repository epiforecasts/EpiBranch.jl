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
        @test required_fields(OnSymptomOnset()) == [:asymptomatic, :onset_time]
        @test required_fields(OnLabConfirmation()) == [:test_positive]
        @test required_fields(OnIsolation()) == [:isolated]
        @test required_fields(TraceEveryone()) == Symbol[]
        @test required_fields(TraceNobody()) == Symbol[]
        @test required_fields(SymptomaticParent()) == [:asymptomatic, :isolated]

        @test Set(required_fields(OnSymptomOnset() | OnIsolation())) ==
              Set([:asymptomatic, :onset_time, :isolated])
        @test required_fields(!OnLabConfirmation()) == [:test_positive]
        @test Set(required_fields(OnSymptomOnset() & !OnIsolation())) ==
              Set([:asymptomatic, :onset_time, :isolated])
    end

    @testset "Trigger time follows the eligibility condition" begin
        tt(policy, infector) = EpiBranch.trigger_time(policy, infector, nothing)
        infector = infector_with(asymptomatic = false, onset_time = 4.0,
            isolated = true, isolation_time = 9.0)
        # OnSymptomOnset times from onset; isolation/default policies from isolation.
        @test tt(OnSymptomOnset(), infector) == 4.0
        @test tt(OnIsolation(), infector) == 9.0
        @test tt(SymptomaticParent(), infector) == 9.0
        # Combinators: AnyOf fires at the earliest trigger, AllOf at the latest.
        @test tt(OnSymptomOnset() | OnIsolation(), infector) == 4.0
        @test tt(OnSymptomOnset() & OnIsolation(), infector) == 9.0
    end

    @testset "Combinators time only the conditions the infector meets" begin
        tt(policy, infector) = EpiBranch.trigger_time(policy, infector, nothing)
        # Asymptomatic, so onset is NaN, but confirmed and isolated at 6.
        confirmed_asymptomatic = infector_with(asymptomatic = true, onset_time = NaN,
            test_positive = true, isolated = true, isolation_time = 6.0)
        @test tt(OnSymptomOnset() | OnLabConfirmation(), confirmed_asymptomatic) == 6.0
        @test tt(OnLabConfirmation() | OnSymptomOnset(), confirmed_asymptomatic) == 6.0
        @test tt(TraceEveryone() | OnSymptomOnset(), confirmed_asymptomatic) == 6.0
        # AllOf needs every condition, and onset never happens.
        @test tt(OnSymptomOnset() & OnLabConfirmation(), confirmed_asymptomatic) == Inf

        # An unmet condition must not make the time earlier either. This
        # infector was quarantined at 2, before onset at 4, and never tested positive.
        quarantined_negative = infector_with(asymptomatic = false, onset_time = 4.0,
            test_positive = false, isolated = true, isolation_time = 2.0)
        @test tt(OnLabConfirmation() | OnSymptomOnset(), quarantined_negative) == 4.0
        @test tt(TraceNobody() | OnSymptomOnset(), quarantined_negative) == 4.0

        # AnyOf with nothing met never triggers.
        unconfirmed_asymptomatic = infector_with(asymptomatic = true, onset_time = NaN,
            test_positive = false, isolated = false)
        @test tt(OnSymptomOnset() | OnLabConfirmation(), unconfirmed_asymptomatic) == Inf

        # A negation holds from infection, so inside AllOf the other
        # conditions set the time. Once its condition is met it never triggers.
        unisolated = Individual(id = 1, infection_time = 1.0,
            state = Dict{Symbol, Any}(:asymptomatic => false, :onset_time => 4.0,
                :isolated => false))
        @test tt(!OnIsolation(), unisolated) == 1.0
        @test tt(OnSymptomOnset() & !OnIsolation(), unisolated) == 4.0
        @test tt(!OnSymptomOnset(), unisolated) == Inf
        @test tt(OnSymptomOnset() & !OnIsolation(),
            infector_with(
                asymptomatic = false, onset_time = 4.0, isolated = true, isolation_time = 9.0)) ==
              Inf

        # A custom policy may depend on the contact, so it counts as met at its
        # trigger time (isolation by default). Built-in conditions beside it are checked.
        custom = infector_with(asymptomatic = false, age = 70, onset_time = 4.0,
            test_positive = false, isolated = true, isolation_time = 3.0)
        @test tt(SymptomaticOver65() | OnSymptomOnset(), custom) == 3.0
        @test tt(SymptomaticOver65() & OnLabConfirmation(), custom) == Inf
    end

    @testset "Tracing through a non-onset branch of AnyOf" begin
        # Asymptomatic cases have a NaN onset, so their contacts can be traced
        # only through the lab-confirmation branch.
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.5)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases())
        ct = ContactTracing(OnSymptomOnset() | OnLabConfirmation(), 1.0, Exponential(1.0))
        state = simulate(
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso, ct], attributes = clinical);
            n_initial = 20, max_cases = 300, rng = StableRNG(42))

        traced = filter(is_traced, state.individuals)
        @test !isempty(traced)
        @test !any(ind -> isnan(isolation_time(ind)), traced)
        via_asymptomatic = filter(
            ind -> is_asymptomatic(state.individuals[ind.parent_id]), traced)
        @test count(ind -> isfinite(isolation_time(ind)), via_asymptomatic) > 0
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
        @test Set(required_fields(ct2)) == Set([:asymptomatic, :onset_time, :test_positive])

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
