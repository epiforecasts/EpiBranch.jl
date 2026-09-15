using EpiBranch: is_eligible, required_fields

# Custom eligibility used by the extension test below. Structs must be
# defined at top level, so it lives here rather than inside the testset.
struct SymptomaticOver65 <: EpiBranch.TraceEligibility end
function EpiBranch.is_eligible(::SymptomaticOver65, infector, contact, state)
    !EpiBranch.is_asymptomatic(infector) && get(infector.state, :age, 0) >= 65
end

# Custom eligibility that reads the contact instead of the infector.
struct ContactOver65 <: EpiBranch.TraceEligibility end
function EpiBranch.is_eligible(::ContactOver65, infector, contact, state)
    get(contact.state, :age, 0) >= 65
end

# Custom eligibility timed from the contact: half a day after its infection.
struct AfterContactInfected <: EpiBranch.TraceEligibility end
function EpiBranch.trigger_time(::AfterContactInfected, infector, contact, state)
    contact.infection_time + 0.5
end

# Custom trace action that records the trace time it receives.
struct RecordTraceTime <: EpiBranch.TraceAction end
function EpiBranch.apply_trace!(::RecordTraceTime, contact, state, trace_time, rng)
    contact.state[:recorded_trace_time] = trace_time
    return nothing
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

        # A negation that holds has no time of its own. Inside AllOf the other
        # conditions set the time. On its own it takes the default isolation
        # time, never isolated here, so it never triggers, the same as
        # TraceEveryone. Once its condition is met it never triggers either.
        unisolated = Individual(id = 1, infection_time = 1.0,
            state = Dict{Symbol, Any}(:asymptomatic => false, :onset_time => 4.0,
                :isolated => false))
        @test tt(!OnIsolation(), unisolated) == tt(TraceEveryone(), unisolated) == Inf
        @test tt(OnSymptomOnset() & !OnIsolation(), unisolated) == 4.0
        @test tt(!OnSymptomOnset(), unisolated) == Inf
        @test tt(OnSymptomOnset() & !OnIsolation(),
            infector_with(
                asymptomatic = false, onset_time = 4.0, isolated = true, isolation_time = 9.0)) ==
              Inf

        # With an isolation time, a holding negation on its own or inside AnyOf
        # triggers there, as TraceEveryone does.
        @test tt(!TraceNobody(), quarantined_negative) == 2.0
        @test tt(!OnLabConfirmation(), quarantined_negative) == 2.0
        @test tt(OnSymptomOnset() | !OnLabConfirmation(), quarantined_negative) ==
              tt(OnSymptomOnset() | TraceEveryone(), quarantined_negative) == 2.0
        @test tt(OnSymptomOnset() & !OnLabConfirmation(), quarantined_negative) == 4.0
        # An AllOf of negations only, or of nothing, also takes the default.
        @test tt(!OnLabConfirmation() & !TraceNobody(), quarantined_negative) == 2.0
        @test tt(AllOf(), quarantined_negative) == 2.0
        @test tt(!OnSymptomOnset() & !TraceNobody(), quarantined_negative) == Inf
        # An asymptomatic case that is never isolated or confirmed is never traced
        # through the negation branch.
        @test tt(OnLabConfirmation() | !OnSymptomOnset(), unconfirmed_asymptomatic) == Inf

        # A custom policy is timed at its trigger time (isolation by default)
        # when its `is_eligible` method says it is met, and skipped otherwise.
        custom = infector_with(asymptomatic = false, age = 70, onset_time = 4.0,
            test_positive = false, isolated = true, isolation_time = 3.0)
        @test tt(SymptomaticOver65() | OnSymptomOnset(), custom) == 3.0
        @test tt(SymptomaticOver65() & OnLabConfirmation(), custom) == Inf
        # Aged 40, so the custom condition is not met and quarantine at 2 does
        # not make the trace earlier than onset.
        young = infector_with(asymptomatic = false, age = 40, onset_time = 4.0,
            test_positive = false, isolated = true, isolation_time = 2.0)
        @test tt(SymptomaticOver65() | OnSymptomOnset(), young) == 4.0
        @test tt(SymptomaticOver65() & OnSymptomOnset(), young) == Inf

        # A policy that reads the contact is checked against the contact traced.
        function ttc(policy, contact_age)
            contact = Individual(id = 2, parent_id = 1, infection_time = 3.0,
                state = Dict{Symbol, Any}(:age => contact_age))
            return EpiBranch.trigger_time(policy, young, contact, nothing)
        end
        @test ttc(ContactOver65() | OnSymptomOnset(), 80) == 2.0
        @test ttc(ContactOver65() | OnSymptomOnset(), 30) == 4.0
        @test ttc(ContactOver65() & OnSymptomOnset(), 30) == Inf
        # A policy timed from the contact is timed with the contact traced.
        @test ttc(AfterContactInfected() | OnSymptomOnset(), 30) == 3.5
        @test ttc(AfterContactInfected() & OnSymptomOnset(), 30) == 4.0
        @test ttc(AfterContactInfected(), 30) == 3.5
        # Without a contact the atomic policies still time a combinator.
        @test EpiBranch.trigger_time(SymptomaticOver65() | OnSymptomOnset(), young, nothing) ==
              4.0
    end

    @testset "Contact tracing times the trace with the contact" begin
        ct = ContactTracing(AfterContactInfected() | TraceNobody(), ConstantRate(1.0),
            ConstantDelay(Dirac(0.0)), RecordTraceTime())
        state = simulate(
            ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [ct]);
            n_initial = 5, max_cases = 100, rng = StableRNG(3))
        traced = filter(ind -> haskey(ind.state, :recorded_trace_time), state.individuals)
        @test !isempty(traced)
        @test all(ind -> ind.state[:recorded_trace_time] == ind.infection_time + 0.5,
            traced)
    end

    @testset "Logically equal policies get the same trigger time" begin
        tt(policy, infector) = EpiBranch.trigger_time(policy, infector, nothing)
        S, L, I, N = OnSymptomOnset(), OnLabConfirmation(), OnIsolation(), TraceNobody()
        function case(; isolation = nothing, lab = false, asymptomatic = false)
            state = Dict{Symbol, Any}(:asymptomatic => asymptomatic,
                :onset_time => asymptomatic ? NaN : 4.0, :test_positive => lab,
                :isolated => isolation !== nothing)
            isolation === nothing || (state[:isolation_time] = isolation)
            return Individual(id = 1, infection_time = 1.0, state = state)
        end
        infectors = [case(), case(isolation = 9.0), case(isolation = 2.0),
            case(isolation = 6.0, lab = true), case(lab = true),
            case(isolation = 6.0, asymptomatic = true),
            case(isolation = 6.0, lab = true, asymptomatic = true)]
        equivalent = [
            # De Morgan
            (S & !(L & I), S & (!L | !I)),
            (S & !(L | I), S & (!L & !I)),
            (!(!(S & !I) | L), (S & !I) & !L),
            (S | !I, !(!S & I)),
            (!!S, AllOf(S)),
            # Distributing `&` over `|`
            (S & (L | !I), (S & L) | (S & !I)),
            (S & (L | !I) & (I | !L),
                (S & L & I) | (S & L & !L) | (S & !I & I) | (S & !I & !L)),
            ((S | !L) & !N, (S & !N) | (!L & !N)),
            (L | (S & !(L | (I & !S))), L | (S & !L & (!I | S))),
            # `TraceNobody()` is the identity of `|`
            (S & (N | !L), S & !L),
            (N | (S | !I), S | !I),
            # Double negation, order and grouping
            (!!(S | !I), S | !I),
            (!I | S, S | !I),
            (!L & S, S & !L),
            ((S | L) | !I, S | (L | !I)),
            ((S & !L) & !I, S & (!L & !I))
        ]
        for (a, b) in equivalent
            @test all(infector -> isequal(tt(a, infector), tt(b, infector)), infectors)
        end

        unisolated, isolated_late = infectors[1], infectors[2]
        @test tt(S & (L | !I), unisolated) == 4.0
        @test tt(S & (!L | !I), isolated_late) == 4.0
        @test tt(S & (N | !L), isolated_late) == 4.0
        # At the top level a policy met with no time of its own starts at the
        # earlier of its timed branches and the default isolation time.
        @test tt(S | !I, unisolated) == 4.0
        @test tt((S | !L) & !N, isolated_late) == 4.0
        @test tt(!L & !N, isolated_late) == 9.0

        # Logically equal policies that differ because a negation that holds
        # has no time of its own, as documented for `trigger_time`.
        @test tt(!(L & (!I | !S)), isolated_late) == 9.0
        @test tt(!((L & !I) | (L & !S)), isolated_late) == 4.0
        @test tt(!L | (!L & S), isolated_late) == 4.0
        @test tt(!L, isolated_late) == 9.0
        @test tt(S & (I | !I), isolated_late) == 9.0
        @test tt(S, isolated_late) == 4.0
        @test tt(S & !N, isolated_late) == 4.0
        @test tt(S & TraceEveryone(), isolated_late) == 9.0
        @test tt(!N, isolated_late) == tt(TraceEveryone(), isolated_late) == 9.0
    end

    @testset "Nested and flat policies trace alike" begin
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0), test_sensitivity = 0.5)
        S, L, I, N = OnSymptomOnset(), OnLabConfirmation(), OnIsolation(), TraceNobody()
        function simulate_with(eligibility, seed)
            ct = ContactTracing(eligibility, 1.0, Exponential(1.0))
            simulate(
                ModelSpec(BranchingProcess(Poisson(2.5), Exponential(5.0));
                    interventions = [iso, ct], attributes = clinical);
                n_initial = 5, max_cases = 300, rng = StableRNG(seed))
        end
        pairs = [(S & !L, S & (N | !L)),
            (S & !(L & I), S & (!L | !I)),
            ((S & L) | (S & !I), S & (L | !I))]
        for (flat, nested) in pairs
            for seed in 1:3
                a = simulate_with(flat, seed)
                b = simulate_with(nested, seed)
                @test count(is_traced, a.individuals) > 0
                @test a.cumulative_cases == b.cumulative_cases
                @test isequal(isolation_time.(a.individuals), isolation_time.(b.individuals))
            end
        end
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

    @testset "A negation on its own traces from the default time" begin
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0), test_sensitivity = 0.5)
        function simulate_with(eligibility, action = Quarantine())
            ct = ContactTracing(eligibility, ConstantRate(1.0),
                ConstantDelay(Exponential(1.0)), action)
            simulate(
                ModelSpec(BranchingProcess(Poisson(2.5), Exponential(5.0));
                    interventions = [iso, ct], attributes = clinical);
                n_initial = 5, max_cases = 300, rng = StableRNG(7))
        end

        everyone = simulate_with(TraceEveryone())
        negated = simulate_with(!TraceNobody())
        @test count(is_traced, negated.individuals) > 0
        @test negated.cumulative_cases == everyone.cumulative_cases
        @test isequal(isolation_time.(negated.individuals),
            isolation_time.(everyone.individuals))

        # Under `!OnIsolation()` only infectors not yet isolated are eligible,
        # and their contacts are traced no earlier than the infector's isolation.
        recorded = simulate_with(!OnIsolation(), RecordTraceTime())
        traced = filter(ind -> haskey(ind.state, :recorded_trace_time),
            recorded.individuals)
        @test !isempty(traced)
        @test all(traced) do ind
            infector = recorded.individuals[ind.parent_id]
            ind.state[:recorded_trace_time] >= isolation_time(infector)
        end
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
