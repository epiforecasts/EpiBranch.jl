# Custom TraceEligibility used by the user-extension test below.
struct WithinChain <: EpiBranchInterventions.TraceEligibility end
function EpiBranchInterventions.is_eligible(::WithinChain, parent, contact, state)
    parent.chain_id == contact.chain_id
end

@testset "ContactTracing trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default constructor lowers kwargs to trait form" begin
        ct = ContactTracing(probability = 0.5, delay = Exponential(1.0))
        @test ct.eligibility isa SymptomaticParent
        @test ct.trace_rate isa ConstantRate
        @test ct.trace_rate.p == 0.5
        @test ct.delay isa ConstantDelay
        @test ct.action isa Quarantine
    end

    @testset "quarantine_on_trace = false selects FlagOnly" begin
        ct = ContactTracing(probability = 1.0, delay = Exponential(0.1),
            quarantine_on_trace = false)
        @test ct.action isa FlagOnly
    end

    @testset "Default ContactTracing reproduces existing behaviour" begin
        # Tracing on isolated symptomatic parents, with quarantine.
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0, delay = Exponential(0.5))
        state = simulate(model;
            interventions = [iso, ct], attributes = clinical,
            sim_opts = SimOpts(max_cases = 50), rng = rng)
        # Some traces should fire: at least one quarantined contact.
        @test any(get(ind.state, :quarantined, false) for ind in state.individuals)
    end

    @testset "Custom TraceEligibility trait integrates via constructor" begin
        # User-defined eligibility slots in by type. We check the
        # struct accepts the custom trait and exposes it on the
        # intervention without any further hook changes.
        ct = ContactTracing(WithinChain(), ConstantRate(0.5),
            ConstantDelay(Exponential(1.0)), Quarantine())
        @test ct.eligibility isa WithinChain
        @test EpiBranchInterventions.is_eligible(ct.eligibility,
            Individual(id = 1, chain_id = 7),
            Individual(id = 2, chain_id = 7, parent_id = 1),
            nothing)
        @test !EpiBranchInterventions.is_eligible(ct.eligibility,
            Individual(id = 1, chain_id = 7),
            Individual(id = 2, chain_id = 8, parent_id = 1),
            nothing)
    end
end
