# Custom TraceEligibility used by the user-extension test below.
struct WithinChain <: EpiBranch.TraceEligibility end
function EpiBranch.is_eligible(::WithinChain, infector, contact, state)
    infector.chain_id == contact.chain_id
end

@testset "ContactTracing trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default constructor lowers kwargs to trait form" begin
        ct = ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(1.0))
        @test ct.eligibility isa SymptomaticParent
        @test ct.trace_rate isa ConstantRate
        @test ct.trace_rate.p == 0.5
        @test ct.isolation_to_trace_delay isa ConstantDelay
        @test ct.action isa Quarantine
    end

    @testset "quarantine_on_trace = false selects FlagOnly" begin
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.1),
            quarantine_on_trace = false)
        @test ct.action isa FlagOnly
    end

    @testset "Default ContactTracing reproduces existing behaviour" begin
        # Tracing on isolated symptomatic parents, with quarantine.
        rng = StableRNG(42)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
        state = simulate(
            BranchingProcess(Poisson(2.0), Exponential(5.0);
                interventions = [iso, ct], attributes = clinical);
            max_cases = 50, rng = rng)
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
        @test EpiBranch.is_eligible(ct.eligibility,
            Individual(id = 1, chain_id = 7),
            Individual(id = 2, chain_id = 7, parent_id = 1),
            nothing)
        @test !EpiBranch.is_eligible(ct.eligibility,
            Individual(id = 1, chain_id = 7),
            Individual(id = 2, chain_id = 8, parent_id = 1),
            nothing)
    end
end

@testset "ContactTracing ring depth" begin
    # All cases symptomatic, so an infected case always seeds a ring.
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)

    @testset "depth defaults to 1" begin
        @test ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5)).depth == 1
        @test ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5)).depth == 1
    end

    @testset "depth is settable through every constructor" begin
        @test ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 2).depth == 2
        @test ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5), depth = 3).depth == 3
        @test ContactTracing(OnSymptomOnset(), ConstantRate(1.0),
            ConstantDelay(Exponential(0.5)), Quarantine(); depth = 2).depth == 2
    end

    @testset "depth below 1 is rejected" begin
        @test_throws ArgumentError ContactTracing(
            OnSymptomOnset(), 1.0, Exponential(0.5); depth = 0)
        @test_throws ArgumentError ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(0.5), depth = -1)
        @test_throws ArgumentError ContactTracing(OnSymptomOnset(), ConstantRate(1.0),
            ConstantDelay(Exponential(0.5)), Quarantine(); depth = 0)
    end

    # Offspring-as-ring with susceptibility 0.5: each case has four
    # contacts, about half of which go uninfected. That uninfected fringe
    # is what a level-2 ring has to reach past.
    attrs = compose(clinical, transmission_traits(susceptibility = 0.5))
    opts = (; n_initial = 3, max_generations = 4)

    @testset "depth 1 traces direct contacts only; the fringe does not grow" begin
        ct = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 1)
        state = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct], attributes = attrs);
            opts..., rng = StableRNG(1))
        # No uninfected contact ever generated contacts of its own.
        for ind in state.individuals
            ind.parent_id == 0 && continue
            parent = state.individuals[ind.parent_id]
            @test is_infected(parent)
        end
    end

    @testset "depth 2 reaches contacts-of-contacts past the uninfected fringe" begin
        ct1 = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 1)
        ct2 = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 2)
        s1 = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct1], attributes = attrs);
            opts..., rng = StableRNG(1))
        s2 = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct2], attributes = attrs);
            opts..., rng = StableRNG(1))

        # The uninfected fringe now grows its own contacts: more nodes.
        @test length(s2.individuals) > length(s1.individuals)

        # Some traced contact has an uninfected parent: a contact-of-
        # contact the ring could only reach by growing past the fringe.
        reached = false
        for ind in s2.individuals
            ind.parent_id == 0 && continue
            parent = s2.individuals[ind.parent_id]
            if !is_infected(parent) && is_traced(ind)
                reached = true
            end
            # An uninfected source never infects its contacts.
            is_infected(parent) || @test !is_infected(ind)
        end
        @test reached
    end

    @testset "the ring is bounded by depth" begin
        # depth 2: a contact two hops from any infected case (its parent
        # is an uninfected ring member) carries no remaining budget, so
        # the ring stops there rather than running away.
        ct = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 2)
        state = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct], attributes = attrs);
            n_initial = 3, max_generations = 6, rng = StableRNG(7))
        for ind in state.individuals
            ind.parent_id == 0 && continue
            parent = state.individuals[ind.parent_id]
            # A contact of an uninfected ring member is the outer edge:
            # its own ring budget is exhausted.
            if !is_infected(parent) && is_traced(ind)
                @test get(ind.state, :ring_remaining, 0) == 0
            end
        end
        @test state.cumulative_cases >= 1
    end

    @testset "RingVaccination vaccinates the depth-2 ring" begin
        ct = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 2)
        rv = RingVaccination(efficacy = 0.9)
        state = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct, rv], attributes = attrs);
            opts..., rng = StableRNG(3))
        # At least one vaccinated contact sits past the fringe (its parent
        # was never infected): the level-2 ring delivered doses there.
        outer_vaccinated = false
        for ind in state.individuals
            ind.parent_id == 0 && continue
            parent = state.individuals[ind.parent_id]
            if !is_infected(parent) && is_vaccinated(ind)
                outer_vaccinated = true
            end
        end
        @test outer_vaccinated
    end
end
