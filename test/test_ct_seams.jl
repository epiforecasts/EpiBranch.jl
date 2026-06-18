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

@testset "traced_by and compute_trace_level!" begin
    @testset "compute_trace_level! walks traced_by back to the index" begin
        mini(inds) = SimulationState(
            inds, Int[], 0, StableRNG(1), 0, false, nothing, Inf, nothing,
            AbstractClinicalTransition[])

        # 1 (seed) ← 2 ← 3 ; 1 ← 4 ; 5 never traced.
        inds = [Individual(id = i) for i in 1:5]
        inds[2].state[:traced_by] = 1
        inds[3].state[:traced_by] = 2
        inds[4].state[:traced_by] = 1
        state = mini(inds)
        @test compute_trace_level!(state) === state
        @test inds[1].state[:trace_level] == 0       # anchor: untraced but referenced
        @test inds[2].state[:trace_level] == 1
        @test inds[3].state[:trace_level] == 2
        @test inds[4].state[:trace_level] == 1
        @test !haskey(inds[5].state, :trace_level)   # never traced, never an anchor

        # A defensive cycle must terminate rather than recurse forever.
        cyc = [Individual(id = i) for i in 1:2]
        cyc[1].state[:traced_by] = 2
        cyc[2].state[:traced_by] = 1
        @test compute_trace_level!(mini(cyc)) isa SimulationState
    end

    @testset "batch overload stamps every state" begin
        mini(inds) = SimulationState(
            inds, Int[], 0, StableRNG(1), 0, false, nothing, Inf, nothing,
            AbstractClinicalTransition[])
        a = [Individual(id = i) for i in 1:2]
        a[2].state[:traced_by] = 1
        b = [Individual(id = i) for i in 1:2]
        b[2].state[:traced_by] = 1
        states = [mini(a), mini(b)]
        @test compute_trace_level!(states) === states
        @test a[2].state[:trace_level] == 1
        @test b[2].state[:trace_level] == 1
    end

    @testset "reset! clears traced_by and trace_level" begin
        ct = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(1.0))
        ind = Individual(id = 1)
        ind.state[:traced] = true
        ind.state[:traced_by] = 7
        ind.state[:trace_level] = 2
        EpiBranch.reset!(ct, ind)
        @test ind.state[:traced] == false
        @test !haskey(ind.state, :traced_by)
        @test !haskey(ind.state, :trace_level)
    end

    @testset "trace_level lines up with the ring on a tree sim" begin
        clinical = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
        attrs = compose(clinical, transmission_traits(susceptibility = 0.5))
        ct = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5); depth = 2)
        state = simulate(
            BranchingProcess((rng, ind) -> 4, Exponential(5.0);
                interventions = [ct], attributes = attrs);
            n_initial = 3, max_generations = 4, rng = StableRNG(1))

        # Every traced node records the infector it was traced from; on a
        # branching process that is its (final) parent.
        for ind in state.individuals
            is_traced(ind) || continue
            @test get(ind.state, :traced_by, nothing) == ind.parent_id
        end

        compute_trace_level!(state)
        levels = collect(skipmissing(
            get(ind.state, :trace_level, missing) for ind in state.individuals))
        @test 0 in levels        # the index/anchor
        @test 1 in levels        # directly traced contacts
        @test 2 in levels        # contacts-of-contacts (the depth-2 reach)
        # Every traced node carries a positive level; only the anchor is 0.
        for ind in state.individuals
            is_traced(ind) || continue
            @test get(ind.state, :trace_level, 0) >= 1
        end

        # It flows into the linelist automatically.
        df = linelist(state)
        @test :traced_by in propertynames(df)
        @test :trace_level in propertynames(df)
    end
end
