# Test intervention exercising the `keep_active` seam: keep every
# uninfected contact active so it grows contacts of its own next generation.
struct KeepUninfectedActive <: EpiBranch.AbstractIntervention end
function EpiBranch.keep_active(::KeepUninfectedActive, state, targets, is_new)
    [t.id for t in targets if !is_infected(t)]
end

@testset "Competing risks" begin
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    model = BranchingProcess(Poisson(2.0), Exponential(5.0))

    @testset "Population susceptibility prevents overshoot" begin
        # With a hard population cap, cumulative cases should never
        # exceed population_size even with per-step Bernoulli draws,
        # because pop_suscept is recomputed per contact within the
        # resolution step.
        for seed in 1:5
            m = BranchingProcess(Poisson(3.0), Exponential(5.0); population_size = 50)
            state = simulate(m; max_cases = 500,
                rng = StableRNG(seed))
            @test state.cumulative_cases <= 50
        end
    end

    @testset "State-aware offspring caps per-parent contact count" begin
        # A gathering-limit-style cap expressed as a state-aware
        # offspring distribution. The (rng, ind, state) form lets it
        # depend on cumulative_cases.
        cap_after_20 = function (rng, ind, state)
            n = rand(rng, Poisson(5.0))
            return state.cumulative_cases >= 20 ? min(n, 2) : n
        end
        model_cap = BranchingProcess(cap_after_20, Exponential(5.0))

        rng = StableRNG(42)
        state = simulate(with_attributes(model_cap, clinical); max_cases = 200, rng = rng)
        # After the cap kicks in (cumulative_cases >= 20), every parent
        # should emit at most 2 contacts.
        for ind in state.individuals
            ind.id < 20 && continue
            @test length(ind.secondary_case_ids) <= 2
        end
    end

    @testset "MassVaccination at t=0 with full efficacy stops the outbreak at the index" begin
        # Every contact's immunity is in place by their transmission
        # time, so the engine blocks every secondary transmission.
        mv = MassVaccination(efficacy = 1.0, eligibility_time = 0.0,
            delay_to_immunity = 0.0)
        for seed in 1:5
            state = simulate(
                with_attributes(
                    with_interventions(BranchingProcess(Poisson(3.0), Exponential(5.0)), [mv]),
                    clinical);
                max_cases = 500,
                rng = StableRNG(seed))
            @test state.cumulative_cases == 1
        end
    end

    @testset "MassVaccination respects delay to immunity" begin
        # Vaccination eligible far in the future — no contact's
        # transmission should land after eligibility + delay, so the
        # outbreak should proceed unaffected.
        mv = MassVaccination(efficacy = 1.0, eligibility_time = 1.0e6,
            delay_to_immunity = 0.0)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(3.0), Exponential(5.0)), [mv]),
                clinical);
            condition = 50:500,
            max_cases = 500,
            rng = StableRNG(1))
        @test state.cumulative_cases >= 50
        # Every contact was marked vaccinated (eligibility was finite)
        # but none had immunity in time, so the risk never fires.
        @test all(ind.state[:vaccination_time] ≈ 1.0e6
        for ind in state.individuals if ind.parent_id != 0)
    end

    @testset "MassVaccination per-individual eligibility function" begin
        # Age-conditional rollout: 65+ become eligible at t=0, younger
        # at t=1e6. With efficacy=1 and delay=0, only 65+ contacts
        # should have transmissions blocked.
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        mv = MassVaccination(
            efficacy = 1.0,
            eligibility_time = (rng, ind) -> ind.state[:age] >= 65 ? 0.0 : 1.0e6,
            delay_to_immunity = 0.0
        )

        rng = StableRNG(42)
        state = simulate(with_attributes(with_interventions(model, [mv]), attrs); max_cases = 200, rng = rng)
        # No infected case should have age >= 65 (those got blocked).
        for ind in state.individuals
            ind.parent_id == 0 && continue
            is_infected(ind) || continue
            @test ind.state[:age] < 65
        end
    end

    @testset "MassVaccination with distributional efficacy" begin
        # Efficacy drawn from Beta(8,2) per individual. Use a far-future
        # eligibility so vaccination doesn't actually block any contacts
        # — we just want to check the per-contact sampling mechanism.
        mv = MassVaccination(efficacy = Beta(8, 2),
            eligibility_time = 1.0e6, delay_to_immunity = 0.0)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(2.0), Exponential(5.0)), [mv]),
                clinical);
            condition = 50:500,
            max_cases = 500,
            rng = StableRNG(1))
        effs = [ind.state[:vaccine_efficacy]
                for ind in state.individuals if ind.parent_id != 0]
        @test !isempty(effs)
        @test all(0 .<= effs .<= 1)
        # Variation confirms per-individual sampling rather than a
        # single sample reused across all contacts.
        @test length(unique(effs)) > 10
    end

    @testset "MassVaccination with callable efficacy reads contact state" begin
        # Age-conditional efficacy: high in <65, low in 65+.
        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        mv = MassVaccination(
            efficacy = (rng, ind) -> ind.state[:age] >= 65 ? 0.3 : 0.95,
            eligibility_time = 0.0, delay_to_immunity = 0.0
        )
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(2.0), Exponential(5.0)), [mv]),
                attrs);
            max_cases = 100,
            rng = StableRNG(2))
        for ind in state.individuals
            ind.parent_id == 0 && continue
            expected = ind.state[:age] >= 65 ? 0.3 : 0.95
            @test ind.state[:vaccine_efficacy] == expected
        end
    end

    @testset "Multi-dose MassVaccination composes via dose_label" begin
        # Two doses with different state namespaces. State keys are
        # suffixed; both doses contribute independent competing risks.
        prime = MassVaccination(efficacy = 1.0, eligibility_time = 0.0,
            delay_to_immunity = 0.0, dose_label = :prime)
        boost = MassVaccination(efficacy = 1.0, eligibility_time = 0.0,
            delay_to_immunity = 0.0, dose_label = :boost)
        state = simulate(
            with_attributes(
                with_interventions(BranchingProcess(Poisson(3.0), Exponential(5.0)), [
                    prime, boost]),
                clinical);
            max_cases = 200,
            rng = StableRNG(3))
        # The default :vaccinated key is untouched; dose-labelled
        # keys carry the state.
        for ind in state.individuals
            ind.parent_id == 0 && continue
            @test !haskey(ind.state, :vaccinated)
            @test ind.state[:vaccinated_prime] == true
            @test ind.state[:vaccinated_boost] == true
            @test ind.state[:vaccination_time_prime] == 0.0
            @test ind.state[:vaccination_time_boost] == 0.0
        end
        # Both doses block transmission with probability 1 from t=0,
        # so the outbreak should stop at the index.
        @test state.cumulative_cases == 1
    end

    @testset "Risk with callable block_probability sees parent and contact" begin
        struct AgeConditionalBlock <: AbstractIntervention
            threshold::Int
        end
        function EpiBranch.competing_risk(b::AgeConditionalBlock, parent, contact, state)
            Risk(event_time = -Inf,
                block_probability = (rng, parent, contact,
                    state) -> contact.state[:age] >= b.threshold ? 1.0 : 0.0)
        end

        attrs = compose(clinical, demographics(age_distribution = Uniform(0, 90)))
        rng = StableRNG(42)
        state = simulate(
            with_attributes(with_interventions(model, [AgeConditionalBlock(50)]), attrs);
            max_cases = 200, rng = rng)
        for ind in state.individuals
            ind.parent_id == 0 && continue
            is_infected(ind) || continue
            @test ind.state[:age] < 50
        end
    end

    @testset "Susceptibility and infectiousness are default risk sources on the surface" begin
        # Host susceptibility and parent infectiousness expose themselves
        # through the same `competing_risk` seam as interventions, as a
        # block probability `1 - trait`; trait == 1 contributes no risk.
        parent = Individual(id = 1, infectiousness = 0.6)
        contact = Individual(id = 2, parent_id = 1, susceptibility = 0.25)
        rs = EpiBranch.competing_risk(EpiBranch.HostSusceptibility(),
            parent, contact, nothing)
        @test rs isa Risk
        @test rs.block_probability ≈ 0.75
        ri = EpiBranch.competing_risk(EpiBranch.InfectorInfectiousness(),
            parent, contact, nothing)
        @test ri.block_probability ≈ 0.4
        # Default trait (1.0) ⇒ no risk contributed.
        plain = Individual(id = 3, parent_id = 1)
        @test EpiBranch.competing_risk(EpiBranch.HostSusceptibility(),
            parent, plain, nothing) === nothing
        @test EpiBranch.competing_risk(EpiBranch.InfectorInfectiousness(),
            Individual(id = 4), plain, nothing) === nothing

        # And they actually thin transmission: susceptibility 0.3 ⇒ ~30%
        # of contacts infected, end to end.
        m = BranchingProcess((rng, ind) -> 6, Exponential(5.0))
        attrs = transmission_traits(susceptibility = 0.3)
        frac = Float64[]
        for seed in 1:8
            s = simulate(with_attributes(m, attrs); n_initial = 50, max_generations = 1, rng = StableRNG(seed))
            kids = filter(i -> i.parent_id != 0, s.individuals)
            push!(frac, count(is_infected, kids) / length(kids))
        end
        @test 0.2 < sum(frac) / length(frac) < 0.4
    end

    @testset "keep_active grows uninfected contacts without infecting them" begin
        # ~half of each case's contacts go uninfected (susceptibility 0.5).
        m = BranchingProcess((rng, ind) -> 4, Exponential(5.0))
        attrs = transmission_traits(susceptibility = 0.5)
        opts = (; n_initial = 5, max_generations = 4)
        base = simulate(with_attributes(m, attrs); opts..., rng = StableRNG(1))
        grown = simulate(
            with_attributes(with_interventions(m, [KeepUninfectedActive()]), attrs);
            opts..., rng = StableRNG(1))

        # The uninfected fringe now spawns its own contacts: far more nodes.
        @test length(grown.individuals) > length(base.individuals)
        # But an uninfected source never transmits (`InfectiousSource`):
        # every contact of an uninfected parent is itself uninfected.
        for ind in grown.individuals
            ind.parent_id == 0 && continue
            is_infected(grown.individuals[ind.parent_id]) && continue
            @test !is_infected(ind)
        end
        # InfectiousSource itself: an uninfected source contributes a
        # full block; an infected one contributes nothing.
        well = Individual(id = 1)
        well.state[:infected] = false
        sick = Individual(id = 2)
        sick.state[:infected] = true
        c = Individual(id = 3, parent_id = 1)
        @test EpiBranch.competing_risk(EpiBranch.InfectiousSource(),
            well, c, nothing).block_probability == 1.0
        @test EpiBranch.competing_risk(EpiBranch.InfectiousSource(),
            sick, c, nothing) === nothing
    end
end
