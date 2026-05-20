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
            state = simulate(m; sim_opts = SimOpts(max_cases = 500),
                rng = StableRNG(seed))
            @test state.cumulative_cases <= 50
        end
    end

    @testset "Tree-shaping via function-form offspring (state-aware)" begin
        # Express a "gathering limit" as a state-aware offspring
        # distribution rather than an intervention. The engine already
        # supports this via the (rng, ind, state) form.
        cap_after_20 = function (rng, ind, state)
            n = rand(rng, Poisson(5.0))
            return state.cumulative_cases >= 20 ? min(n, 2) : n
        end
        model_cap = BranchingProcess(cap_after_20, Exponential(5.0); n_types = 1)

        rng = StableRNG(42)
        state = simulate(model_cap; attributes = clinical,
            sim_opts = SimOpts(max_cases = 200), rng = rng)
        # After the cap kicks in (cumulative_cases >= 20), every parent
        # should emit at most 2 contacts.
        for ind in state.individuals
            ind.id < 20 && continue
            @test length(ind.secondary_case_ids) <= 2
        end
    end

    @testset "MassVaccination blocks every secondary infection at full efficacy from t=0" begin
        # With eligibility at t=0, no delay, and efficacy=1, every
        # contact's vaccination has immunity by their transmission
        # time, so every secondary transmission is blocked. The
        # outbreak should never get past the index case.
        mv = MassVaccination(efficacy = 1.0, eligibility_time = 0.0,
            delay_to_immunity = 0.0)
        for seed in 1:5
            state = simulate(BranchingProcess(Poisson(3.0), Exponential(5.0));
                interventions = [mv], attributes = clinical,
                sim_opts = SimOpts(max_cases = 500), rng = StableRNG(seed))
            @test state.cumulative_cases == 1
        end
    end

    @testset "MassVaccination respects delay to immunity" begin
        # Vaccination eligible far in the future — no contact's
        # transmission should land after eligibility + delay, so the
        # outbreak should proceed unaffected.
        mv = MassVaccination(efficacy = 1.0, eligibility_time = 1.0e6,
            delay_to_immunity = 0.0)
        state = simulate(BranchingProcess(Poisson(3.0), Exponential(5.0));
            condition = 50:500,
            interventions = [mv], attributes = clinical,
            sim_opts = SimOpts(max_cases = 500), rng = StableRNG(1))
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
        state = simulate(model;
            interventions = [mv], attributes = attrs,
            sim_opts = SimOpts(max_cases = 200), rng = rng)
        # No infected case should have age >= 65 (those got blocked).
        for ind in state.individuals
            ind.parent_id == 0 && continue
            is_infected(ind) || continue
            @test ind.state[:age] < 65
        end
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
        state = simulate(model;
            interventions = [AgeConditionalBlock(50)], attributes = attrs,
            sim_opts = SimOpts(max_cases = 200), rng = rng)
        for ind in state.individuals
            ind.parent_id == 0 && continue
            is_infected(ind) || continue
            @test ind.state[:age] < 50
        end
    end
end
