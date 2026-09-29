# A household clique exposes a susceptible to the same infector more than
# once, so this is exactly the setting the pairwise likelihood's vaccine
# argument is for: a leaky discount and an all-or-nothing mixture both need to
# see every exposure of a susceptible together, not pair by pair.
@testset "Vaccinated households: simulate → loglikelihood round trip" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    # Vaccinate as soon as the index case shows symptoms, without isolating or
    # quarantining anyone, so a vaccinated member keeps meeting its infector.
    ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
    progression = [Transition(:recovered; delay = 15.0, terminal = true)]

    @testset "the structured likelihood matches an explicit vaccine argument" begin
        rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        process = HouseholdProcess(fill(6, 200), Exponential(0.3))
        m = ModelSpec(process; progression, attributes = clinical,
            interventions = [ct, rv])
        state = simulate(m; rng = StableRNG(31))
        data = household_infections(state, m)

        # immunity times round-trip from the simulated state
        @test data.immunity_time == EpiBranch.immunity_time.(state.individuals)
        @test any(isfinite, data.immunity_time)

        explicit = pairwise_surv_loglik(m.process.kernel, data;
            external_hazard = m.process.external_hazard,
            vaccine = EpiBranch.vaccine_effect(rv))
        @test loglikelihood(data, m) ≈ explicit

        # scoring the same data as if nobody were vaccinated gives a
        # different value: the fix changes the number, not just the plumbing.
        unvaccinated = ModelSpec(process; progression, attributes = clinical,
            interventions = [ct])
        @test loglikelihood(data, m) != loglikelihood(data, unvaccinated)
    end

    @testset "AllOrNothingMode round-trips through the structured likelihood too" begin
        rv = RingVaccination(efficacy = 0.4, mode = AllOrNothingMode())
        process = HouseholdProcess(fill(6, 200), Exponential(0.3))
        m = ModelSpec(process; progression, attributes = clinical,
            interventions = [ct, rv])
        state = simulate(m; rng = StableRNG(32))
        data = household_infections(state, m)
        explicit = pairwise_surv_loglik(m.process.kernel, data;
            external_hazard = m.process.external_hazard,
            vaccine = EpiBranch.vaccine_effect(rv))
        @test loglikelihood(data, m) ≈ explicit
        @test isfinite(loglikelihood(data, m))
    end

    @testset "a vaccination with an onward or post-exposure effect is rejected" begin
        rv = RingVaccination(efficacy = 0.5, onward_efficacy = 0.3)
        process = HouseholdProcess(fill(6, 20), Exponential(0.3))
        m = ModelSpec(process; progression, attributes = clinical,
            interventions = [ct, rv])
        state = simulate(m; rng = StableRNG(33))
        data = household_infections(state, m)
        @test_throws ArgumentError loglikelihood(data, m)
    end
end
