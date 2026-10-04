# A household clique exposes a susceptible to the same infector more than
# once, so this is exactly the setting the pairwise likelihood's
# susceptibility effect is for: a leaky discount and an all-or-nothing mixture
# both need to see every exposure of a susceptible together, not pair by pair.

# An intervention written outside the package whose only declared effect is on
# susceptibility: every hazard is halved from `from` on. It names a host time
# it reads so the extraction can be checked to record it.
struct _HalvedFrom <: EpiBranch.AbstractIntervention
    from::Float64
end
function EpiBranch.susceptibility_components(h::_HalvedFrom, host)
    return (1 => EpiBranch.HazardScaling(h.from, 0.5),)
end
EpiBranch.susceptibility_host_times(::_HalvedFrom) = (:onset_time,)
EpiBranch.infection_likelihood_compatible(::_HalvedFrom) = true

@testset "Vaccinated households: simulate → loglikelihood round trip" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    # Vaccinate as soon as the index case shows symptoms, without isolating or
    # quarantining anyone, so a vaccinated member keeps meeting its infector.
    ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
    progression = [Transition(:recovered; delay = 15.0, terminal = true)]

    @testset "the structured likelihood matches an explicit vaccine effect" begin
        rv = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        process = HouseholdProcess(fill(6, 200), Exponential(0.3))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; rng = StableRNG(31))
        data = household_infections(state, m)

        # immunity times are recorded from the simulated state because the
        # model composes a vaccination
        recorded = coalesce.(data.host_times.immunity_time, Inf)
        @test recorded == EpiBranch.immunity_time.(state.individuals)
        @test any(isfinite, recorded)
        # a model without one records no host times
        @test isempty(household_infections(state, ModelSpec(process)).host_times)

        explicit = pairwise_surv_loglik(
            m.process.kernel, data;
            external_hazard = m.process.external_hazard,
            susceptibility = EpiBranch.vaccine_effect(rv)
        )
        @test loglikelihood(data, m) ≈ explicit

        # evaluating the same data as if nobody were vaccinated gives a
        # different value
        unvaccinated = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct]
        )
        @test loglikelihood(data, m) != loglikelihood(data, unvaccinated)
    end

    @testset "AllOrNothingMode round-trips through the structured likelihood too" begin
        rv = RingVaccination(efficacy = 0.4, mode = AllOrNothingMode())
        process = HouseholdProcess(fill(6, 200), Exponential(0.3))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; rng = StableRNG(32))
        data = household_infections(state, m)
        explicit = pairwise_surv_loglik(
            m.process.kernel, data;
            external_hazard = m.process.external_hazard,
            susceptibility = EpiBranch.vaccine_effect(rv)
        )
        @test loglikelihood(data, m) ≈ explicit
        @test isfinite(loglikelihood(data, m))
    end

    @testset "a vaccination with an onward or post-exposure effect is rejected" begin
        rv = RingVaccination(efficacy = 0.5, onward_efficacy = 0.3)
        process = HouseholdProcess(fill(6, 20), Exponential(0.3))
        m = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        state = simulate(m; rng = StableRNG(33))
        data = household_infections(state, m)
        @test_throws ArgumentError loglikelihood(data, m)
    end

    @testset "a user-defined susceptibility effect reaches the structured likelihood" begin
        process = HouseholdProcess(fill(6, 50), Exponential(0.3))
        plain = ModelSpec(process; progression, attributes = clinical)
        state = simulate(plain; rng = StableRNG(34))
        halved = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [_HalvedFrom(2.0)]
        )
        data = household_infections(state, halved)
        @test haskey(data.host_times, :onset_time)
        explicit = pairwise_surv_loglik(
            process.kernel, data;
            susceptibility = _HalvedFrom(2.0)
        )
        @test loglikelihood(data, halved) ≈ explicit
        @test loglikelihood(data, halved) != loglikelihood(data, plain)
    end
end
