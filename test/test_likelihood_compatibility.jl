@testset "Infection likelihood compatibility" begin
    compatible = EpiBranch.infection_likelihood_compatible
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
    clinical = clinical_presentation(incubation_period = Exponential(1.0))
    @test compatible(NoAttributes())
    @test compatible(clinical)
    @test compatible([clinical])
    @test compatible((clinical, NoAttributes()))
    @test !compatible((rng, ind) -> nothing)
    @test !compatible(transmission_traits(susceptibility = 0.5))
    @test compatible(Transition(:recovered; delay = 1.0))
    @test compatible(Reporting(delay = 1.0))
    @test compatible(Hospitalisation(delay = 1.0))
    @test compatible(Recovery(delay = 1.0))
    @test compatible(Death(delay = 1.0, probability = 0.1))
    @test compatible(iso)
    @test !compatible(
        Isolation(
            onset_to_isolation_delay = Exponential(1.0),
            post_isolation_transmission = 0.5
        )
    )
    for quarantine in (false, true)
        @test compatible(
            ContactTracing(
                probability = 0.5,
                isolation_to_trace_delay = Exponential(1.0), quarantine_on_trace = quarantine
            )
        )
    end
    @test compatible(Scheduled(iso; start_time = 0.0))
    @test compatible(CapacityConstrained(iso; budget_per_period = 2))
    model = ModelSpec(
        BranchingProcess(Poisson(0.0)); attributes = clinical,
        interventions = [iso], progression = [Recovery(delay = 1.0)]
    )
    @test EpiBranch._validate_infection_likelihood(model) === nothing
    bad = ModelSpec(BranchingProcess(Poisson(0.0)); attributes = (rng, ind) -> nothing)
    @test_throws ArgumentError EpiBranch._validate_infection_likelihood(bad)

    @testset "vaccination" begin
        # the basic susceptibility risk is fully represented by
        # `pairwise_surv_loglik`'s `vaccine` argument
        @test compatible(MassVaccination(efficacy = 0.6, eligibility_time = 10.0))
        @test compatible(GroupVaccination(efficacy = 0.6))
        @test compatible(RingVaccination(efficacy = 0.6))
        # a non-default dose label has no immunity time to read
        @test !compatible(
            MassVaccination(
                efficacy = 0.6, eligibility_time = 10.0, dose_label = :boost
            )
        )
        @test !compatible(GroupVaccination(efficacy = 0.6, dose_label = :boost))
        # `onward_efficacy`/`post_exposure_efficacy` act on hazards the
        # likelihood's kernel never sees
        @test !compatible(RingVaccination(efficacy = 0.6, onward_efficacy = 0.3))
        @test !compatible(RingVaccination(efficacy = 0.6, post_exposure_efficacy = 0.3))
        @test compatible(
            Scheduled(
                MassVaccination(efficacy = 0.6, eligibility_time = 10.0); start_time = 0.0
            )
        )
    end

    @testset "_model_vaccine" begin
        mv = MassVaccination(efficacy = 0.7, eligibility_time = 10.0)
        @test EpiBranch._model_vaccine([iso]) === nothing
        @test EpiBranch._model_vaccine([iso, mv]) === EpiBranch.vaccine_effect(mv)
        @test EpiBranch._model_vaccine([Scheduled(mv; start_time = 0.0)]) ===
            EpiBranch.vaccine_effect(mv)
        @test_throws ArgumentError EpiBranch._model_vaccine(
            [
                mv,
                GroupVaccination(efficacy = 0.5),
            ]
        )
    end
end
