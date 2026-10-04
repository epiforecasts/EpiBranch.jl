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
        # the susceptibility risk reaches the likelihood through
        # `susceptibility_components`
        @test compatible(MassVaccination(efficacy = 0.6, eligibility_time = 10.0))
        @test compatible(GroupVaccination(efficacy = 0.6))
        @test compatible(RingVaccination(efficacy = 0.6))
        # a labelled dose belongs to a schedule whose doses compose as
        # separate competing risks
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

    @testset "susceptibility effects of composed components" begin
        mv = MassVaccination(efficacy = 0.7, eligibility_time = 10.0)
        host(τ) = LayerHost(1, NaN, EpiBranch._LayerHostState((; immunity_time = [τ]), 1))
        effect = EpiBranch.susceptibility_components(EpiBranch.vaccine_effect(mv), host(2.0))
        @test EpiBranch.susceptibility_components(iso, host(2.0)) === nothing
        @test EpiBranch.susceptibility_components(mv, host(2.0)) == effect
        @test EpiBranch.susceptibility_components([iso, mv], host(2.0)) == effect
        @test EpiBranch.susceptibility_components(
            [Scheduled(mv; start_time = 0.0)], host(2.0)
        ) == effect
        # a host never vaccinated is left as it is
        @test EpiBranch.susceptibility_components([iso, mv], host(Inf)) === nothing
        # two components modifying one host are not combined silently
        @test_throws ArgumentError EpiBranch.susceptibility_components(
            [mv, GroupVaccination(efficacy = 0.5)], host(2.0)
        )
        # a layer that never recorded the time a vaccination reads says so
        no_times = LayerHost(1, NaN, EpiBranch._LayerHostState((;), 1))
        @test_throws ArgumentError EpiBranch.susceptibility_components(mv, no_times)

        @test EpiBranch.susceptibility_host_times(iso) == ()
        @test EpiBranch.susceptibility_host_times(mv) == (:immunity_time,)
        @test EpiBranch.susceptibility_host_times(
            CapacityConstrained(RingVaccination(efficacy = 0.6); budget_per_period = 2)
        ) == (:immunity_time,)
        labelled = MassVaccination(
            efficacy = 0.6, eligibility_time = 10.0, dose_label = :boost
        )
        @test EpiBranch.susceptibility_host_times(labelled) == (:immunity_time_boost,)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0)); interventions = [iso, mv, labelled]
        )
        @test EpiBranch._layer_host_time_keys(spec, (:onset_time, :immunity_time)) ==
            [:onset_time, :immunity_time, :immunity_time_boost]
    end
end
