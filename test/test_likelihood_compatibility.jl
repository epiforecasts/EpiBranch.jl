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
    @test !compatible(Isolation(onset_to_isolation_delay = Exponential(1.0),
        post_isolation_transmission = 0.5))
    for quarantine in (false, true)
        @test compatible(ContactTracing(probability = 0.5,
            isolation_to_trace_delay = Exponential(1.0), quarantine_on_trace = quarantine))
    end
    @test compatible(Scheduled(iso; start_time = 0.0))
    @test compatible(CapacityConstrained(iso; budget_per_period = 2))
    model = ModelSpec(BranchingProcess(Poisson(0.0)); attributes = clinical,
        interventions = [iso], progression = [Recovery(delay = 1.0)])
    @test EpiBranch._validate_infection_likelihood(model) === nothing
    bad = ModelSpec(BranchingProcess(Poisson(0.0)); attributes = (rng, ind) -> nothing)
    @test_throws ArgumentError EpiBranch._validate_infection_likelihood(bad)
end
