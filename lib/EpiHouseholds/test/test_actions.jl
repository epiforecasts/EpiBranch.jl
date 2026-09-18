include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "continuous_actions.jl"))
test_continuous_vaccine_actions(k -> HouseholdProcess([4], k))

@testset "Capacity across separate household races" begin
    process = HouseholdProcess([3, 3], (i, j) -> Dirac(2.0))
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    iso = Isolation(onset_to_isolation_delay = Dirac(1.0),
        post_isolation_transmission = 1.0)
    ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
        quarantine_on_trace = false)
    rv = RingVaccination(efficacy = 0.0, dose_delay = 2.0)
    build(p, v) = ModelSpec(p; attributes = clinical,
        progression = [Transition(:recovered; delay = 10.0, terminal = true)],
        interventions = [iso, ct, v])
    for carry_over in (false, true),
        wrap in (
            v -> CapacityConstrained(Scheduled(v; start_after_cases = 2);
            budget_per_period = 1.0, period = 1.0, carry_over),
            v -> Scheduled(
            CapacityConstrained(v; budget_per_period = 1.0,
                period = 1.0, carry_over);
            start_after_cases = 2))

        @test_throws ArgumentError simulate(build(process, wrap(rv)); rng = StableRNG(32))
        state = simulate(build(HouseholdProcess([3], (i, j) -> Dirac(2.0)), wrap(rv)); rng = StableRNG(32))
        @test count(is_vaccinated, state.individuals) >= 1
    end
    lifetime = CapacityConstrained(rv; budget_per_period = 1.0, period = Inf)
    state = simulate(build(process, lifetime); rng = StableRNG(32))
    @test count(is_vaccinated, state.individuals) == 1
    @test state.cumulative_cases == 6
    nested = CapacityConstrained(
        Scheduled(
            CapacityConstrained(rv; budget_per_period = 1.0, period = 1.0); start_time = 0.0);
        budget_per_period = 1.0)
    @test_throws ArgumentError simulate(build(process, nested); rng = StableRNG(32))
end
