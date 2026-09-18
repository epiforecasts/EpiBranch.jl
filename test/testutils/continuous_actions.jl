function test_continuous_vaccine_actions(make_process)
    @testset "Continuous-time action delivery" begin
        process = make_process(Exponential(2.0))
        clinical = clinical_presentation(incubation_period = Dirac(0.0))
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0),
            post_isolation_transmission = 1.0)
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
            quarantine_on_trace = false)
        progression = [Transition(:recovered; delay = 3.0, terminal = true)]
        build(v) = ModelSpec(process; progression, attributes = clinical,
            interventions = [iso, ct, v])
        rv = RingVaccination(efficacy = 1.0)
        state = simulate(build(rv); rng = StableRNG(22))
        @test state.cumulative_cases == 1
        @test count(is_vaccinated, state.individuals) == 3
        for wrap in (
            v -> CapacityConstrained(Scheduled(v; start_time = 0.0); budget_per_period = 1.0),
            v -> Scheduled(CapacityConstrained(v; budget_per_period = 1.0); start_time = 0.0))
            state = simulate(build(wrap(rv)); rng = StableRNG(22))
            @test count(is_vaccinated, state.individuals) == 1
        end
        gv = CapacityConstrained(GroupVaccination(efficacy = 1.0); budget_per_period = 2.0)
        grouped = ModelSpec(process; progression,
            attributes = [clinical, (rng, ind) -> (ind.state[:group] = 1)],
            interventions = [iso, gv])
        state = simulate(grouped; rng = StableRNG(23))
        @test count(is_vaccinated, state.individuals) == 2
    end
end
