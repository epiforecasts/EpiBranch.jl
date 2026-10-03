function test_continuous_vaccine_actions(make_process)
    @testset "Continuous-time action delivery" begin
        process = make_process(Exponential(2.0))
        clinical = clinical_presentation(incubation_period = Dirac(0.0))
        iso = Isolation(
            onset_to_isolation_delay = Dirac(0.0),
            post_isolation_transmission = 1.0
        )
        ct = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
            quarantine_on_trace = false
        )
        progression = [Transition(:recovered; delay = 3.0, terminal = true)]
        build(v) = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [iso, ct, v]
        )
        rv = RingVaccination(efficacy = 1.0)
        state = simulate(build(rv); rng = StableRNG(22))
        @test state.cumulative_cases == 1
        @test count(is_vaccinated, state.individuals) == 3
        for wrap in (
                v -> CapacityConstrained(Scheduled(v; start_time = 0.0); budget_per_period = 1.0),
                v -> Scheduled(CapacityConstrained(v; budget_per_period = 1.0); start_time = 0.0),
            )
            state = simulate(build(wrap(rv)); rng = StableRNG(22))
            @test count(is_vaccinated, state.individuals) == 1
        end
        gv = CapacityConstrained(GroupVaccination(efficacy = 1.0); budget_per_period = 2.0)
        grouped = ModelSpec(
            process; progression,
            attributes = [clinical, (rng, ind) -> (ind.state[:group] = 1)],
            interventions = [iso, gv]
        )
        state = simulate(grouped; rng = StableRNG(23))
        @test count(is_vaccinated, state.individuals) == 2
    end

    @testset "Post-exposure abort of a dose given while still pending" begin
        # A dose given by tracing while a member is still uninfected is
        # already on its state by the time its own infection settles; the
        # abort it can cause is decided then, against the exposure the race
        # has just fixed.
        clinical = clinical_presentation(
            incubation_period = Exponential(5.0), prob_asymptomatic = 0.0
        )
        ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
        progression = [Transition(:recovered; delay = 30.0, terminal = true)]
        process = make_process(Exponential(1.0))
        rv = RingVaccination(
            efficacy = 0.0, post_exposure_efficacy = 1.0,
            delay_to_immunity = Exponential(2.0)
        )
        @test EpiBranch.continuous_actions(rv)
        model = ModelSpec(
            process; progression, attributes = clinical,
            interventions = [ct, rv]
        )
        states = [simulate(model; rng = StableRNG(s)) for s in 1:30]
        individuals = vcat((s.individuals for s in states)...)
        @test count(is_vaccinated, individuals) > 0
        aborted = filter(ind -> haskey(ind.state, :infection_aborted_time), individuals)
        @test !isempty(aborted)
        @test all(ind -> is_infected(ind) && isnan(onset_time(ind)), aborted)
        @test_logs min_level = Base.CoreLogging.Warn simulate(model; rng = StableRNG(1))
    end
    return nothing
end
