# A custom transition defined outside the testset (struct definitions
# must be top-level) so we can exercise the public extension interface
# end-to-end below.
struct DummyTest <: AbstractClinicalTransition
    delay::Distribution
end
EpiBranchCore.required_fields(::DummyTest) = [:onset_time]
function EpiBranchCore.initialise_individual!(::DummyTest, ind, state)
    ind.state[:tested] = false
    ind.state[:test_time] = Inf
    return nothing
end
function EpiBranchCore.resolve_individual!(t::DummyTest, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    ind.state[:tested] = true
    ind.state[:test_time] = ot + rand(state.rng, t.delay)
    return nothing
end

# Minimal custom TransmissionModel used to verify the engine handles
# bookkeeping, competing-risks resolution, and clinical-transition
# resolution for new individuals, leaving step! to just describe the
# transmission decision. A single deterministic infection per call.
struct SingleSpawnModel <: EpiBranchCore.TransmissionModel end
function EpiBranchCore.step!(::SingleSpawnModel, state::EpiBranchCore.SimulationState)
    new_contacts = EpiBranchCore.Individual[]
    parent = state.individuals[state.active_ids[1]]
    make_contact!(new_contacts, state, parent, parent.infection_time + 1.0)
    return new_contacts
end

@testset "Clinical transitions" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Empty transitions leave state unchanged" begin
        # Backwards-compat: no transitions kwarg = old behaviour. No new
        # state keys appear on individuals.
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; attributes = clinical,
            sim_opts = SimOpts(max_cases = 20), rng = StableRNG(1))
        for ind in state.individuals
            @test !haskey(ind.state, :reported)
            @test !haskey(ind.state, :admitted)
            @test !haskey(ind.state, :outcome)
        end
    end

    @testset "Reporting marks symptomatic cases only" begin
        rng = StableRNG(2)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        clin_30 = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5),
            prob_asymptomatic = 0.3
        )
        rep = Reporting(delay = LogNormal(1.0, 0.3))
        state = simulate(model; attributes = clin_30, transitions = [rep],
            sim_opts = SimOpts(max_cases = 100), rng = rng)

        for ind in state.individuals
            @test haskey(ind.state, :reported)
            @test haskey(ind.state, :reporting_time)
            if is_asymptomatic(ind)
                @test ind.state[:reported] == false
                @test ind.state[:reporting_time] == Inf
            else
                @test ind.state[:reported] == true
                @test isfinite(ind.state[:reporting_time])
                @test ind.state[:reporting_time] > onset_time(ind)
            end
        end
    end

    @testset "Reporting honours probability < 1" begin
        rng = StableRNG(3)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        rep = Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5)
        state = simulate(model; attributes = clinical, transitions = [rep],
            sim_opts = SimOpts(max_cases = 400), rng = rng)
        frac_reported = count(ind -> ind.state[:reported], state.individuals) /
                        length(state.individuals)
        @test 0.35 <= frac_reported <= 0.65
    end

    @testset "Hospitalisation: prob=0 never admits, prob=1 always admits" begin
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))

        h0 = Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.0)
        state0 = simulate(model; attributes = clinical, transitions = [h0],
            sim_opts = SimOpts(max_cases = 50), rng = StableRNG(4))
        @test all(!ind.state[:admitted] for ind in state0.individuals)

        h1 = Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 1.0)
        state1 = simulate(model; attributes = clinical, transitions = [h1],
            sim_opts = SimOpts(max_cases = 50), rng = StableRNG(5))
        @test all(ind.state[:admitted] for ind in state1.individuals)
    end

    @testset "Hospitalisation gated on reporting via probability closure" begin
        rng = StableRNG(6)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Reporting probability 0.5; admission probability 1.0 if reported,
        # 0.0 otherwise → admitted set ⊆ reported set. The gate is expressed
        # inside `probability`; no special field needed.
        rep = Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5)
        hosp = Hospitalisation(
            delay = LogNormal(2.0, 0.5),
            probability = (rng, ind) -> get(ind.state, :reported, false) ? 1.0 :
                                        0.0
        )
        state = simulate(model; attributes = clinical,
            transitions = [rep, hosp],
            sim_opts = SimOpts(max_cases = 200), rng = rng)
        for ind in state.individuals
            ind.state[:admitted] && @test ind.state[:reported]
        end
    end

    @testset "Death/Recovery: terminal arbitration sets :outcome" begin
        rng = StableRNG(7)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        d = Death(delay = LogNormal(2.5, 0.4), probability = 0.3)
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = simulate(model; attributes = clinical, transitions = [d, r],
            sim_opts = SimOpts(max_cases = 200), rng = rng)
        for ind in state.individuals
            @test haskey(ind.state, :outcome)
            @test ind.state[:outcome] in (:died, :recovered)
            @test isfinite(ind.state[:outcome_time])
            # Outcome time matches the candidate of the chosen label.
            if ind.state[:outcome] == :died
                @test ind.state[:outcome_time] ==
                      ind.state[:death_candidate_time]
                # And the death candidate fires before any recovery candidate.
                @test ind.state[:death_candidate_time] <=
                      ind.state[:recovery_candidate_time]
            else
                @test ind.state[:outcome_time] ==
                      ind.state[:recovery_candidate_time]
            end
        end
    end

    @testset "Death prob=0 → everyone recovers" begin
        rng = StableRNG(8)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        d = Death(delay = LogNormal(2.5, 0.4), probability = 0.0)
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = simulate(model; attributes = clinical, transitions = [d, r],
            sim_opts = SimOpts(max_cases = 50), rng = rng)
        @test all(ind.state[:outcome] == :recovered for ind in state.individuals)
    end

    @testset "Asymptomatic cases skip all transitions" begin
        rng = StableRNG(9)
        all_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5),
            prob_asymptomatic = 1.0
        )
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        ts = [Reporting(delay = LogNormal(1.0, 0.3)),
            Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 1.0),
            Death(delay = LogNormal(2.5, 0.4), probability = 1.0),
            Recovery(delay = LogNormal(2.0, 0.4))]
        state = simulate(model; attributes = all_asymp, transitions = ts,
            sim_opts = SimOpts(max_cases = 50), rng = rng)
        for ind in state.individuals
            @test ind.state[:reported] == false
            @test ind.state[:admitted] == false
            @test !haskey(ind.state, :outcome)
        end
    end

    @testset "Required-field validation catches missing :onset_time" begin
        model = BranchingProcess(Poisson(1.0), Exponential(5.0))
        rep = Reporting(delay = LogNormal(1.0, 0.3))
        # No attributes function → no :onset_time → error.
        @test_throws ErrorException simulate(model; transitions = [rep],
            sim_opts = SimOpts(max_cases = 5), rng = StableRNG(10))
    end

    @testset "Heterogeneous probability via function" begin
        rng = StableRNG(11)
        # Demographics + clinical so :age and :onset_time are both set.
        attrs = compose(
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90))
        )
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Closure CFR: 0% below 80, 100% at 80 and above. Expect deaths
        # only in the 80+ band. Use a short death delay so terminal
        # arbitration deterministically picks death whenever its
        # probability is 1.0, without seed-sensitivity in Recovery's
        # sample.
        d = Death(delay = LogNormal(0.5, 0.1),
            probability = (rng, ind) -> ind.state[:age] >= 80 ? 1.0 : 0.0)
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = simulate(model; condition = 100:500, attributes = attrs,
            transitions = [d, r],
            sim_opts = SimOpts(max_cases = 500), rng = rng)
        n_died_80plus = 0
        n_died_under80 = 0
        for ind in state.individuals
            ind.state[:outcome] == :died || continue
            ind.state[:age] >= 80 ? (n_died_80plus += 1) : (n_died_under80 += 1)
        end
        @test n_died_under80 == 0
        @test n_died_80plus > 0
    end

    @testset "Heterogeneous delay via function" begin
        # Age-conditional admission delay: youngest cases get admitted
        # faster than older ones. Check the per-case admission times
        # respect the rule.
        rng = StableRNG(13)
        attrs = compose(
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90))
        )
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Under-30: fixed 1-day delay. 30+: fixed 5-day delay. Comparing
        # admission_time - onset_time recovers the right band.
        hosp = Hospitalisation(
            delay = (rng, ind) -> ind.state[:age] < 30 ? 1.0 : 5.0,
            probability = 1.0)
        state = simulate(model; attributes = attrs,
            transitions = [hosp],
            sim_opts = SimOpts(max_cases = 100), rng = rng)
        for ind in state.individuals
            ind.state[:admitted] || continue
            d = ind.state[:admission_time] - onset_time(ind)
            if ind.state[:age] < 30
                @test d ≈ 1.0
            else
                @test d ≈ 5.0
            end
        end
    end

    @testset "Anchor on :test_time via `from`" begin
        # Built-in Reporting anchored on a state key set by a custom
        # upstream transition. Reporting fires only after Testing wrote
        # :test_time; the reporting time is test_time + reporting delay,
        # not onset + reporting delay.
        rng = StableRNG(14)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        test_then_report = [
            # DummyTest uses LogNormal(0.0, 0.5) → :test_time stochastic
            DummyTest(LogNormal(0.0, 0.5)),
            # Deterministic 1-day reporting delay so we can assert the
            # anchor relation exactly.
            Reporting(delay = (rng, ind) -> 1.0, from = :test_time)
        ]
        state = simulate(model; attributes = clinical,
            transitions = test_then_report,
            sim_opts = SimOpts(max_cases = 50), rng = rng)
        for ind in state.individuals
            @test ind.state[:reported]
            @test ind.state[:reporting_time] ≈ ind.state[:test_time] + 1.0
        end
    end

    @testset "Anchor via function form (infection_time)" begin
        # Bypass onset entirely: anchor reporting on the Individual's
        # infection_time field via `from = ind -> ind.infection_time`.
        # No clinical_presentation needed — `from` is a function, so the
        # validator does not require :onset_time.
        rng = StableRNG(15)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        rep = Reporting(delay = (rng, ind) -> 1.0,
            from = ind -> ind.infection_time)
        state = simulate(model; transitions = [rep],
            sim_opts = SimOpts(max_cases = 30), rng = rng)
        for ind in state.individuals
            @test ind.state[:reported]
            @test ind.state[:reporting_time] ≈ ind.infection_time + 1.0
        end
    end

    @testset "Custom user-defined transition" begin
        # End-to-end check that the public interface is enough: the
        # struct + methods are defined above this testset (Julia
        # struct definitions can't live inside @testset).
        rng = StableRNG(12)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; attributes = clinical,
            transitions = [DummyTest(LogNormal(0.5, 0.2))],
            sim_opts = SimOpts(max_cases = 30), rng = rng)
        @test all(ind.state[:tested] for ind in state.individuals)
        @test all(isfinite(ind.state[:test_time]) for ind in state.individuals)
    end

    @testset "Custom TransmissionModel — engine resolves transitions automatically" begin
        # SingleSpawnModel's step! just appends infected individuals;
        # it does not touch transitions itself. The engine sweep should
        # still populate DummyTest fields on every infected case.
        rng = StableRNG(7)
        state = simulate(SingleSpawnModel(); attributes = clinical,
            transitions = [DummyTest(LogNormal(0.5, 0.2))],
            sim_opts = SimOpts(max_cases = 5), rng = rng)
        @test all(ind.state[:tested] for ind in state.individuals)
        @test all(isfinite(ind.state[:test_time]) for ind in state.individuals)
    end
end
