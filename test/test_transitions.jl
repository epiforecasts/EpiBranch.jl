# A custom transition defined outside the testset (struct definitions
# must be top-level) so we can exercise the public extension interface
# end-to-end below.
struct DummyTest <: AbstractClinicalTransition
    delay::Distribution
end
EpiBranch.required_fields(::DummyTest) = [:onset_time]
function EpiBranch.initialise_individual!(::DummyTest, ind, state)
    ind.state[:tested] = false
    ind.state[:test_time] = Inf
    return nothing
end
function EpiBranch.resolve_individual!(t::DummyTest, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    ind.state[:tested] = true
    ind.state[:test_time] = ot + rand(state.rng, t.delay)
    return nothing
end

# Minimal custom TransmissionModel used to verify the engine handles
# bookkeeping, competing-risks resolution, and clinical-transition
# resolution for new individuals. As an offspring-driven model it only
# implements `generate_offspring`; the engine owns timing and creation.
# One offspring per parent each generation.
struct SingleSpawnModel{A, O} <: EpiBranch.TransmissionModel
    generation_time::Exponential{Float64}
    progression::Vector{EpiBranch.AbstractClinicalTransition}
    interventions::Vector{EpiBranch.AbstractIntervention}
    attributes::A
    observation::O
end
function SingleSpawnModel(;
        progression = EpiBranch.AbstractClinicalTransition[],
        interventions = EpiBranch.AbstractIntervention[],
        attributes = EpiBranch.NoAttributes(),
        observation::EpiBranch.ObservationModel = EpiBranch.NoObservation()
    )
    return SingleSpawnModel(
        Exponential(1.0),
        convert(Vector{EpiBranch.AbstractClinicalTransition}, progression),
        convert(Vector{EpiBranch.AbstractIntervention}, interventions),
        attributes, observation
    )
end
EpiBranch.generate_offspring(::SingleSpawnModel, parent, state) = 1
EpiBranch._progression(m::SingleSpawnModel) = m.progression
# Carry the model inputs so the model joins the engine.
EpiBranch.interventions(m::SingleSpawnModel) = m.interventions
EpiBranch.attributes(m::SingleSpawnModel) = m.attributes
EpiBranch.observation(m::SingleSpawnModel) = m.observation

@testset "Clinical transitions" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Empty transitions leave state unchanged" begin
        # Backwards-compat: no transitions kwarg = old behaviour. No new
        # state keys appear on individuals.
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = tsim(
            model; attributes = clinical,
            max_cases = 20, rng = StableRNG(1)
        )
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
        state = tsim(
            model; attributes = clin_30, transitions = [rep],
            max_cases = 100, rng = rng
        )

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
        state = tsim(
            model; attributes = clinical, transitions = [rep],
            max_cases = 400, rng = rng
        )
        frac_reported = count(ind -> ind.state[:reported], state.individuals) /
            length(state.individuals)
        @test 0.35 <= frac_reported <= 0.65
    end

    @testset "Hospitalisation: prob=0 never admits, prob=1 always admits" begin
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))

        h0 = Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.0)
        state0 = tsim(
            model; attributes = clinical, transitions = [h0],
            max_cases = 50, rng = StableRNG(4)
        )
        @test all(!ind.state[:admitted] for ind in state0.individuals)

        h1 = Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 1.0)
        state1 = tsim(
            model; attributes = clinical, transitions = [h1],
            max_cases = 50, rng = StableRNG(5)
        )
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
        state = tsim(
            model; attributes = clinical,
            transitions = [rep, hosp],
            max_cases = 200, rng = rng
        )
        for ind in state.individuals
            ind.state[:admitted] && @test ind.state[:reported]
        end
    end

    @testset "Chained transition skips an un-reached (Inf) anchor" begin
        # Hospitalisation never occurs (probability 0), so :admission_time stays
        # at its Inf default. A Reporting anchored on :admission_time must not
        # occur either — an Inf anchor means the upstream state was never
        # reached. (Previously the isnan guard let Inf through, reporting the
        # case at time Inf.)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        hosp = Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.0)
        rep = Reporting(
            delay = LogNormal(1.0, 0.3), probability = 1.0,
            from = :admission_time
        )
        state = tsim(
            model; attributes = clinical, transitions = [hosp, rep],
            max_cases = 200, rng = StableRNG(11)
        )
        @test all(!ind.state[:admitted] for ind in state.individuals)
        @test all(!ind.state[:reported] for ind in state.individuals)
        @test all(!isfinite(ind.state[:reporting_time]) for ind in state.individuals)
    end

    @testset "A non-terminal event after the outcome is censored" begin
        # Admission delay (10 days) outlasts recovery (2 days) for every case,
        # so every admission this would otherwise record falls after the
        # outcome that ends the case's clinical course.
        rng = StableRNG(10)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        hosp = Hospitalisation(delay = (rng, ind) -> 10.0, probability = 1.0)
        recovery = Recovery(delay = (rng, ind) -> 2.0)
        state = tsim(
            model; attributes = clinical, transitions = [hosp, recovery],
            max_cases = 50, rng = rng
        )
        for ind in state.individuals
            @test ind.state[:outcome] == :recovered
            @test ind.state[:admitted] == false
            @test ind.state[:admission_time] == Inf
        end
    end

    @testset "Death/Recovery: terminal arbitration sets :outcome" begin
        rng = StableRNG(7)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        d = Death(delay = LogNormal(2.5, 0.4), probability = 0.3)
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = tsim(
            model; attributes = clinical, transitions = [d, r],
            max_cases = 200, rng = rng
        )
        for ind in state.individuals
            @test haskey(ind.state, :outcome)
            @test ind.state[:outcome] in (:died, :recovered)
            @test isfinite(ind.state[:outcome_time])
            # Outcome time matches the candidate of the chosen label.
            if ind.state[:outcome] == :died
                @test ind.state[:outcome_time] ==
                    ind.state[:death_candidate_time]
                # And the death candidate's time comes before any recovery candidate's.
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
        state = tsim(
            model; attributes = clinical, transitions = [d, r],
            max_cases = 50, rng = rng
        )
        @test all(ind.state[:outcome] == :recovered for ind in state.individuals)
    end

    @testset "Asymptomatic cases skip all transitions" begin
        rng = StableRNG(9)
        all_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5),
            prob_asymptomatic = 1.0
        )
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        ts = [
            Reporting(delay = LogNormal(1.0, 0.3)),
            Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 1.0),
            Death(delay = LogNormal(2.5, 0.4), probability = 1.0),
            Recovery(delay = LogNormal(2.0, 0.4)),
        ]
        state = tsim(
            model; attributes = all_asymp, transitions = ts,
            max_cases = 50, rng = rng
        )
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
        @test_throws ErrorException tsim(
            model; transitions = [rep],
            max_cases = 5, rng = StableRNG(10)
        )
    end

    @testset "Heterogeneous probability via function" begin
        rng = StableRNG(11)
        # Demographics + clinical so :age and :onset_time are both set.
        attrs = [
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90)),
        ]
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Closure CFR: 0% below 80, 100% at 80 and above. Expect deaths
        # only in the 80+ band. Use a short death delay so terminal
        # arbitration deterministically picks death whenever its
        # probability is 1.0, without seed-sensitivity in Recovery's
        # sample.
        d = Death(
            delay = LogNormal(0.5, 0.1),
            probability = (rng, ind) -> ind.state[:age] >= 80 ? 1.0 : 0.0
        )
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = tsim(
            model; condition = 100:500, attributes = attrs,
            transitions = [d, r],
            max_cases = 500, rng = rng
        )
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
        attrs = [
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90)),
        ]
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Under-30: fixed 1-day delay. 30+: fixed 5-day delay. Comparing
        # admission_time - onset_time recovers the right band.
        hosp = Hospitalisation(
            delay = (rng, ind) -> ind.state[:age] < 30 ? 1.0 : 5.0,
            probability = 1.0
        )
        state = tsim(
            model; attributes = attrs,
            transitions = [hosp],
            max_cases = 100, rng = rng
        )
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
        # upstream transition. Reporting occurs only after Testing wrote
        # :test_time; the reporting time is test_time + reporting delay,
        # not onset + reporting delay.
        rng = StableRNG(14)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        test_then_report = [
            # DummyTest uses LogNormal(0.0, 0.5) → :test_time stochastic
            DummyTest(LogNormal(0.0, 0.5)),
            # Deterministic 1-day reporting delay so we can assert the
            # anchor relation exactly.
            Reporting(delay = (rng, ind) -> 1.0, from = :test_time),
        ]
        state = tsim(
            model; attributes = clinical,
            transitions = test_then_report,
            max_cases = 50, rng = rng
        )
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
        rep = Reporting(
            delay = (rng, ind) -> 1.0,
            from = ind -> ind.infection_time
        )
        state = tsim(
            model; transitions = [rep],
            max_cases = 30, rng = rng
        )
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
        state = tsim(
            model; attributes = clinical,
            transitions = [DummyTest(LogNormal(0.5, 0.2))],
            max_cases = 30, rng = rng
        )
        @test all(ind.state[:tested] for ind in state.individuals)
        @test all(isfinite(ind.state[:test_time]) for ind in state.individuals)
    end

    @testset "Custom TransmissionModel — engine resolves transitions automatically" begin
        # SingleSpawnModel's generate_offspring just yields infected cases;
        # it does not touch transitions itself. The engine sweep should
        # still populate DummyTest fields on every infected case.
        rng = StableRNG(7)
        state = simulate(
            SingleSpawnModel(; progression = [DummyTest(LogNormal(0.5, 0.2))], attributes = clinical);
            max_cases = 5,
            rng = rng
        )
        @test all(ind.state[:tested] for ind in state.individuals)
        @test all(isfinite(ind.state[:test_time]) for ind in state.individuals)
    end
end

struct FollowupVisit <: AbstractClinicalTransition end
function EpiBranch.resolve_individual!(::FollowupVisit, ind, state)
    time = EpiBranch.transition_time(
        state.rng, ind, ind.infection_time, 2.0;
        probability = 1.0
    )
    time === nothing || (ind.state[:followup_time] = time)
    return nothing
end

@testset "Shared clinical event sampling" begin
    ind = Individual(id = 1, infection_time = 3.0)
    for start in (NaN, Inf, -Inf)
        rng = StableRNG(14)
        @test EpiBranch.transition_time(
            rng, ind, start,
            (rng, ind) -> error("unreached delay");
            probability = (rng, ind) -> error("unreached probability")
        ) === nothing
        @test rand(rng) == rand(StableRNG(14))
    end
    for delay in (2.0, Exponential(2.0), (rng, ind) -> rand(rng) + 1)
        rng, expected = StableRNG(15), StableRNG(15)
        # The probability callback consumes a draw before the acceptance draw.
        p = rand(expected)
        accepted = rand(expected) < p
        wait = accepted ? EpiBranch._resolve_delay(delay, expected, ind) : nothing
        result = EpiBranch.transition_time(
            rng, ind, 3.0, delay;
            probability = (rng, ind) -> rand(rng)
        )
        @test result === nothing ? !accepted : result == 3.0 + wait
        @test rand(rng) == rand(expected)
    end
    rng = StableRNG(16)
    @test EpiBranch.transition_time(rng, ind, 3.0, 2.0) == 5.0
    @test rand(rng) == rand(StableRNG(16))

    # Generic and named transitions preserve what delay callbacks can observe.
    generic = Transition(:arrived; delay = (rng, ind) -> ind.state[:arrived] ? 2.0 : 9.0)
    reporting = Reporting(
        from = ind -> ind.infection_time,
        delay = (rng, ind) -> ind.state[:reported] ? 9.0 : 2.0
    )
    admission = Hospitalisation(
        from = ind -> ind.infection_time, probability = 1.0,
        delay = (rng, ind) -> ind.state[:admitted] ? 9.0 : 2.0
    )
    model = ModelSpec(
        BranchingProcess(Poisson(0.0));
        progression = [generic, reporting, admission, FollowupVisit()]
    )
    state = simulate(model; rng = StableRNG(17))
    for key in (:arrived_time, :reporting_time, :admission_time, :followup_time)
        @test only(state.individuals).state[key] == 2.0
    end
end

@testset "exclusive_probabilities" begin
    @testset "validates its input" begin
        @test_throws ArgumentError exclusive_probabilities([0.6, 0.6])
        @test_throws ArgumentError exclusive_probabilities([-0.1, 1.1])
    end

    @testset "exactly one sibling occurs, whichever order they resolve in" begin
        ps = [0.2, 0.3, 0.5]
        # Calling the gates in a different order each time checks the shared
        # draw is cached on first use, not on a fixed position in the group.
        orders = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
        for seed in 1:100
            gates = exclusive_probabilities(ps)
            ind = Individual(id = 1, infection_time = 0.0)
            order = orders[mod1(seed, length(orders))]
            occurred = [gates[i](StableRNG(seed + i), ind) for i in order]
            @test count(==(1.0), occurred) == 1
            @test count(==(0.0), occurred) == length(ps) - 1
        end
    end

    @testset "partitions the population in the given proportions" begin
        ps = [0.36, 0.64]
        n_low = 0
        n_high = 0
        N = 20_000
        for seed in 1:N
            gates = exclusive_probabilities(ps)
            ind = Individual(id = 1, infection_time = 0.0)
            low, high = gates[1](StableRNG(seed), ind), gates[2](nothing, ind)
            @test low + high == 1.0  # never both, never neither
            n_low += low == 1.0
            n_high += high == 1.0
        end
        @test isapprox(n_low / N, ps[1]; atol = 0.02)
        @test isapprox(n_high / N, ps[2]; atol = 0.02)
    end

    @testset "a shortfall below 1 is the probability neither occurs" begin
        ps = [0.3, 0.3]
        n_neither = 0
        N = 20_000
        for seed in 1:N
            gates = exclusive_probabilities(ps)
            ind = Individual(id = 1, infection_time = 0.0)
            low, high = gates[1](StableRNG(seed), ind), gates[2](nothing, ind)
            @test low + high in (0.0, 1.0)
            n_neither += (low + high == 0.0)
        end
        @test isapprox(n_neither / N, 1 - sum(ps); atol = 0.02)
    end

    @testset "an exact case-fatality ratio, end to end" begin
        # The reproducer from the issue: two terminal transitions gated
        # independently at p and 1 - p leave some cases with neither outcome
        # (about p(1 - p) of them). `exclusive_probabilities` fixes that: no
        # case has both, or neither.
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        rng = StableRNG(21)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        died_p, recovered_p = exclusive_probabilities([0.36, 0.64])
        died = Transition(
            :died; from = :onset, delay = LogNormal(2.5, 0.4),
            probability = died_p, terminal = true
        )
        recovered = Transition(
            :recovered; from = :onset, delay = LogNormal(2.0, 0.4),
            probability = recovered_p, terminal = true
        )
        state = tsim(
            model; attributes = clinical, transitions = [died, recovered],
            condition = 500:1000, max_cases = 1000, rng = rng
        )
        @test all(ind -> haskey(ind.state, :outcome), state.individuals)
        n_died = count(ind -> ind.state[:outcome] == :died, state.individuals)
        @test isapprox(n_died / length(state.individuals), 0.36; atol = 0.05)
    end
end
