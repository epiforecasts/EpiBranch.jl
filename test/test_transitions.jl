# A custom transition defined outside the testset (struct definitions
# must be top-level) so we can exercise the public extension interface
# end-to-end below.
struct DummyTest <: AbstractClinicalTransition
    delay::Distribution
end
EpiBranch.required_fields(::DummyTest) = [:onset_time, :asymptomatic]
function EpiBranch.initialise_individual!(::DummyTest, ind, state)
    ind.state[:tested] = false
    ind.state[:test_time] = Inf
    return nothing
end
function EpiBranch.resolve_individual!(t::DummyTest, ind, state)
    is_asymptomatic(ind) && return nothing
    ot = onset_time(ind)
    isnan(ot) && return nothing
    ind.state[:tested] = true
    ind.state[:test_time] = ot + rand(state.rng, t.delay)
    return nothing
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

    @testset "Hospitalisation gated on reporting" begin
        rng = StableRNG(6)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Reporting probability 0.5; admission probability 1.0 but gated on
        # reporting → admitted set ⊆ reported set.
        rep = Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5)
        hosp = Hospitalisation(delay = LogNormal(2.0, 0.5),
            probability = 1.0, requires_reporting = true)
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
        state = simulate(model; attributes = all_asymp,
            transitions = clinical_default(
                reporting_delay = LogNormal(1.0, 0.3),
                admission_delay = LogNormal(2.0, 0.5),
                outcome_delay = LogNormal(2.5, 0.4)),
            sim_opts = SimOpts(max_cases = 50), rng = rng)
        for ind in state.individuals
            @test ind.state[:reported] == false
            @test ind.state[:admitted] == false
            @test !haskey(ind.state, :outcome)
        end
    end

    @testset "clinical_default builds the expected stack" begin
        ts = clinical_default(
            reporting_delay = LogNormal(1.0, 0.3),
            admission_delay = LogNormal(2.0, 0.5),
            outcome_delay = LogNormal(2.5, 0.4))
        @test length(ts) == 4
        @test ts[1] isa Reporting
        @test ts[2] isa Hospitalisation
        @test ts[3] isa Death
        @test ts[4] isa Recovery

        ts_min = clinical_default(reporting_delay = LogNormal(1.0, 0.3))
        @test length(ts_min) == 1
        @test ts_min[1] isa Reporting

        @test isempty(clinical_default())
    end

    @testset "Required-field validation catches missing :onset_time" begin
        model = BranchingProcess(Poisson(1.0), Exponential(5.0))
        rep = Reporting(delay = LogNormal(1.0, 0.3))
        # No attributes function → no :onset_time → error.
        @test_throws ErrorException simulate(model; transitions = [rep],
            sim_opts = SimOpts(max_cases = 5), rng = StableRNG(10))
    end

    @testset "Age-specific CFR overrides probability" begin
        rng = StableRNG(11)
        # Demographics + clinical so :age and :onset_time are both set.
        attrs = compose(
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90))
        )
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # 0% baseline, 100% for 80+. Expect deaths concentrated in 80+ band.
        d = Death(delay = LogNormal(2.5, 0.4),
            probability = 0.0,
            age_specific_cfr = [(0, 79) => 0.0, (80, 120) => 1.0])
        r = Recovery(delay = LogNormal(2.0, 0.4))
        state = simulate(model; attributes = attrs,
            transitions = [d, r],
            sim_opts = SimOpts(max_cases = 300), rng = rng)
        n_died_80plus = 0
        n_died_under80 = 0
        for ind in state.individuals
            ind.state[:outcome] == :died || continue
            ind.state[:age] >= 80 ? (n_died_80plus += 1) : (n_died_under80 += 1)
        end
        @test n_died_under80 == 0
        @test n_died_80plus > 0
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
end
