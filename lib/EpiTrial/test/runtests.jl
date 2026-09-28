using Test
using EpiTrial
using EpiBranch
using EpiHouseholds
using DataFrames
using Distributions
using StableRNGs
using Statistics

# A cohort of `n` people living alone, facing a constant force of infection
# `λ` from outside the trial over `[0, T]`. With no transmission inside the
# cohort the cumulative hazard of an unvaccinated participant is exactly λT,
# which is what the closed forms in `expected_ve` take.
function cohort_trial(; n, λ, T, efficacy, mode = LeakyMode(), vaccine_fraction = 0.5)
    spec = ModelSpec(
        HouseholdProcess(fill(1, n), Exponential(1.0); external_hazard = λ, obs_end = T);
        progression = [Transition(:recovered; from = :infection, delay = 5.0,
            terminal = true)],
        attributes = IndividualRandomisation(; vaccine_fraction),
        interventions = [TrialVaccination(; efficacy, mode)])
    return Trial(spec; follow_up = T)
end

# Whether an estimate lies within `k` standard errors of `target`, the standard
# error read back from the width of its 95% interval on the log ratio scale.
function within_se(est, target; k = 4)
    se = (log(1 - est.lower) - log(1 - est.upper)) / (2 * 1.96)
    return abs(log(1 - est.ve) - log(1 - target)) < k * se
end

@testset "EpiTrial" begin
    @testset "randomisation" begin
        @test_throws ArgumentError IndividualRandomisation(vaccine_fraction = 1.5)
        trial = cohort_trial(n = 4000, λ = 0.0, T = 10.0, efficacy = 0.5,
            vaccine_fraction = 0.3)
        state = simulate(trial.spec; rng = StableRNG(1))
        frac = mean(arm(ind) === :vaccine for ind in state.individuals)
        @test abs(frac - 0.3) < 4 * sqrt(0.3 * 0.7 / 4000)
    end

    @testset "trial vaccination doses the vaccine arm at enrolment" begin
        trial = cohort_trial(n = 200, λ = 0.0, T = 10.0, efficacy = 0.5)
        state = simulate(trial.spec; rng = StableRNG(2))
        for ind in state.individuals
            @test is_vaccinated(ind) == (arm(ind) === :vaccine)
            is_vaccinated(ind) && @test ind.state[:vaccination_time] == 0.0
        end

        # Without a randomisation there is no arm to read.
        spec = ModelSpec(HouseholdProcess(fill(1, 5), Exponential(1.0));
            progression = [Transition(:recovered; from = :infection, delay = 5.0,
                terminal = true)],
            interventions = [TrialVaccination(efficacy = 0.5)])
        @test_throws ArgumentError simulate(spec; rng = StableRNG(3))
    end

    @testset "trial data" begin
        trial = cohort_trial(n = 500, λ = 0.01, T = 50.0, efficacy = 0.5)
        data = simulate(trial; rng = StableRNG(4))
        @test names(data) == ["id", "arm", "exit_time", "event"]
        @test nrow(data) == 500
        @test all(data.exit_time .<= 50.0)
        @test all(data.exit_time[.!data.event] .== 50.0)
        @test 0 < count(data.event) < 500

        # Index cases infected at enrolment are not participants: one per
        # household when there is no community force of infection.
        spec = ModelSpec(HouseholdProcess(fill(3, 40), Exponential(2.0));
            progression = [Transition(:recovered; from = :infection, delay = 5.0,
                terminal = true)],
            attributes = IndividualRandomisation(),
            interventions = [TrialVaccination(efficacy = 0.5)])
        @test nrow(simulate(Trial(spec; follow_up = 30.0); rng = StableRNG(5))) == 80

        # Simulation options reach the outbreak: seeding a homogeneous epidemic
        # with ten cases removes them from the participants.
        spec = ModelSpec(
            HomogeneousProcess(transmission_rate = 1.5, population_size = 500);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)],
            attributes = IndividualRandomisation(),
            interventions = [TrialVaccination(efficacy = 0.6)])
        trial = Trial(spec; follow_up = 60.0)
        @test nrow(simulate(trial; rng = StableRNG(12), n_initial = 10)) == 490
        reps = trial_estimates(trial, [RiskRatio()]; n_sim = 3, rng = StableRNG(13),
            n_initial = 10)
        @test all(reps.events .> 0)
    end

    @testset "estimators on a fixed table" begin
        # 10/100 events in the vaccine arm, 20/100 in the control arm.
        data = DataFrame(arm = [fill(:vaccine, 100); fill(:control, 100)],
            exit_time = fill(1.0, 200),
            event = [fill(true, 10); fill(false, 90); fill(true, 20); fill(false, 80)])
        rr = estimate(RiskRatio(), data)
        @test rr.ve ≈ 0.5
        se = sqrt(1 / 10 - 1 / 100 + 1 / 20 - 1 / 100)
        @test rr.lower ≈ 1 - exp(log(0.5) + 1.959963984540054 * se)
        @test rr.upper ≈ 1 - exp(log(0.5) - 1.959963984540054 * se)
        @test rr.p_value ≈ 2 * cdf(Normal(), -abs(log(0.5) / se))

        # No events in the vaccine arm: the risk ratio is continuity-corrected
        # and the Cox model has no finite estimate.
        data.event[1:10] .= false
        @test estimate(RiskRatio(), data).ve ≈ 1 - (0.5 / 101) / (20.5 / 101)
        @test isnan(estimate(CoxHazardRatio(), data).ve)
    end

    @testset "leaky vaccine: estimates match the closed forms" begin
        for Λ in (0.3, 2.0)
            T = 100.0
            trial = cohort_trial(n = 40_000, λ = Λ / T, T = T, efficacy = 0.6)
            data = simulate(trial; rng = StableRNG(6))
            for est in (RiskRatio(), CoxHazardRatio())
                @test within_se(estimate(est, data),
                    expected_ve(est, LeakyMode(), 0.6, Λ))
            end
        end
        # The attack-rate estimate is attenuated as the cumulative hazard grows.
        @test expected_ve(RiskRatio(), LeakyMode(), 0.6, 2.0) <
              expected_ve(RiskRatio(), LeakyMode(), 0.6, 0.3) < 0.6
    end

    @testset "all-or-nothing closed forms" begin
        @test expected_ve(RiskRatio(), AllOrNothingMode(), 0.6, 2.0) == 0.6
        # The Cox limit starts at θ and rises with the cumulative hazard.
        cox(Λ) = expected_ve(CoxHazardRatio(), AllOrNothingMode(), 0.6, Λ)
        @test cox(1e-6) ≈ 0.6 atol = 1e-4
        @test 0.6 < cox(0.5) < cox(2.0) < 1

        # Against trial data drawn directly: a vaccinee is protected with
        # probability θ, and everyone else is infected at rate λ.
        rng = StableRNG(11)
        n, λ, T, θ = 40_000, 0.02, 100.0, 0.6
        vac = rand(rng, n) .< 0.5
        t = [vac[i] && rand(rng) < θ ? Inf : rand(rng, Exponential(1 / λ))
             for i in 1:n]
        data = DataFrame(arm = ifelse.(vac, :vaccine, :control),
            exit_time = min.(t, T), event = t .<= T)
        for est in (RiskRatio(), CoxHazardRatio())
            @test within_se(estimate(est, data),
                expected_ve(est, AllOrNothingMode(), θ, λ * T))
        end

        # All-or-nothing protection is not yet honoured by the engine (#312): a
        # vaccine given `AllOrNothingMode` acts as leaky. Flip these to `@test`
        # once it is.
        trial = cohort_trial(n = 40_000, λ = 0.02, T = 100.0, efficacy = 0.6,
            mode = AllOrNothingMode())
        data = simulate(trial; rng = StableRNG(7))
        for est in (RiskRatio(), CoxHazardRatio())
            @test_broken within_se(estimate(est, data),
                expected_ve(est, AllOrNothingMode(), 0.6, 2.0))
        end
    end

    @testset "operating characteristics" begin
        n, T, Λ, e = 1000, 100.0, 0.2, 0.4
        trial = cohort_trial(; n, λ = Λ / T, T, efficacy = e)
        oc = operating_characteristics(trial, [CoxHazardRatio(), RiskRatio()];
            n_sim = 400, rng = StableRNG(8),
            target = est -> expected_ve(est, LeakyMode(), e, Λ))
        @test nrow(oc) == 2
        cox = oc[oc.estimator .=== CoxHazardRatio(), :]

        # Power against Schoenfeld's approximation, with 1:1 allocation and the
        # expected number of events.
        events = n / 2 * (1 - exp(-Λ)) + n / 2 * (1 - exp(-(1 - e) * Λ))
        @test only(cox.mean_events) ≈ events rtol = 0.05
        schoenfeld = cdf(Normal(), abs(log(1 - e)) * sqrt(events / 4) - 1.959963984540054)
        @test abs(only(cox.power) - schoenfeld) < 0.06
        @test abs(only(cox.coverage) - 0.95) < 0.04
        @test abs(only(cox.bias)) < 0.05

        # With no effect, "power" is the one-sided type I error, 2.5%.
        null = operating_characteristics(cohort_trial(; n, λ = Λ / T, T, efficacy = 0.0),
            [CoxHazardRatio()]; n_sim = 400, rng = StableRNG(9))
        @test only(null.power) < 0.06

        # Summarising stored replicates gives the same table.
        reps = trial_estimates(trial, [CoxHazardRatio()]; n_sim = 20, rng = StableRNG(10))
        @test nrow(reps) == 20
        @test operating_characteristics(reps).power ==
              operating_characteristics(trial, [CoxHazardRatio()];
            n_sim = 20, rng = StableRNG(10)).power
    end
end
