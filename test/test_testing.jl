# A user-supplied acute course: load (log10 copies) climbs linearly to a peak
# `peak_day` after infection, then falls linearly, so the load at a given time
# is known exactly.
const TEST_PEAK_DAY = 5.0
const TEST_PEAK = 8.0
test_curve(rng, ind) = t -> max(0.0, TEST_PEAK - abs(t - TEST_PEAK_DAY) * 1.6)
test_assay = Assay(value = load -> 40.0 - 3.0 * load, detect = 2.0)

function testing_spec(; progression, interventions = [], seed_cases = 1, kwargs...)
    return ModelSpec(
        BranchingProcess(NegBin(2.5, 0.5), LogNormal(1.6, 0.5));
        attributes = [
            clinical_presentation(
                incubation_period = Dirac(3.0), prob_asymptomatic = 0.3
            ),
            within_host(test_curve),
        ],
        progression, interventions
    )
end

@testset "Testing" begin
    @testset "Ct value and result follow the load at the sampling time" begin
        spec = testing_spec(
            progression = [Testing(assay = test_assay, delay = 4.0)]
        )
        state = simulate(spec; max_cases = 200, rng = StableRNG(1), n_initial = 20)
        symptomatic = filter(ind -> is_infected(ind) && !is_asymptomatic(ind), state.individuals)
        @test !isempty(symptomatic)
        for ind in symptomatic
            @test ind.state[:test_tested]
            # onset at 3 days, sample 4 days later
            t = ind.state[:test_time] - ind.infection_time
            @test t ≈ 7.0
            load = test_curve(nothing, nothing)(t)
            @test ind.state[:test_value] ≈ 40.0 - 3.0 * load
            @test ind.state[:test_result] == (load >= 2.0)
        end
        for ind in filter(ind -> is_infected(ind) && is_asymptomatic(ind), state.individuals)
            @test !ind.state[:test_tested]
            @test isnan(ind.state[:test_value])
        end
    end

    @testset "sampling time changes the result" begin
        # load is 8 - 1.6 * |t - 5|: 4.8 at day 3, 0.0 at day 15
        early = testing_spec(
            progression = [Testing(assay = test_assay, delay = 0.0)]
        )
        late = testing_spec(
            progression = [Testing(assay = test_assay, delay = 12.0)]
        )
        for (spec, expected) in ((early, true), (late, false))
            state = simulate(spec; max_cases = 100, rng = StableRNG(2), n_initial = 20)
            tested = filter(ind -> is_infected(ind) && ind.state[:test_tested], state.individuals)
            @test !isempty(tested)
            @test all(ind -> ind.state[:test_result] == expected, tested)
        end
    end

    @testset "line list" begin
        spec = testing_spec(
            progression = [Testing(assay = test_assay, delay = 1.0)]
        )
        ll = linelist(simulate(spec; max_cases = 50, rng = StableRNG(4), n_initial = 20))
        for col in (:test_tested, :date_test, :test_value, :test_result)
            @test col in propertynames(ll)
        end
    end

    @testset "a test timed after an earlier outcome is not taken" begin
        spec = testing_spec(
            progression = [
                Recovery(delay = 1.0),
                Testing(assay = test_assay, delay = 2.0, from = :onset_time),
            ]
        )
        state = simulate(spec; max_cases = 100, rng = StableRNG(5), n_initial = 20)
        for ind in filter(is_infected, state.individuals)
            # recovery 1 day after onset, test 2 days after it
            @test !ind.state[:test_tested]
            @test !ind.state[:test_result]
            @test isnan(ind.state[:test_value])
        end
    end

    @testset "Isolation takes its sensitivity from the test" begin
        positive = Assay(value = load -> 30.0, detect = 0.0)
        negative = Assay(value = load -> NaN, detect = 1.0e6)
        for (assay, isolated) in ((positive, true), (negative, false))
            testing = Testing(assay = assay, delay = 0.0)
            spec = testing_spec(
                progression = [testing],
                interventions = [
                    Isolation(
                        delay = 1.0, duration = Inf,
                        test_sensitivity = testing,
                    ),
                ]
            )
            state = simulate(spec; max_cases = 50, rng = StableRNG(6), n_initial = 20)
            for ind in filter(is_infected, state.individuals)
                is_asymptomatic(ind) && continue
                @test is_isolated(ind) == isolated
                @test is_test_positive(ind) == isolated
            end
        end
    end

    @testset "a sensitivity given as a function of the test time reads the load then" begin
        seen = Float64[]
        function by_load(threshold)
            return AtTestTime() do rng, ind, t
                push!(seen, t - ind.infection_time)
                load = EpiBranch.trajectory_value(ind, t - ind.infection_time)
                return load >= threshold ? 1.0 : 0.0
            end
        end
        # onset at 3 days, isolation delay 1: the load at 4 days is 6.4
        for (threshold, isolated) in ((6.0, true), (7.0, false))
            empty!(seen)
            spec = testing_spec(
                progression = AbstractClinicalTransition[],
                interventions = [
                    Isolation(
                        delay = 1.0, duration = Inf,
                        test_sensitivity = by_load(threshold),
                    ),
                ]
            )
            state = simulate(spec; max_cases = 50, rng = StableRNG(6), n_initial = 20)
            cases = filter(ind -> is_infected(ind) && !is_asymptomatic(ind), state.individuals)
            @test !isempty(cases)
            for ind in cases
                @test is_test_positive(ind) == isolated
                @test is_isolated(ind) == isolated
            end
            @test all(≈(4.0), seen)
            asymptomatic = filter(ind -> is_infected(ind) && is_asymptomatic(ind), state.individuals)
            @test !any(is_test_positive, asymptomatic)
        end
    end

    @testset "a plain function is drawn at creation whatever its arity" begin
        calls = Ref(0)
        spec = testing_spec(
            progression = AbstractClinicalTransition[],
            interventions = [
                Isolation(
                    delay = 1.0, duration = Inf,
                    test_sensitivity = (rng, ind, extra = nothing) -> (calls[] += 1; 1.0),
                ),
            ]
        )
        state = simulate(spec; max_cases = 20, rng = StableRNG(6), n_initial = 10)
        @test calls[] >= length(filter(is_infected, state.individuals))
        @test_throws ArgumentError EpiBranch._closed_form_value(AtTestTime((rng, ind, t) -> 1.0))
    end

    @testset "a Testing sensitivity without a Testing step is an error" begin
        testing = Testing(assay = Assay(value = load -> 30.0, detect = 0.0), delay = 0.0)
        @test_throws ArgumentError testing_spec(
            progression = AbstractClinicalTransition[],
            interventions = [
                Isolation(
                    delay = 1.0, duration = Inf,
                    test_sensitivity = testing,
                ),
            ],
        )
    end

    @testset "numeric test_sensitivity is unchanged and keeps its closed form" begin
        iso = Isolation(
            delay = Exponential(2.0), duration = Inf,
            test_sensitivity = 0.8
        )
        @test iso.test_sensitivity == 0.8
        testing = Testing(assay = test_assay, delay = 0.0)
        iso_t = Isolation(
            delay = Exponential(2.0), duration = Inf,
            test_sensitivity = testing
        )
        @test_throws ArgumentError EpiBranch._closed_form_value(iso_t.test_sensitivity)
    end

    @testset "Isolation with a Testing step absent from the progression is reported" begin
        testing = Testing(assay = test_assay, delay = 0.0)
        @test_throws ArgumentError testing_spec(
            progression = [],
            interventions = [
                Isolation(
                    delay = 1.0, duration = Inf,
                    test_sensitivity = testing,
                ),
            ]
        )
    end

    @testset "AfterPositiveTest without a Testing step of that name is refused" begin
        iso(name) = Isolation(
            delay = 1.0, duration = Inf,
            trigger = AfterPositiveTest(name = name)
        )
        process = BranchingProcess(NegBin(2.5, 0.5), LogNormal(1.6, 0.5))
        testing = Testing(assay = test_assay, delay = 0.0, name = :pcr)
        @test_throws ArgumentError ModelSpec(process; interventions = [iso(:pcr)])
        @test_throws ArgumentError ModelSpec(
            process; progression = [testing], interventions = [iso(:test)]
        )
        @test ModelSpec(process; progression = [testing], interventions = [iso(:pcr)]) isa ModelSpec
        @test_throws ArgumentError ModelSpec(
            process; interventions = [OnRoutes(iso(:pcr), [:household])]
        )
    end

    @testset "missing trajectory is reported" begin
        spec = ModelSpec(
            BranchingProcess(NegBin(2.5, 0.5), LogNormal(1.6, 0.5));
            attributes = clinical_presentation(incubation_period = Dirac(3.0)),
            progression = [Testing(assay = test_assay, delay = 0.0)]
        )
        @test_throws ArgumentError simulate(spec; max_cases = 20, rng = StableRNG(7), n_initial = 20)
    end

    @testset "a stochastic assay draws from the rng" begin
        always = Assay(value = load -> 30.0, detect = (rng, load) -> rand(rng) < 0.5)
        spec = testing_spec(progression = [Testing(assay = always, delay = 0.0)])
        state = simulate(spec; max_cases = 300, rng = StableRNG(8), n_initial = 20)
        tested = filter(ind -> is_infected(ind) && ind.state[:test_tested], state.individuals)
        share = count(ind -> ind.state[:test_result], tested) / length(tested)
        @test 0.35 < share < 0.65
    end

    @testset "named tests keep separate keys and columns" begin
        # a second test, read from a different trajectory, as a serology titre
        titre(rng, ind) = t -> 10.0 * t
        titre_assay = Assay(value = v -> v, detect = 20.0)
        spec = ModelSpec(
            BranchingProcess(NegBin(2.5, 0.5), LogNormal(1.6, 0.5));
            attributes = [
                clinical_presentation(incubation_period = Dirac(3.0)),
                within_host(test_curve),
                within_host(titre; key = :titre),
            ],
            progression = [
                Testing(assay = test_assay, delay = 0.0, name = :pcr),
                Testing(assay = titre_assay, delay = 4.0, name = :serology, key = :titre),
            ]
        )
        state = simulate(spec; max_cases = 100, rng = StableRNG(9), n_initial = 20)
        cases = filter(ind -> is_infected(ind) && !is_asymptomatic(ind), state.individuals)
        @test !isempty(cases)
        for ind in cases
            st = ind.state
            @test st[:pcr_tested] && st[:serology_tested]
            @test st[:pcr_time] - ind.infection_time ≈ 3.0
            @test st[:serology_time] - ind.infection_time ≈ 7.0
            @test st[:pcr_value] ≈ 40.0 - 3.0 * test_curve(nothing, nothing)(3.0)
            @test st[:serology_value] ≈ 70.0
            @test st[:pcr_result]
            @test st[:serology_result]
            @test !haskey(st, :test_time)
        end
        ll = linelist(state)
        for col in (
                :pcr_tested, :date_pcr, :pcr_value, :pcr_result,
                :serology_tested, :date_serology, :serology_value, :serology_result,
            )
            @test col in propertynames(ll)
        end
    end
end

@testset "Testing likelihood" begin
    load_at(ind, time) = test_curve(nothing, ind)(time - ind.infection_time)
    noisy = Assay(
        value = load -> 40.0 - 3.0 * load,
        detect = DetectionProbability(load -> 1 / (1 + exp(-(load - 2.0)))),
        noise = Normal(0.0, 1.5)
    )
    delay = Gamma(2.0, 1.0)
    spec = testing_spec(
        progression = [Testing(assay = noisy, delay = delay, probability = 0.8)]
    )
    state = simulate(spec; max_cases = 100, rng = StableRNG(2), n_initial = 20)

    @testset "matches the probability, delay and assay terms case by case" begin
        expected = 0.0
        for ind in individuals(state)
            is_infected(ind) && !is_asymptomatic(ind) || continue
            st = ind.state
            if st[:test_tested]
                load = load_at(ind, st[:test_time])
                p = 1 / (1 + exp(-(load - 2.0)))
                expected += log(0.8) + logpdf(delay, st[:test_time] - st[:onset_time]) +
                    (st[:test_result] ? log(p) : log1p(-p)) +
                    logpdf(Normal(0.0, 1.5), st[:test_value] - (40.0 - 3.0 * load))
            else
                expected += log(0.2)
            end
        end
        @test any(ind -> ind.state[:test_tested], individuals(state))
        @test progression_loglik(spec, state) ≈ expected
    end

    @testset "the reading's density depends on the trajectory" begin
        ind = first(filter(i -> is_infected(i) && i.state[:test_tested], individuals(state)))
        step = only(spec.progression)
        base = EpiBranch.transition_loglik(step, ind)
        ind.state[:trajectory] = t -> 0.0
        @test EpiBranch.transition_loglik(step, ind) != base
    end

    @testset "a limit of detection is exact and a reading without noise must match" begin
        sharp = Testing(assay = test_assay, delay = 4.0)
        sharp_spec = testing_spec(progression = [sharp])
        sharp_state = simulate(sharp_spec; max_cases = 50, rng = StableRNG(3), n_initial = 10)
        @test isfinite(progression_loglik(sharp_spec, sharp_state))
        ind = first(filter(i -> is_infected(i) && i.state[:test_tested], individuals(sharp_state)))
        ind.state[:test_result] = !ind.state[:test_result]
        @test EpiBranch.transition_loglik(sharp, ind) == -Inf
        ind.state[:test_result] = !ind.state[:test_result]
        ind.state[:test_value] += 1.0
        @test EpiBranch.transition_loglik(sharp, ind) == -Inf
    end

    @testset "a detect function has no probability to score" begin
        coin = Assay(value = load -> 30.0, detect = (rng, load) -> rand(rng) < 0.5)
        coin_spec = testing_spec(progression = [Testing(assay = coin, delay = 4.0)])
        coin_state = simulate(coin_spec; max_cases = 20, rng = StableRNG(4), n_initial = 5)
        @test_throws ArgumentError progression_loglik(coin_spec, coin_state)
    end

    @testset "a test that was not taken adds the not-tested probability only" begin
        ind = first(filter(i -> is_infected(i) && !i.state[:test_tested] && !is_asymptomatic(i), individuals(state)))
        @test EpiBranch.transition_loglik(only(spec.progression), ind) ≈ log(0.2)
    end
end
