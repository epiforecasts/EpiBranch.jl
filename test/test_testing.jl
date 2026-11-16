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
                        onset_to_isolation_delay = 1.0, duration = Inf,
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

    @testset "numeric test_sensitivity is unchanged and keeps its closed form" begin
        iso = Isolation(
            onset_to_isolation_delay = Exponential(2.0), duration = Inf,
            test_sensitivity = 0.8
        )
        @test iso.test_sensitivity == 0.8
        testing = Testing(assay = test_assay, delay = 0.0)
        iso_t = Isolation(
            onset_to_isolation_delay = Exponential(2.0), duration = Inf,
            test_sensitivity = testing
        )
        @test_throws ArgumentError EpiBranch._closed_form_value(iso_t.test_sensitivity)
    end

    @testset "Isolation with a Testing step absent from the progression is reported" begin
        testing = Testing(assay = test_assay, delay = 0.0)
        spec = testing_spec(
            progression = [],
            interventions = [
                Isolation(
                    onset_to_isolation_delay = 1.0, duration = Inf,
                    test_sensitivity = testing,
                ),
            ]
        )
        @test_throws ArgumentError simulate(spec; max_cases = 20, rng = StableRNG(7), n_initial = 20)
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
