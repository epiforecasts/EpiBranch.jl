using Distributions
using LinearAlgebra
using StableRNGs

# Two equal age bands assigned alternately, so the `n_initial` infections the
# exact race places on the first ids split evenly between them, as the
# approximation splits them in proportion to size. A case's onset is its
# infection time, so a group's trigger is exactly its first infection.
function _vacc_spec(
        ; n = 2_000, mode = LeakyMode(), budget = nothing, delay = 3.0, wrap = identity,
        efficacy = 0.9, coverage = 1.0, group_key = :age_band, eligibility = OnSymptomOnset()
    )
    gv = GroupVaccination(;
        efficacy, mode, coverage, eligibility, group_key, dose_delay = delay
    )
    iv = budget === nothing ? gv :
        CapacityConstrained(gv; budget_per_period = budget, period = 1.0)
    return _vacc_model(n, [wrap(iv)])
end

function _vacc_model(n, ivs)
    band = (rng, ind) -> (ind.state[:age_band] = ind.id % 2 + 1)
    clinical = clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0)
    return ModelSpec(
        MixingProcess(;
            population_size = 2n, mixing_by = (:age_band,),
            rate = (from, to) -> (from == to ? 0.5 : 0.1) / n
        );
        progression = [Transition(:recovered; rate = 0.2, terminal = true)],
        attributes = [band, clinical], interventions = ivs
    )
end

struct _VaccOtherMode <: EpiBranch.AbstractEffectMode end

_vacc_infected(model, s) = sum(model.group_sizes) - sum(s.S) - sum(s.V) - sum(s.P)

@testset "Vaccination in the mixing approximations" begin
    # Doses of 2 800 a day over the days 3 to 10: the 4 000 people have their
    # doses fall due on day 3, three days after the first case, when 2 800 are
    # available.
    budget = Steps([3.0, 10.0], [0.0, 2_800.0, 0.0])

    @testset "agree with exact simulation of the same campaign in a large population" begin
        for mode in (LeakyMode(), AllOrNothingMode())
            spec = _vacc_spec(; mode, budget)
            model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
            approx = simulate(model; n_initial = 20, max_time = 80.0)
            runs = [
                simulate(spec; n_initial = 20, max_time = 80.0, rng = StableRNG(seed))
                    for seed in 1:4
            ]
            cases = [count(i -> isfinite(i.infection_time), r.individuals) for r in runs]
            doses = [count(is_vaccinated, r.individuals) for r in runs]
            # The exact doses are the budget, whole people, so the batches
            # the approximation gives add up to the same number.
            @test sum(approx.D) ≈ 2_800
            @test all(==(2_800), doses)
            @test isapprox(_vacc_infected(model, approx), sum(cases) / 4; atol = 0.06 * 4_000)
            # The campaign makes a difference: without it nearly everybody
            # is infected.
            unvaccinated = MixingMeanFieldProcess(_vacc_spec(; budget = 0.0); removal_rate = 0.2)
            bare = simulate(unvaccinated; n_initial = 20, max_time = 80.0)
            @test _vacc_infected(model, approx) < _vacc_infected(unvaccinated, bare) - 400
        end
    end

    @testset "a campaign with no doses changes nothing" begin
        plain = ModelSpec(
            MixingProcess(;
                population_size = 4_000, mixing_by = (:age_band,),
                rate = (from, to) -> (from == to ? 0.5 : 0.1) / 2_000
            );
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            attributes = (rng, ind) -> (ind.state[:age_band] = ind.id % 2 + 1)
        )
        without = MixingMeanFieldProcess(plain; removal_rate = 0.2)
        b = simulate(without; n_initial = 20, max_time = 40.0)
        lna_b = simulate(
            MixingLinearNoiseProcess(plain; removal_rate = 0.2); n_initial = 20, max_time = 40.0
        )
        for zero_dose in (
                _vacc_spec(; budget = 0.0),
                _vacc_spec(; budget = Steps([3.0], [0.0, 0.0])),
                _vacc_spec(; coverage = 0.0),
            )
            a = simulate(
                MixingMeanFieldProcess(zero_dose; removal_rate = 0.2);
                n_initial = 20, max_time = 40.0
            )
            @test a.S ≈ b.S
            @test a.I ≈ b.I
            @test a.R ≈ b.R
            @test all(iszero, a.V) && all(iszero, a.P)

            lna_a = simulate(
                MixingLinearNoiseProcess(zero_dose; removal_rate = 0.2);
                n_initial = 20, max_time = 40.0
            )
            @test lna_a.S ≈ lna_b.S
            @test lna_a.covariance[1:4, 1:4] ≈ lna_b.covariance[1:4, 1:4]
        end

        # A vaccine that does nothing still uses up doses, but the people who
        # get them are as susceptible as the rest.
        for mode in (LeakyMode(), AllOrNothingMode())
            useless = MixingMeanFieldProcess(
                _vacc_spec(; efficacy = 0.0, mode); removal_rate = 0.2
            )
            a = simulate(useless; n_initial = 20, max_time = 40.0)
            @test sum(a.D) ≈ 4_000
            @test a.S + a.V ≈ b.S
            @test a.I ≈ b.I
            @test a.R ≈ b.R
        end
    end

    @testset "a leaky dose scales the hazard; an all-or-nothing dose removes a share" begin
        leaky = MixingMeanFieldProcess(_vacc_spec(; mode = LeakyMode(), budget); removal_rate = 0.2)
        aon = MixingMeanFieldProcess(
            _vacc_spec(; mode = AllOrNothingMode(), budget); removal_rate = 0.2
        )
        for model in (leaky, aon)
            s = simulate(model; n_initial = 20, max_time = 30.0)
            @test s.S + s.V + s.I + s.R + s.P ≈ model.group_sizes
        end
        # Just after the batches on day 3: a leaky dose leaves everyone it
        # reached susceptible, an all-or-nothing one makes 90% of those it
        # reached immune and leaves the other 10% as exposed as before.
        l = simulate(leaky; n_initial = 20, max_time = 3.0)
        a = simulate(aon; n_initial = 20, max_time = 3.0)
        @test all(iszero, l.P)
        @test sum(l.D) ≈ 2_800
        @test sum(a.P) ≈ 9 * sum(a.V) rtol = 1.0e-6
        @test sum(l.V) ≈ sum(a.V) + sum(a.P)
        @test _vacc_infected(leaky, simulate(leaky; n_initial = 20, max_time = 80.0)) !=
            _vacc_infected(aon, simulate(aon; n_initial = 20, max_time = 80.0))
    end

    @testset "only the share of a dose that reaches a susceptible protects" begin
        # Everyone is dosed on day 0, 20 of them already infected: the dose
        # reaches them too but protects nobody who is not susceptible.
        spec = _vacc_spec(; efficacy = 1.0, mode = AllOrNothingMode(), delay = 0.0)
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        s = simulate(model; n_initial = 20, max_time = 0.001)
        @test sum(s.D) ≈ 4_000
        @test sum(s.P) ≈ 3_980 atol = 0.5
        @test sum(s.I) ≈ 20 atol = 0.5
    end

    @testset "a Scheduled window decides whether the doses are given" begin
        late = MixingMeanFieldProcess(
            _vacc_spec(; wrap = iv -> Scheduled(iv; start_time = 4.0)); removal_rate = 0.2
        )
        # The doses fall due on day 3, before the window opens: none are given.
        @test all(iszero, simulate(late; n_initial = 20, max_time = 30.0).D)
        inside = MixingMeanFieldProcess(
            _vacc_spec(; wrap = iv -> Scheduled(iv; start_time = 2.0, end_time = 5.0));
            removal_rate = 0.2
        )
        @test sum(simulate(inside; n_initial = 20, max_time = 30.0).D) ≈ 4_000
        closed = MixingMeanFieldProcess(
            _vacc_spec(; wrap = iv -> Scheduled(iv; end_time = 2.0)); removal_rate = 0.2
        )
        @test all(iszero, simulate(closed; n_initial = 20, max_time = 30.0).D)
    end

    @testset "coverage limits who can be reached" begin
        model = MixingMeanFieldProcess(_vacc_spec(; coverage = 0.5); removal_rate = 0.2)
        @test sum(simulate(model; n_initial = 20, max_time = 30.0).D) ≈ 2_000
    end

    @testset "groups share a budget in the order they were triggered" begin
        # Both groups' doses fall due on day 3, and the first group takes what
        # it can reach before the second gets what is left.
        model = MixingMeanFieldProcess(_vacc_spec(; budget); removal_rate = 0.2)
        @test simulate(model; n_initial = 20, max_time = 4.0).D == [2_000, 800]
        # A later trigger (more expected cases needed) delays the doses.
        slow = MixingMeanFieldProcess(
            _vacc_spec(; budget); removal_rate = 0.2,
            allocation = TriggeredInPriority(threshold = 100.0)
        )
        early = simulate(model; n_initial = 20, max_time = 0.1)
        late = simulate(slow; n_initial = 20, max_time = 0.1)
        @test all(==(3.0), early.campaign.dose_time)
        @test all(late.campaign.dose_time .> early.campaign.dose_time)
    end

    @testset "a budget can be shared by group size or by susceptibles" begin
        n = [1_000, 3_000]
        band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
        gv = GroupVaccination(efficacy = 0.9, group_key = :age_band)
        prog = [Transition(:recovered; rate = 0.2, terminal = true)]
        mixing = MixingProcess(;
            population_size = sum(n), mixing_by = (:age_band,),
            rate = (from, to) -> (from == to ? 1.0 : 0.1) / n[to[1]]
        )
        spec = ModelSpec(
            mixing; progression = prog, attributes = band,
            interventions = [
                CapacityConstrained(
                    gv; budget_per_period = Steps([10.0], [100.0, 0.0]), period = 1.0
                ),
            ]
        )
        by_size = MixingMeanFieldProcess(spec; removal_rate = 0.2, allocation = SplitByGroupSize())
        s = simulate(by_size; n_initial = 10, max_time = 10.0)
        @test sum(s.D) ≈ 1_000 rtol = 1.0e-3
        @test s.D[1] / s.D[2] ≈ n[1] / n[2] rtol = 1.0e-3
        # No doses once the window has closed.
        @test sum(simulate(by_size, s; max_time = 30.0).D) ≈ 1_000 rtol = 1.0e-3

        by_susceptibles = MixingMeanFieldProcess(
            spec; removal_rate = 0.2, allocation = SplitBySusceptibles()
        )
        u = simulate(by_susceptibles; n_initial = 10, max_time = 10.0)
        @test sum(u.D) ≈ 1_000 rtol = 1.0e-3
        # Group 1 mixes harder inside itself, so it loses susceptibles
        # faster and ends up with a smaller share of the doses than its size.
        @test u.D[1] / sum(u.D) < n[1] / sum(n)

        lna = MixingLinearNoiseProcess(spec; removal_rate = 0.2, allocation = SplitByGroupSize())
        l = simulate(lna; n_initial = 10, max_time = 10.0)
        @test l.D ≈ s.D
        @test all(>=(-1.0e-6), eigvals(Symmetric(l.covariance)))

        # A split needs a budget with a finite period to read a rate from.
        unlimited = ModelSpec(mixing; progression = prog, attributes = band, interventions = [gv])
        @test_throws ArgumentError MixingMeanFieldProcess(
            unlimited; removal_rate = 0.2, allocation = SplitByGroupSize()
        )
        total = ModelSpec(
            mixing; progression = prog, attributes = band,
            interventions = [CapacityConstrained(gv; budget_per_period = 500.0)]
        )
        @test_throws ArgumentError MixingMeanFieldProcess(
            total; removal_rate = 0.2, allocation = SplitBySusceptibles()
        )
    end

    @testset "a group_key spanning several mixing groups" begin
        regions = (rng, ind) -> begin
            ind.state[:age_band] = ind.id % 2 + 1
            ind.state[:region] = ind.id % 4 < 2 ? :north : :south
        end
        spec = ModelSpec(
            MixingProcess(;
                population_size = 4_000, mixing_by = (:region, :age_band),
                rate = (from, to) -> 0.3 / 2_000
            );
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            attributes = [
                regions,
                clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0),
            ],
            interventions = [
                GroupVaccination(efficacy = 0.9, eligibility = OnSymptomOnset(), group_key = :region),
            ]
        )
        model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
        @test length(model.groups) == 4
        s = simulate(model; n_initial = 20, max_time = 1.0)
        @test length(s.campaign.dose_time) == 2
        @test sum(simulate(model; n_initial = 20, max_time = 5.0).D) ≈ 4_000
    end

    @testset "continuing a state reproduces a straight run" begin
        for ctor in (MixingMeanFieldProcess, MixingLinearNoiseProcess)
            model = ctor(_vacc_spec(; mode = AllOrNothingMode(), budget); removal_rate = 0.2)
            direct = simulate(model; n_initial = 20, max_time = 30.0)
            first_part = simulate(model; n_initial = 20, max_time = 2.5)
            # Split before the doses fall due and again after.
            second = simulate(model, first_part; max_time = 3.5)
            split = simulate(model, second; max_time = 30.0)
            @test split.S ≈ direct.S rtol = 1.0e-3
            @test split.P ≈ direct.P rtol = 1.0e-3
            @test split.D ≈ direct.D rtol = 1.0e-3
            ctor === MixingLinearNoiseProcess &&
                @test split.covariance ≈ direct.covariance rtol = 1.0e-2
            # The state that was continued is left as it was.
            @test first_part.t == 2.5
            @test !any(first_part.campaign.fired)
        end
    end

    @testset "the linear noise mean path matches the mean field with a campaign" begin
        spec = _vacc_spec(; mode = AllOrNothingMode(), budget)
        mf = simulate(
            MixingMeanFieldProcess(spec; removal_rate = 0.2); n_initial = 20, max_time = 40.0
        )
        lna = simulate(
            MixingLinearNoiseProcess(spec; removal_rate = 0.2); n_initial = 20, max_time = 40.0
        )
        @test lna.S ≈ mf.S
        @test lna.V ≈ mf.V
        @test lna.P ≈ mf.P
        @test lna.covariance ≈ lna.covariance'
        @test all(>=(-1.0e-6), eigvals(Symmetric(lna.covariance)))
        @test size(lna.covariance) == (8, 8)
    end

    @testset "rejects what it cannot approximate" begin
        # `group_key` must be one the compartments are kept for, and the
        # message names both.
        err = try
            MixingMeanFieldProcess(_vacc_spec(; group_key = :group); removal_rate = 0.2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(":group", err.msg) && occursin("age_band", err.msg)
        @test_throws ArgumentError MixingLinearNoiseProcess(
            _vacc_spec(; group_key = :group); removal_rate = 0.2
        )

        # Other interventions would be silently left out.
        for other in (
                MassVaccination(; efficacy = 0.7, eligibility_time = 30.0),
                Isolation(onset_to_isolation_delay = 1.0, duration = 7.0),
            )
            @test_throws ArgumentError MixingMeanFieldProcess(
                _vacc_model(2_000, [other]); removal_rate = 0.2
            )
        end
        gv = GroupVaccination(efficacy = 0.9, group_key = :age_band)
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_model(2_000, [gv, gv]); removal_rate = 0.2
        )

        # Gates and inputs with no counterpart in a compartment.
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_spec(; wrap = iv -> Scheduled(iv; start_after_cases = 10));
            removal_rate = 0.2
        )
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_spec(; wrap = iv -> Scheduled(iv, s -> true)); removal_rate = 0.2
        )
        deferred = _vacc_model(
            2_000, [
                CapacityConstrained(
                    gv; budget_per_period = 100.0, period = 1.0, on_refusal = DeferRefused()
                ),
            ]
        )
        @test_throws ArgumentError MixingMeanFieldProcess(deferred; removal_rate = 0.2)
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_spec(; delay = Exponential(2.0)); removal_rate = 0.2
        )
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_model(
                2_000,
                [GroupVaccination(efficacy = 0.9, group_key = :age_band, coverage = (rng, ind) -> 0.5)]
            ); removal_rate = 0.2
        )
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_model(
                2_000,
                [GroupVaccination(efficacy = 0.9, group_key = :age_band, delay_to_immunity = 5.0)]
            ); removal_rate = 0.2
        )
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_model(
                2_000,
                [GroupVaccination(efficacy = 0.9, group_key = :age_band, waning = dt -> 0.9^dt)]
            ); removal_rate = 0.2
        )
        @test_throws ArgumentError TriggeredInPriority(threshold = 0.0)
    end

    @testset "a mode without a dose model is refused, not approximated" begin
        @test_throws ArgumentError MixingMeanFieldProcess(
            _vacc_spec(; mode = _VaccOtherMode()); removal_rate = 0.2
        )
    end
end

@testset "Initial immunity in the mixing approximations" begin
    n = [2_000, 8_000]
    band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
    spec = ModelSpec(
        MixingProcess(;
            population_size = sum(n), mixing_by = (:age_band,),
            rate = (from, to) -> (from == to ? 0.3 : 0.1) / n[to[1]]
        );
        progression = [Transition(:recovered; rate = 0.2, terminal = true)],
        attributes = band
    )
    model = MixingMeanFieldProcess(spec; removal_rate = 0.2)

    @testset "immune people are taken out of the susceptibles at the start" begin
        counted = simulate(
            model; n_initial = 10, immune = ImmuneCount([500, 4_000]), max_time = 1.0e-6
        )
        @test counted.P ≈ [500, 4_000]
        @test counted.S .+ counted.I .+ counted.P ≈ model.group_sizes atol = 1.0e-3
        shares = simulate(
            model; n_initial = 10, immune = ImmuneShare([0.25, 0.5]), max_time = 1.0e-6
        )
        @test shares.P ≈ [500, 4_000]
        everywhere = simulate(
            model; n_initial = 10, immune = ImmuneShare(0.5), max_time = 1.0e-6
        )
        @test everywhere.P ≈ [1_000, 4_000]
        same = simulate(model; n_initial = 10, immune = ImmuneCount(100), max_time = 1.0e-6)
        @test same.P ≈ [100, 100]
        lna = simulate(
            MixingLinearNoiseProcess(spec; removal_rate = 0.2);
            n_initial = 10, immune = ImmuneShare(0.5), max_time = 1.0e-6
        )
        @test lna.P ≈ [1_000, 4_000]
    end

    @testset "agrees with the final size relation of a single well-mixed group" begin
        N, β, γ = 10_000, 0.4, 0.2
        one_group = ModelSpec(
            MixingProcess(; population_size = N, rate = (from, to) -> β / N);
            progression = [Transition(:recovered; rate = γ, terminal = true)]
        )
        m = MixingMeanFieldProcess(one_group; removal_rate = γ)
        p = 0.3
        s = simulate(m; n_initial = 1, immune = ImmuneShare(p), max_time = 400.0)
        # With nobody infected at the end, the susceptibles left solve
        # S∞ = S0 exp(-(β/γ)(N - S∞ - P0)/N) for S0 = N(1 - p) - 1.
        S0 = N * (1 - p) - 1
        f(S) = S - S0 * exp(-(β / γ) * (N - S - N * p) / N)
        lo, hi = 0.0, S0
        for _ in 1:100
            mid = (lo + hi) / 2
            f(mid) > 0 ? (hi = mid) : (lo = mid)
        end
        @test only(s.S) ≈ lo rtol = 1.0e-3
        none = simulate(m; n_initial = 1, max_time = 400.0)
        @test only(s.S) + only(s.P) > only(none.S)
        herd = simulate(m; n_initial = 1, immune = ImmuneShare(0.6), max_time = 400.0)
        @test only(herd.S) > 0.99 * (N * 0.4 - 1)
    end

    @testset "rejects immunity it cannot place" begin
        @test_throws ArgumentError simulate(model; immune = ImmuneShare(1.5), max_time = 1.0)
        @test_throws ArgumentError simulate(model; immune = ImmuneShare([0.1]), max_time = 1.0)
        @test_throws ArgumentError simulate(
            model; immune = ImmuneCount([3_000, 0]), max_time = 1.0
        )
        @test_throws ArgumentError simulate(
            model; n_initial = 1_000, immune = ImmuneShare(1.0), max_time = 1.0
        )
    end
end
