using Test
using EpiHouseholds
using EpiBranch
using Distributions
using StableRNGs
using ForwardDiff

@testset "EpiHouseholds.jl" begin
    @testset "construction" begin
        m = HouseholdProcess([3, 4, 2], Exponential(3.0); infectious_period = 6.0)
        @test m isa HouseholdProcess
        @test household_sizes(m) == [3, 4, 2]
        @test length(m.household_of) == 9
        @test m.members[1] == [1, 2, 3]
        @test m.members[3] == [8, 9]
        @test m.from == :infection            # no latent period
        @test length(m.progression) == 1      # the recovery removal transition
        @test m.external_hazard == 0.0
        @test EpiHouseholds.interventions(m) == AbstractIntervention[]

        # a latent period anchors the kernel at :infectious and adds a transition
        ml = HouseholdProcess([3], Exponential(3.0);
            latent_period = LogNormal(1.0, 0.4), infectious_period = Gamma(6, 1))
        @test ml.from == :infectious
        @test length(ml.progression) == 2

        # an explicit progression is used as given, and sets the anchor
        prog = AbstractClinicalTransition[
            Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.4)),
            Transition(
                :recovered; from = :infectious, delay = Gamma(6, 1), terminal = true)]
        mp = HouseholdProcess([3], Exponential(3.0); progression = prog)
        @test mp.from == :infectious
        @test length(mp.progression) == 2

        # a scalar and a distribution external hazard are both accepted
        @test HouseholdProcess([2], Exponential(1.0); external_hazard = 0.05) isa
              HouseholdProcess
        @test HouseholdProcess(
            [2], Exponential(1.0); external_hazard = Exponential(20.0)) isa
              HouseholdProcess

        @test_throws ArgumentError HouseholdProcess([0, 2], Exponential(1.0))
        @test_throws ArgumentError HouseholdProcess(
            [2], Exponential(1.0); external_hazard = -1.0)
    end

    @testset "simulation seeds one index per household and spreads within it" begin
        m = HouseholdProcess(fill(4, 100), Weibull(1.3, 2.0); infectious_period = 6.0)
        state = simulate(m; rng = StableRNG(1))
        df = linelist(state)
        @test count(df.index) == 100              # one index per household
        @test 100 <= size(df, 1) <= 400           # indexes plus within-household spread
        @test issubset((:date_infection, :household, :index), propertynames(df))
    end

    @testset "the timeline is stamped through the progression (line-list columns)" begin
        # a latent period and an infectious period give onset and recovery dates
        m = HouseholdProcess(fill(4, 50), Exponential(2.0);
            latent_period = LogNormal(1.0, 0.3), infectious_period = Gamma(6, 1))
        df = linelist(simulate(m; rng = StableRNG(2)))
        @test :date_infectious in propertynames(df)   # :infectious_time → date_infectious
        @test :date_recovered in propertynames(df)     # the removal transition
    end

    @testset "no within-household spread when the kernel is far out of the period" begin
        # contact intervals almost never fall within a tiny infectious period,
        # so only the index cases are infected.
        m = HouseholdProcess(fill(5, 200), Exponential(50.0); infectious_period = 0.001)
        df = linelist(simulate(m; rng = StableRNG(3)))
        @test 200 <= size(df, 1) <= 205
    end

    @testset "final size grows with transmissibility" begin
        sizes = fill(5, 300)
        low = HouseholdProcess(sizes, Exponential(20.0); infectious_period = 6.0)
        high = HouseholdProcess(sizes, Exponential(2.0); infectious_period = 6.0)
        n_low = size(linelist(simulate(low; rng = StableRNG(4))), 1)
        n_high = size(linelist(simulate(high; rng = StableRNG(4))), 1)
        @test n_high > n_low
    end

    @testset "external force of infection introduces community cases" begin
        m = HouseholdProcess(fill(4, 300), Exponential(3.0);
            infectious_period = 6.0, external_hazard = 0.05)
        df = linelist(simulate(m; rng = StableRNG(5), obs_end = 30.0))
        @test count(df.index) >= 1                 # community introductions happened
        @test size(df, 1) > count(df.index)        # plus within-household spread
        @test count(df.index) != length(m.members) # not the one-index-per-household fallback
    end

    @testset "pairwise survival likelihood: basics and differentiability" begin
        rows = PairwiseSurvivalData([1, 1, 2, 2], [0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 1.5, 5.0], [true, false, true, false])
        @test isfinite(pairwise_surv_loglik(Exponential(3.0), rows))
        # a constant per-row callable equals the shared distribution
        @test pairwise_surv_loglik(r -> Exponential(3.0), rows) ≈
              pairwise_surv_loglik(Exponential(3.0), rows)
        # differentiable in a log-rate parameter (what makes it fittable)
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ)), rows)
        g = ForwardDiff.derivative(f, log(3.0))
        @test isfinite(g)
        fd = (f(log(3.0) + 1e-6) - f(log(3.0) - 1e-6)) / 2e-6
        @test isapprox(g, fd; rtol = 1e-4)
    end

    @testset "simulate → loglikelihood round trip recovers the kernel" begin
        # the Sellke construction is the generative model the pairwise likelihood
        # assumes, so the simulated infection layer recovers the kernel scale.
        true_scale = 4.0
        L = 6.0
        m = HouseholdProcess(fill(4, 1500), Exponential(true_scale); infectious_period = L)
        state = simulate(m; rng = StableRNG(20260615))
        data = household_infections(state, m)
        @test count(data.is_index) == 1500            # one index per household

        rows = first(EpiHouseholds._survival_rows(data))
        ll(s) = pairwise_surv_loglik(Exponential(s), rows)
        @test ll(true_scale) > ll(true_scale / 2)
        @test ll(true_scale) > ll(true_scale * 2)
        grid = 2.0:0.5:6.0
        @test abs(grid[argmax([ll(s) for s in grid])] - true_scale) <= 1.0

        # the dispatched loglikelihood routes through pairwise_surv_loglik
        @test loglikelihood(data, m) ≈ ll(true_scale)
    end

    @testset "external (community) term: round trip recovers the kernel" begin
        # with a community hazard, indexes emerge from it and the within-household
        # kernel scale is still recovered through the same likelihood.
        true_scale = 3.0
        L = 6.0
        Tobs = 30.0
        m = HouseholdProcess(fill(4, 1500), Exponential(true_scale);
            infectious_period = L, external_hazard = 0.05)
        state = simulate(m; rng = StableRNG(7), obs_end = Tobs)
        data = household_infections(state, m; obs_end = Tobs)
        @test count(data.is_index) >= 1
        @test isfinite(loglikelihood(data, m))          # dispatched form with external

        rows, _, is_ext = EpiHouseholds._survival_rows(
            data; external = true, obs_end = Tobs)
        extdist = Exponential(1 / 0.05)
        ll(s) = pairwise_surv_loglik(r -> is_ext[r] ? extdist : Exponential(s), rows)
        grid = 1.5:0.5:5.0
        @test abs(grid[argmax([ll(s) for s in grid])] - true_scale) <= 1.5
    end

    @testset "inference-friendly likelihood: kernel varies over a fixed infection layer" begin
        # the form a household @model evaluates each iteration: the kernel carries
        # the fitted parameter, the infection layer is the augmented latent state.
        m = HouseholdProcess(fill(4, 400), Exponential(3.0); infectious_period = 6.0)
        data = household_infections(simulate(m; rng = StableRNG(8)), m)

        f(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data)
        @test f(log(1 / 3)) ≈ loglikelihood(data, m)            # matches the model-dispatch sugar
        g = ForwardDiff.derivative(f, log(1 / 3))               # gradient for HMC
        @test isfinite(g)
        grid = 2.0:0.5:6.0
        @test abs(grid[argmax([f(log(1 / s)) for s in grid])] - 3.0) <= 1.5

        # composes with a progression observation term (onset = infection + incubation)
        incubation = LogNormal(1.0, 0.3)
        obs = findall(!isnan, data.infection_time)
        onset = copy(data.infection_time)
        onset[obs] .+= rand(StableRNG(9), incubation, length(obs))
        joint(logβ) = f(logβ) + sum(logpdf(incubation, onset[i] - data.infection_time[i])
        for i in obs)
        @test isfinite(joint(log(1 / 3)))
    end

    @testset "the model's observation is applied in simulate" begin
        # a HouseholdProcess carrying a PerCaseObservation reports cases through
        # the shared observation protocol, like core simulate.
        obs = PerCaseObservation(; detection_prob = 0.5, delay = Exponential(2.0))
        m = HouseholdProcess(fill(4, 200), Exponential(3.0);
            infectious_period = 6.0, observation = obs)
        df = linelist(simulate(m; rng = StableRNG(11)))
        @test :reported in propertynames(df)            # observation ran
        @test 0 < count(df.reported) < size(df, 1)       # ~half detected, not all
    end

    @testset "compiled pair layout matches the dynamic path (shared kernel)" begin
        # the layout captures the fixed row structure once; evaluating it over a
        # grid of kernel scales must reproduce the dynamic HouseholdInfections
        # path exactly (up to row order) while the same layout object is reused.
        m = HouseholdProcess(fill(4, 500), Exponential(3.0); infectious_period = 6.0)
        data = household_infections(simulate(m; rng = StableRNG(101)), m)
        layout = compile_household_pairs(data)

        @test layout isa HouseholdPairsLayout
        @test !layout.external
        @test length(layout) == length(layout.sus)
        @test length(layout) > 0                          # there is real spread to score

        for s in 1.5:0.5:6.0
            @test pairwise_surv_loglik(Exponential(s), data, layout) ≈
                  pairwise_surv_loglik(Exponential(s), data)
        end

        # single-arg constructor reads the at-risk mask off the data and agrees
        layout1 = compile_household_pairs(data.household_of, data.is_index,
            .!isnan.(data.infection_time))
        @test length(layout1) == length(layout)
        @test pairwise_surv_loglik(Exponential(3.0), data, layout1) ≈
              pairwise_surv_loglik(Exponential(3.0), data, layout)
    end

    @testset "compiled pair layout matches the dynamic path (external hazard)" begin
        # with a community term every susceptible also carries an external row;
        # the layout must be built with external=true and agree with the dynamic
        # external path across kernel scales.
        Tobs = 30.0
        m = HouseholdProcess(fill(4, 500), Exponential(3.0);
            infectious_period = 6.0, external_hazard = 0.05)
        state = simulate(m; rng = StableRNG(102), obs_end = Tobs)
        data = household_infections(state, m; obs_end = Tobs)
        layout = compile_household_pairs(data; external = true)

        @test layout.external
        for s in 1.5:0.5:5.0
            @test pairwise_surv_loglik(Exponential(s), data, layout;
                external_hazard = 0.05) ≈
                  pairwise_surv_loglik(Exponential(s), data; external_hazard = 0.05)
        end

        # a distribution-valued community hazard routes the same way
        @test pairwise_surv_loglik(Exponential(3.0), data, layout;
            external_hazard = Exponential(20.0)) ≈
              pairwise_surv_loglik(Exponential(3.0), data;
            external_hazard = Exponential(20.0))
    end

    @testset "compiled pair layout: covariate (per-pair) kernel" begin
        # a two-argument (infector, susceptible) -> Distribution kernel routes
        # through _pair on both the dynamic and the compiled path, so they agree.
        m = HouseholdProcess(fill(4, 300), Exponential(3.0); infectious_period = 6.0)
        data = household_infections(simulate(m; rng = StableRNG(103)), m)
        layout = compile_household_pairs(data)

        # a mild dependence on the pair ids exercises the routing, not the physics
        kern(i, j) = Exponential(3.0 + 0.01 * (i + j))
        @test pairwise_surv_loglik(kern, data, layout) ≈
              pairwise_surv_loglik(kern, data)
    end

    @testset "compiled pair layout: differentiable and matches dynamic gradient" begin
        # the fast path exists to be differentiated in the kernel parameters
        # (its whole reason for being reused across gradient evaluations). The
        # fitted parameter rides the kernel, not the data, so the layout must
        # carry the AD duals through both the cumulative-hazard pass and the
        # per-susceptible log-sum-exp. Checked in all three kernel modes.
        m = HouseholdProcess(fill(4, 300), Exponential(3.0); infectious_period = 6.0)
        data = household_infections(simulate(m; rng = StableRNG(104)), m)
        layout = compile_household_pairs(data)

        f_fast(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data, layout)
        f_dyn(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data)
        @test f_fast(log(1 / 3)) ≈ f_dyn(log(1 / 3))

        g_fast = ForwardDiff.derivative(f_fast, log(1 / 3))
        g_dyn = ForwardDiff.derivative(f_dyn, log(1 / 3))
        @test isfinite(g_fast)
        @test g_fast ≈ g_dyn

        # covariate (per-pair) kernel: the parameter still rides the kernel, now
        # resolved per pair — the accumulator must promote to hold it.
        kern(β) = (i, j) -> Exponential(1 / (exp(β) * (1 + 0.001 * (i + j))))
        h_fast(β) = pairwise_surv_loglik(kern(β), data, layout)
        h_dyn(β) = pairwise_surv_loglik(kern(β), data)
        @test ForwardDiff.derivative(h_fast, log(1 / 3)) ≈
              ForwardDiff.derivative(h_dyn, log(1 / 3))

        # external mode: differentiate the within-household kernel while a fixed
        # community hazard also contributes rows.
        Tobs = 30.0
        me = HouseholdProcess(fill(4, 300), Exponential(3.0);
            infectious_period = 6.0, external_hazard = 0.05)
        de = household_infections(simulate(me; rng = StableRNG(106), obs_end = Tobs),
            me; obs_end = Tobs)
        le = compile_household_pairs(de; external = true)
        e_fast(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), de, le;
            external_hazard = 0.05)
        e_dyn(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), de;
            external_hazard = 0.05)
        @test e_fast(log(1 / 3)) ≈ e_dyn(log(1 / 3))
        @test ForwardDiff.derivative(e_fast, log(1 / 3)) ≈
              ForwardDiff.derivative(e_dyn, log(1 / 3))
    end

    @testset "compiled pair layout: hand-built multi-infector household" begin
        # one household, member 3 infected late with two eligible infectors
        # (members 1 and 2) — exercises the per-susceptible log-sum-exp over more
        # than one event row, plus a single-infector susceptible and a conditioned
        # index case.
        data = HouseholdInfections([1, 1, 1], [0.0, 0.5, 2.0], [0.0, 0.5, 2.0],
            [Inf, Inf, Inf], [true, false, false])
        layout = compile_household_pairs(data)

        # the layout enumerates every ordered (susceptible, infector) structural
        # pair among the non-index members: 2←{1,3} and 3←{1,2}, i.e. four rows —
        # even pairs whose timing later contributes nothing are kept, and pruned
        # on the fly at evaluation. The index (member 1) is conditioned on.
        @test length(layout) == 4
        @test 3 in layout.sus_unique && 2 in layout.sus_unique
        @test !(1 in layout.sus_unique)

        for s in 1.0:1.0:5.0
            @test pairwise_surv_loglik(Exponential(s), data, layout) ≈
                  pairwise_surv_loglik(Exponential(s), data)
        end
    end

    @testset "compiled pair layout: edge cases" begin
        # empty population → empty layout, zero log-likelihood, consistent length
        empty = HouseholdInfections(Int[], Float64[], Float64[], Float64[], Bool[])
        elayout = compile_household_pairs(empty)
        @test length(elayout) == 0
        @test pairwise_surv_loglik(Exponential(3.0), empty, elayout) == 0.0
        @test compile_household_pairs(Int[], Bool[], Bool[]) isa HouseholdPairsLayout

        # a household where the sole housemate escapes: the index recovers at
        # t=3 and member 2 is never infected. There is still one structural row
        # (member 2 at risk from the index), whose only contribution is the
        # escaped cumulative hazard — finite and equal to the dynamic path.
        lone = HouseholdInfections([1, 1], [0.0, NaN], [0.0, NaN], [3.0, Inf],
            [true, false])
        llayout = compile_household_pairs(lone)
        @test length(llayout) == 1
        ll_lone = pairwise_surv_loglik(Exponential(3.0), lone, llayout)
        @test isfinite(ll_lone)
        @test ll_lone < 0                                 # pure escaped hazard
        @test ll_lone ≈ pairwise_surv_loglik(Exponential(3.0), lone)

        # calling the external-built layout without an external hazard (and vice
        # versa) is a mismatch and must raise
        m = HouseholdProcess(fill(4, 50), Exponential(3.0); infectious_period = 6.0)
        data = household_infections(simulate(m; rng = StableRNG(105)), m)
        ext_layout = compile_household_pairs(data; external = true)
        int_layout = compile_household_pairs(data; external = false)
        @test_throws ArgumentError pairwise_surv_loglik(Exponential(3.0), data,
            ext_layout)                                   # no external_hazard given
        @test_throws ArgumentError pairwise_surv_loglik(Exponential(3.0), data,
            int_layout; external_hazard = 0.05)

        # mismatched input lengths are rejected at compile time
        @test_throws ArgumentError compile_household_pairs([1, 1], [true],
            [true, false])
    end
end
