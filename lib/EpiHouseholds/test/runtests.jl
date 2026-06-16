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
            Transition(:recovered; from = :infectious, delay = Gamma(6, 1), terminal = true)]
        mp = HouseholdProcess([3], Exponential(3.0); progression = prog)
        @test mp.from == :infectious
        @test length(mp.progression) == 2

        # a scalar and a distribution external hazard are both accepted
        @test HouseholdProcess([2], Exponential(1.0); external_hazard = 0.05) isa
              HouseholdProcess
        @test HouseholdProcess([2], Exponential(1.0); external_hazard = Exponential(20.0)) isa
              HouseholdProcess

        @test_throws ArgumentError HouseholdProcess([0, 2], Exponential(1.0))
        @test_throws ArgumentError HouseholdProcess([2], Exponential(1.0); external_hazard = -1.0)
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

        rows, _, is_ext = EpiHouseholds._survival_rows(data; external = true, obs_end = Tobs)
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
end
