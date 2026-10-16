@testset "Incidence likelihood" begin
    @testset "IncidenceCounts constructor" begin
        @test IncidenceCounts([1, 2, 3]).data == [1, 2, 3]
        @test IncidenceCounts([1, 2, 3]).seeds == 1
        @test IncidenceCounts([1, 2, 3]; seeds = 5).seeds == 5
        @test_throws ArgumentError IncidenceCounts(Int[])
        @test_throws ArgumentError IncidenceCounts([1, -1, 2])
        @test_throws ArgumentError IncidenceCounts([1, 2]; seeds = 0)
    end

    @testset "Discretised generation interval" begin
        w = EpiBranch._discretize_generation_interval(LogNormal(1.6, 0.5))
        @test sum(w) ≈ 1.0
        @test all(>=(0), w)

        # A generation time with (almost) all its mass on lag 1 gives a pmf
        # concentrated there.
        w2 = EpiBranch._discretize_generation_interval(Dirac(1.0))
        @test w2[1] ≈ 1.0
        @test all(x -> x ≈ 0, w2[2:end])
    end

    @testset "incidence_moments matches the renewal equation for the mean" begin
        R, gt, n, seeds = 1.4, LogNormal(1.0, 0.4), 8, 4
        w = EpiBranch._discretize_generation_interval(gt)
        μ_renewal = zeros(Float64, n)
        get_μ(t) = t == 0 ? Float64(seeds) : (t < 0 ? 0.0 : μ_renewal[t])
        for t in 1:n
            μ_renewal[t] = R * sum(w[j] * get_μ(t - j) for j in 1:length(w))
        end

        m = incidence_moments(Poisson(R), gt, n; seeds = seeds)
        @test m.mean ≈ μ_renewal
    end

    @testset "incidence_moments matches a discrete-time branching simulation" begin
        # A hand-rolled discrete-time simulation sharing exactly the
        # assumptions `incidence_moments` encodes (each step-k individual
        # draws Poisson offspring, each landing at step k + lag with
        # lag ~ Categorical(w)): this isolates the recursion's correctness
        # from the separate approximation of binning a continuous
        # generation time into calendar steps (see the "Limits" note on
        # `loglikelihood(::IncidenceCounts, ::Distribution, ::Distribution)`).
        R, gt, n, seeds = 1.4, LogNormal(1.0, 0.4), 6, 3
        w = EpiBranch._discretize_generation_interval(gt)
        m = incidence_moments(Poisson(R), gt, n; seeds = seeds)

        function simulate_discrete(rng, R, w, n, seeds)
            cohort = Dict(0 => seeds)
            for k in 0:(n - 1)
                haskey(cohort, k) || continue
                for _ in 1:cohort[k]
                    for _ in 1:rand(rng, Poisson(R))
                        day = k + rand(rng, Categorical(w))
                        day <= n && (cohort[day] = get(cohort, day, 0) + 1)
                    end
                end
            end
            return [get(cohort, day, 0) for day in 1:n]
        end

        nsim = 8000
        sims = reduce(
            hcat, (simulate_discrete(StableRNG(i), R, w, n, seeds) for i in 1:nsim)
        )'
        emp_mean = vec(sum(sims, dims = 1)) ./ nsim
        emp_cov = [
            sum((sims[:, t] .- emp_mean[t]) .* (sims[:, t′] .- emp_mean[t′])) / (nsim - 1)
                for t in 1:n, t′ in 1:n
        ]

        @test maximum(abs.(emp_mean .- m.mean)) < 0.3
        @test maximum(abs.(emp_cov .- m.cov)) < 0.8
    end

    @testset "Discretised reporting delay" begin
        @test EpiBranch._discretize_report_delay(nothing) == [1.0]
        d = EpiBranch._discretize_report_delay(Gamma(2.0, 1.5))
        @test sum(d) ≈ 1.0
        @test d[1] ≈ 0.0
        # A fixed delay of two steps puts all its mass on lag 2.
        d2 = EpiBranch._discretize_report_delay(2)
        @test d2[3] ≈ 1.0
        @test sum(d2) ≈ 1.0
        @test EpiBranch._discretize_report_delay(0) == [1.0]
        @test_throws ArgumentError EpiBranch._discretize_report_delay(Normal(1.0, 1.0))
    end

    @testset "Full reporting leaves the moments unchanged" begin
        offspring, gt, n = NegBin(1.3, 0.5), LogNormal(1.0, 0.4), 7
        m = incidence_moments(offspring, gt, n; seeds = 2)
        m1 = incidence_moments(
            offspring, gt, n; seeds = 2, reporting_probability = 1, reporting_delay = 0
        )
        @test m1.mean ≈ m.mean
        @test m1.cov ≈ m.cov
        @test_throws ArgumentError incidence_moments(
            offspring, gt, n; reporting_probability = 0
        )
        @test_throws ArgumentError incidence_moments(
            offspring, gt, n; reporting_probability = 1.2
        )
    end

    @testset "Reported means thin and shift the infection means" begin
        R, gt, n, seeds, ρ = 1.4, LogNormal(1.0, 0.4), 8, 3, 0.6
        inf = incidence_moments(Poisson(R), gt, n; seeds)
        thinned = incidence_moments(Poisson(R), gt, n; seeds, reporting_probability = ρ)
        @test thinned.mean ≈ ρ .* inf.mean
        # Binomial thinning: Var = ρ²·Var + ρ(1 - ρ)·mean, and covariances
        # between steps scale by ρ².
        @test [thinned.cov[t, t] for t in 1:n] ≈
            ρ^2 .* [inf.cov[t, t] for t in 1:n] .+ ρ * (1 - ρ) .* inf.mean
        @test thinned.cov[2, 5] ≈ ρ^2 * inf.cov[2, 5]

        # A delay convolves the infection means, the seeds' own reports
        # included once they land in an observed step.
        delay = Gamma(2.0, 1.0)
        d = EpiBranch._discretize_report_delay(delay)
        delayed = incidence_moments(
            Poisson(R), gt, n; seeds, reporting_probability = ρ, reporting_delay = delay
        )
        infected(t) = t == 0 ? Float64(seeds) : (t < 0 ? 0.0 : inf.mean[t])
        expected = [
            ρ * sum(d[l + 1] * infected(t - l) for l in 0:(length(d) - 1)) for t in 1:n
        ]
        @test delayed.mean ≈ expected
    end

    @testset "Reported moments match a discrete-time simulation with reporting" begin
        R, gt, n, seeds, ρ = 1.4, LogNormal(1.0, 0.4), 6, 3, 0.6
        delay = Gamma(2.0, 1.0)
        w = EpiBranch._discretize_generation_interval(gt)
        d = EpiBranch._discretize_report_delay(delay)
        m = incidence_moments(
            Poisson(R), gt, n; seeds, reporting_probability = ρ, reporting_delay = delay
        )

        function simulate_reported(rng, R, w, d, ρ, n, seeds)
            cohort = Dict(0 => seeds)
            reports = zeros(Int, n)
            for k in 0:n
                haskey(cohort, k) || continue
                for _ in 1:cohort[k]
                    if rand(rng) < ρ
                        day = k + rand(rng, Categorical(d)) - 1
                        1 <= day <= n && (reports[day] += 1)
                    end
                    k < n || continue
                    for _ in 1:rand(rng, Poisson(R))
                        day = k + rand(rng, Categorical(w))
                        day <= n && (cohort[day] = get(cohort, day, 0) + 1)
                    end
                end
            end
            return reports
        end

        nsim = 8000
        sims = reduce(
            hcat, (simulate_reported(StableRNG(i), R, w, d, ρ, n, seeds) for i in 1:nsim)
        )'
        emp_mean = vec(sum(sims, dims = 1)) ./ nsim
        emp_cov = [
            sum((sims[:, t] .- emp_mean[t]) .* (sims[:, t′] .- emp_mean[t′])) / (nsim - 1)
                for t in 1:n, t′ in 1:n
        ]

        @test maximum(abs.(emp_mean .- m.mean)) < 0.2
        @test maximum(abs.(emp_cov .- m.cov)) < 0.5
    end

    @testset "loglikelihood with offspring and generation time" begin
        data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18]; seeds = 2)
        gt = LogNormal(1.6, 0.5)
        offspring = NegBin(1.3, 0.5)

        ll_normal = loglikelihood(data, offspring, gt)
        @test isfinite(ll_normal)
        m = incidence_moments(offspring, gt, length(data.data); seeds = data.seeds)
        @test ll_normal ≈ logpdf(MvNormal(m.mean, m.cov), data.data)

        ll_lognormal = loglikelihood(data, offspring, gt, LognormalApproximation())
        @test isfinite(ll_lognormal)
        @test ll_lognormal != ll_normal

        # GeneratingFunctionOffspring shares the same `_pgf` dispatch, so the
        # unified offspring argument covers it too.
        gfo = GeneratingFunctionOffspring(s -> exp(1.3 * (s - 1)), 1.3)
        @test loglikelihood(data, gfo, gt) ≈ loglikelihood(data, Poisson(1.3), gt)

        mr = incidence_moments(
            offspring, gt, length(data.data); seeds = data.seeds,
            reporting_probability = 0.6, reporting_delay = Gamma(2.0, 1.5)
        )
        @test loglikelihood(
            data, offspring, gt; reporting_probability = 0.6, reporting_delay = Gamma(2.0, 1.5)
        ) ≈ logpdf(MvNormal(mr.mean, mr.cov), data.data)
    end

    @testset "LognormalApproximation needs strictly positive counts" begin
        data = IncidenceCounts([0, 2, 4, 5])
        @test_throws ArgumentError loglikelihood(
            data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5), LognormalApproximation()
        )
    end

    @testset "loglikelihood on a model" begin
        data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18]; seeds = 2)
        offspring = NegBin(1.3, 0.5)
        gt = LogNormal(1.6, 0.5)
        model = BranchingProcess(offspring, gt)

        @test loglikelihood(data, model) ≈ loglikelihood(data, offspring, gt)
        @test loglikelihood(data, ModelSpec(model)) ≈ loglikelihood(data, offspring, gt)
        @test loglikelihood(data, model, LognormalApproximation()) ≈
            loglikelihood(data, offspring, gt, LognormalApproximation())

        @test_throws ArgumentError loglikelihood(data, BranchingProcess(offspring))

        with_iv = ModelSpec(
            model;
            interventions = [Isolation(onset_to_isolation_delay = 1.0, duration = 14.0)]
        )
        @test_throws ArgumentError loglikelihood(data, with_iv)

        with_obs = ModelSpec(model; observation = PerCaseObservation(detection_prob = 0.7))
        @test_throws ArgumentError loglikelihood(data, with_obs)

        with_reporting = ModelSpec(
            model;
            progression = [
                Reporting(
                    delay = 0.0, probability = 0.7, from = ind -> ind.infection_time
                ),
            ]
        )
        @test_throws ArgumentError loglikelihood(data, with_reporting)

        multi_window = BranchingProcess(
            Infectiousness(offspring; kernel = gt), Infectiousness(offspring; kernel = gt)
        )
        @test_throws ArgumentError loglikelihood(data, multi_window)
    end
end
