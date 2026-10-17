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

    @testset "IncidenceCounts by type" begin
        by_type = IncidenceCounts([0 1; 1 1; 1 3]; seeds = [0, 1])
        @test by_type.data == [0 1; 1 1; 1 3]
        @test by_type.seeds == [0, 1]
        @test IncidenceCounts([1, 2]; seeds = [1, 0]).seeds == [1, 0]
        @test_throws ArgumentError IncidenceCounts([0 1; 1 1])
        @test_throws ArgumentError IncidenceCounts([0 1; 1 1]; seeds = [1, 0, 0])
        @test_throws ArgumentError IncidenceCounts([1, 2]; seeds = [0, 0])
        @test_throws ArgumentError IncidenceCounts([1, 2]; seeds = [1, -1])
    end

    @testset "Multi-type incidence moments" begin
        gt = LogNormal(1.0, 0.4)
        w = EpiBranch._discretize_generation_interval(gt)

        # One type is the single-type process.
        one_type = EpiBranch.MultiTypeOffspring(reshape([1.3], 1, 1), R -> NegBin(R, 0.5))
        a = incidence_moments(one_type, gt, 6; seeds = [2], reporting_probability = 0.7)
        b = incidence_moments(NegBin(1.3, 0.5), gt, 6; seeds = 2, reporting_probability = 0.7)
        @test vec(a.mean) ≈ b.mean
        @test a.cov ≈ b.cov

        # The means follow the multi-type renewal equation.
        M = [1.2 0.4; 0.3 0.9]
        o = EpiBranch.MultiTypeOffspring(M, R -> NegBin(R, 0.5))
        n, seeds = 7, [1, 2]
        m = incidence_moments(o, gt, n; seeds)
        μ = zeros(n, 2)
        past(t) = t == 0 ? Float64.(seeds) : (t < 0 ? zeros(2) : μ[t, :])
        for t in 1:n
            μ[t, :] = M * sum(w[l] * past(t - l) for l in 1:length(w))
        end
        @test m.mean ≈ μ
        @test size(m.cov) == (2n, 2n)

        @test_throws ArgumentError incidence_moments(o, gt, n; seeds = [1])
        @test_throws ArgumentError incidence_moments(o, gt, n; seeds = [0, 0])
    end

    @testset "Multi-type moments match a discrete-time simulation" begin
        M = [1.0 0.5; 0.3 0.8]
        o = EpiBranch.MultiTypeOffspring(M, R -> Poisson(R))
        gt, n, seeds, ρ = LogNormal(1.0, 0.4), 5, [2, 1], 0.7
        w = EpiBranch._discretize_generation_interval(gt)
        m = incidence_moments(o, gt, n; seeds, reporting_probability = ρ)

        function simulate_types(rng)
            cohort = zeros(Int, n + 1, 2)
            cohort[1, :] = seeds
            reports = zeros(Int, n, 2)
            for k in 0:n, j in 1:2
                for _ in 1:cohort[k + 1, j]
                    k >= 1 && rand(rng) < ρ && (reports[k, j] += 1)
                    k < n || continue
                    for _ in 1:rand(rng, Poisson(o.R_by_type[j]))
                        i = rand(rng, Categorical(o.alloc_probs[:, j]))
                        day = k + rand(rng, Categorical(w))
                        day <= n && (cohort[day + 1, i] += 1)
                    end
                end
            end
            return vec(reports)
        end

        nsim = 8000
        sims = reduce(hcat, (simulate_types(StableRNG(i)) for i in 1:nsim))'
        emp_mean = vec(sum(sims, dims = 1)) ./ nsim
        emp_cov = [
            sum((sims[:, a] .- emp_mean[a]) .* (sims[:, b] .- emp_mean[b])) / (nsim - 1)
                for a in 1:(2n), b in 1:(2n)
        ]
        @test maximum(abs.(emp_mean .- vec(m.mean))) < 0.2
        @test maximum(abs.(emp_cov .- m.cov)) < 0.5
    end

    @testset "Multi-type loglikelihood" begin
        M = [1.2 0.4; 0.3 0.9]
        o = EpiBranch.MultiTypeOffspring(M, R -> NegBin(R, 0.5))
        gt = LogNormal(1.6, 0.5)
        counts = [0 1; 1 1; 1 3; 2 4; 3 6]
        by_type = IncidenceCounts(counts; seeds = [0, 1])
        total = IncidenceCounts(vec(sum(counts; dims = 2)); seeds = [0, 1])
        m = incidence_moments(o, gt, 5; seeds = [0, 1])

        @test loglikelihood(by_type, o, gt) ≈
            logpdf(MvNormal(vec(m.mean), m.cov), vec(counts))
        # The total's moments sum the per-type blocks.
        A = [Float64(t == s % 5 + 1) for t in 1:5, s in 0:9]
        Σ_total = A * m.cov * A'
        @test loglikelihood(total, o, gt) ≈
            logpdf(MvNormal(A * vec(m.mean), (Σ_total + Σ_total') / 2), total.data)
        @test isfinite(loglikelihood(total, o, gt, LognormalApproximation()))

        model = BranchingProcess(M, R -> NegBin(R, 0.5), gt)
        @test loglikelihood(by_type, model) ≈ loglikelihood(by_type, o, gt)
        @test loglikelihood(total, ModelSpec(model)) ≈ loglikelihood(total, o, gt)

        @test_throws ArgumentError loglikelihood(IncidenceCounts([1, 2, 4]), o, gt)
        @test_throws ArgumentError loglikelihood(
            IncidenceCounts([0 1 1; 1 1 1]; seeds = [0, 1, 0]), o, gt
        )
        @test_throws ArgumentError loglikelihood(by_type, NegBin(1.3, 0.5), gt)
        @test_throws ArgumentError loglikelihood(total, NegBin(1.3, 0.5), gt)
    end

    @testset "Counts that can only be zero are left out" begin
        # With no type-1 seed and every report delayed, type 1 has no report
        # in the first step: mean and variance 0 there.
        o = EpiBranch.MultiTypeOffspring([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5))
        gt, rd = LogNormal(1.6, 0.5), Gamma(2.0, 1.5)
        kw = (reporting_probability = 0.7, reporting_delay = rd)
        m = incidence_moments(o, gt, 4; seeds = [0, 1], kw...)
        @test m.mean[1, 1] == 0
        @test m.cov[1, 1] == 0
        counts = [0 1; 1 1; 1 3; 2 4]
        keep = 2:8
        ll = loglikelihood(IncidenceCounts(counts; seeds = [0, 1]), o, gt; kw...)
        @test ll ≈ logpdf(MvNormal(vec(m.mean)[keep], Matrix(m.cov)[keep, keep]), vec(counts)[keep])
        @test loglikelihood(IncidenceCounts([1 1; 1 1; 1 3; 2 4]; seeds = [0, 1]), o, gt; kw...) ==
            -Inf
    end

    @testset "Discretised sum of two delays" begin
        inc, rd = LogNormal(1.6, 0.5), Gamma(2.0, 1.5)
        d = EpiBranch._discretize_report_delay(EpiBranch._DelaySum(inc, rd))
        @test sum(d) ≈ 1.0
        rng = StableRNG(1)
        xs = rand(rng, inc, 200_000) .+ rand(rng, rd, 200_000)
        emp = [mean(x -> (l == 0 ? x <= 0 : l - 1 < x <= l), xs) for l in 0:(length(d) - 1)]
        @test maximum(abs.(emp .- d)) < 3.0e-3
        # A fixed second delay shifts the first.
        @test EpiBranch._discretize_report_delay(EpiBranch._DelaySum(inc, 2.0)) ≈
            EpiBranch._bin_delay(t -> cdf(inc, t - 2.0), 1.0e-8, 10_000)
    end

    @testset "loglikelihood on a model with a Reporting step" begin
        data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18]; seeds = 2)
        offspring, gt = NegBin(1.3, 0.5), LogNormal(1.6, 0.5)
        inc, rd = LogNormal(1.6, 0.5), Gamma(2.0, 1.5)
        process = BranchingProcess(offspring, gt)

        from_onset = ModelSpec(
            process;
            attributes = clinical_presentation(incubation_period = inc, prob_asymptomatic = 0.2),
            progression = [Reporting(delay = rd, probability = 0.7)]
        )
        @test loglikelihood(data, from_onset) ≈ loglikelihood(
            data, offspring, gt;
            reporting_probability = 0.7 * 0.8, reporting_delay = EpiBranch._DelaySum(inc, rd)
        )
        # Other attribute builders that record no onset are allowed alongside.
        with_demographics = ModelSpec(
            process;
            attributes = [clinical_presentation(incubation_period = inc), demographics()],
            progression = [Reporting(delay = rd, probability = 0.7)]
        )
        @test loglikelihood(data, with_demographics) ≈ loglikelihood(
            data, offspring, gt;
            reporting_probability = 0.7, reporting_delay = EpiBranch._DelaySum(inc, rd)
        )

        from_infection = ModelSpec(
            process; progression = [Reporting(delay = rd, probability = 0.7, from = :infection)]
        )
        @test loglikelihood(data, from_infection) ≈ loglikelihood(
            data, offspring, gt; reporting_probability = 0.7, reporting_delay = rd
        )
        @test_throws ArgumentError loglikelihood(
            data, from_infection; reporting_probability = 0.5
        )

        multi = ModelSpec(
            BranchingProcess([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5), gt);
            progression = [Reporting(delay = rd, probability = 0.7, from = :infection)]
        )
        by_type = IncidenceCounts([0 1; 1 1; 1 3; 2 4]; seeds = [0, 1])
        @test loglikelihood(by_type, multi) ≈ loglikelihood(
            by_type, EpiBranch._analytic_offspring(multi), gt;
            reporting_probability = 0.7, reporting_delay = rd
        )

        reporting = Reporting(delay = rd, probability = 0.7)
        cp = clinical_presentation(incubation_period = inc)
        refused = [
            # another step can end a case before its report
            ModelSpec(
                process; attributes = cp,
                progression = [reporting, Recovery(delay = Exponential(5.0))]
            ),
            # nothing records the onset the step reports from
            ModelSpec(process; progression = [reporting]),
            # a per-case delay or asymptomatic share
            ModelSpec(
                process; attributes = cp,
                progression = [Reporting(delay = (rng, ind) -> 1.0, probability = 0.7)]
            ),
            ModelSpec(
                process;
                attributes = clinical_presentation(
                    incubation_period = inc, prob_asymptomatic = (rng, ind) -> 0.2
                ),
                progression = [reporting]
            ),
            # a builder of its own that records an onset
            ModelSpec(
                process; attributes = [cp, (rng, ind) -> (ind.state[:onset_time] = 1.0)],
                progression = [reporting]
            ),
        ]
        for model in refused
            @test_throws ArgumentError loglikelihood(data, model)
        end
    end

    @testset "A Reporting step's moments match simulated reports" begin
        # A generation time of exactly one step puts every infection on a step
        # boundary, where binning the reporting delay is exact.
        n = 5
        inc, rd = Gamma(2.0, 0.8), Exponential(1.0)
        model = ModelSpec(
            BranchingProcess(Poisson(1.2), Dirac(1.0));
            attributes = clinical_presentation(incubation_period = inc, prob_asymptomatic = 0.25),
            progression = [Reporting(delay = rd, probability = 0.7)]
        )
        m = incidence_moments(
            Poisson(1.2), Dirac(1.0), n;
            reporting_probability = 0.7 * 0.75, reporting_delay = EpiBranch._DelaySum(inc, rd)
        )
        nsim = 4000
        sims = zeros(nsim, n)
        for i in 1:nsim
            state = simulate(model; rng = StableRNG(i), max_time = n + 0.5)
            for ind in state.individuals
                is_infected(ind) && get(ind.state, :reported, false) || continue
                t = ceil(Int, ind.state[:reporting_time])
                1 <= t <= n && (sims[i, t] += 1)
            end
        end
        emp_mean = vec(sum(sims; dims = 1)) ./ nsim
        emp_cov = [
            sum((sims[:, a] .- emp_mean[a]) .* (sims[:, b] .- emp_mean[b])) / (nsim - 1)
                for a in 1:n, b in 1:n
        ]
        @test maximum(abs.(emp_mean .- m.mean)) < 0.06
        @test maximum(abs.(emp_cov .- m.cov)) < 0.2
    end
end
