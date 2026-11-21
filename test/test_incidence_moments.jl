using ForwardDiff

@testset "Incidence moments" begin
    @testset "Discretised generation interval" begin
        w = EpiBranch._discretize_generation_interval(LogNormal(1.6, 0.5))
        @test sum(w) ≈ 1.0
        @test all(>=(0), w)

        # A fixed generation time of one step puts the infectee in the next
        # step with certainty: an infector at U within its step infects at
        # U + 1, always in the following step.
        w2 = EpiBranch._discretize_generation_interval(Dirac(1.0))
        @test sum(w2) ≈ 1.0
        @test w2[1] ≈ 0.0 atol = 1.0e-6
        @test w2[2] ≈ 1.0 atol = 1.0e-6

        # The realised lag between the steps of an infector and an infectee
        # has the mean of the continuous distribution, same-step mass included.
        for gt in (LogNormal(1.6, 0.5), Gamma(4.0, 0.6), Gamma(2.0, 0.1))
            w = EpiBranch._discretize_generation_interval(gt)
            @test sum((0:(length(w) - 1)) .* w) ≈ mean(gt) atol = 0.05
        end

        # A generation time much shorter than a step is mostly the same step.
        @test EpiBranch._discretize_generation_interval(Gamma(2.0, 0.05))[1] > 0.9
        d = EpiBranch._discretize_report_delay(Gamma(4.0, 0.6))
        @test sum((0:(length(d) - 1)) .* d) ≈ 2.4 atol = 0.03
    end

    @testset "Discretisation matches the uniform-within-step lag" begin
        # Reference: P(lag ≤ l) = ∫_l^{l+1} F(x) dx by a fine midpoint rule.
        function ref_pmf(delay, nlag)
            H(l) = sum(cdf(delay, l + (i - 0.5) / 4000) for i in 1:4000) / 4000
            return [H(l) - (l == 0 ? 0.0 : H(l - 1)) for l in 0:(nlag - 1)]
        end
        for gt in (LogNormal(1.6, 0.5), Gamma(4.0, 0.6), Weibull(2.0, 3.0), Exponential(2.0))
            w = EpiBranch._discretize_generation_interval(gt)
            r = ref_pmf(gt, 12)
            @test w[1:12] ≈ r ./ sum(ref_pmf(gt, 400)) atol = 1.0e-4
            d = EpiBranch._discretize_report_delay(gt)
            @test d[1:12] ≈ w[1:12] atol = 1.0e-12
        end

        # A fixed delay of c steps lands in step floor(c) or the next, with
        # the fractional part of c the chance of the latter.
        d = EpiBranch._discretize_report_delay(1.25)
        @test d ≈ [0.0, 0.75, 0.25]
        @test EpiBranch._discretize_generation_interval(Dirac(2.5)) ≈ [0.0, 0.0, 0.5, 0.5]

        # Continuous time: infect uniformly within a step, add a continuous
        # generation time, and read off the step it lands in.
        gt = LogNormal(1.0, 0.4)
        rng = StableRNG(1)
        lags = [floor(Int, rand(rng) + rand(rng, gt)) for _ in 1:400_000]
        w = EpiBranch._discretize_generation_interval(gt)
        emp = [count(==(j), lags) / length(lags) for j in 0:5]
        @test emp ≈ w[1:6] atol = 0.01
    end

    @testset "Dual parameters pass through the discretisation" begin
        f(p) = (w = EpiBranch._discretize_generation_interval(Gamma(p[1], p[2])); sum((1:length(w)) .* w))
        g = ForwardDiff.gradient(f, [4.0, 0.6])
        h = 1.0e-5
        fd = (f([4.0 + h, 0.6]) - f([4.0 - h, 0.6])) / 2h
        @test g[1] ≈ fd rtol = 1.0e-4
        @test all(isfinite, g)
        r(p) = sum(EpiBranch._discretize_report_delay(LogNormal(p[1], p[2]))[1:3])
        @test all(isfinite, ForwardDiff.gradient(r, [1.0, 0.4]))
    end

    @testset "incidence_moments matches the renewal equation for the mean" begin
        R, gt, n, seeds = 1.4, LogNormal(1.0, 0.4), 8, 4
        w = EpiBranch._discretize_generation_interval(gt)
        μ_renewal = zeros(Float64, n)
        # The seeds' own same-step offspring are infected at step 0 too.
        get_μ(t) = t == 0 ? seeds / (1 - R * w[1]) : (t < 0 ? 0.0 : μ_renewal[t])
        for t in 1:n
            # The same-step term puts μ_t on both sides.
            μ_renewal[t] = R * sum(w[j + 1] * get_μ(t - j) for j in 1:(length(w) - 1)) /
                (1 - R * w[1])
        end

        m = incidence_moments(Poisson(R), gt, n; seeds = seeds)
        @test m.mean ≈ μ_renewal
    end

    @testset "Lags beyond the horizon are marginalised out" begin
        # Both pmfs are longer than the series. The truncated sums use the
        # tail mass in place of the unobserved steps.
        R, ρ, n, seeds = 1.4, 0.6, 5, 3
        gt, delay = Gamma(3.0, 3.0), LogNormal(1.5, 0.5)
        w = EpiBranch._discretize_generation_interval(gt)
        d = EpiBranch._discretize_report_delay(delay)
        @test length(w) > n && length(d) > n
        infections = zeros(Float64, n)
        get_μ(t) = t == 0 ? seeds / (1 - R * w[1]) : (t < 0 ? 0.0 : infections[t])
        for t in 1:n
            infections[t] = R * sum(w[j + 1] * get_μ(t - j) for j in 1:(length(w) - 1)) /
                (1 - R * w[1])
        end
        reported = [
            ρ * sum(d[l + 1] * get_μ(t - l) for l in 0:(length(d) - 1)) for t in 1:n
        ]
        m = incidence_moments(
            Poisson(R), gt, n; seeds, reporting_probability = ρ, reporting_delay = delay
        )
        @test m.mean ≈ reported
    end

    @testset "incidence_moments matches a discrete-time branching simulation" begin
        # A hand-rolled discrete-time simulation sharing exactly the
        # assumptions `incidence_moments` encodes (each step-k individual
        # draws Poisson offspring, each landing at step k + lag with
        # lag ~ Categorical(w)): this isolates the recursion's correctness
        # from the separate approximation of binning a continuous
        # generation time into calendar steps.
        R, gt, n, seeds = 1.4, LogNormal(1.0, 0.4), 6, 3
        w = EpiBranch._discretize_generation_interval(gt)
        m = incidence_moments(Poisson(R), gt, n; seeds = seeds)

        function simulate_discrete(rng, R, w, n, seeds)
            cohort = Dict(0 => seeds)
            for k in 0:n
                haskey(cohort, k) || continue
                i = 0
                while (i += 1) <= cohort[k]
                    for _ in 1:rand(rng, Poisson(R))
                        day = k + rand(rng, Categorical(w)) - 1
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

    @testset "R is recovered when the generation time is shorter than the step" begin
        # Continuous-time branching with a mean generation time of half a
        # step, counted in unit steps. The seeds are infected uniformly in
        # step 0, which is not observed.
        R_true, gt, n, seeds, nrep = 1.1, Gamma(2.0, 0.25), 6, 5, 150
        function simulate_outbreak(rng)
            counts = zeros(Int, n)
            queue = collect(-rand(rng, seeds))
            while !isempty(queue)
                t = pop!(queue)
                step = t < 0 ? 0 : floor(Int, t) + 1
                step > n && continue
                step >= 1 && (counts[step] += 1)
                for _ in 1:rand(rng, Poisson(R_true))
                    push!(queue, t + rand(rng, gt))
                end
            end
            return counts
        end
        rng = StableRNG(2024)
        series = [simulate_outbreak(rng) for _ in 1:nrep]
        grid = 0.9:0.02:1.3
        function ll(R)
            m = incidence_moments(Poisson(R), gt, n; seeds)
            dist = MvNormal(m.mean, Matrix(m.cov))
            return sum(logpdf(dist, y) for y in series)
        end
        R_hat = grid[argmax(ll.(grid))]
        @test abs(R_hat - R_true) <= 0.04

        # A substantial share of generations falls within a step here.
        @test EpiBranch._discretize_generation_interval(gt)[1] > 0.2
    end

    @testset "Same-step offspring that cannot converge are rejected" begin
        # The generation time is almost entirely within a step, so a
        # reproduction number above one makes the count within a step unbounded.
        @test_throws ArgumentError incidence_moments(Poisson(3.0), Gamma(2.0, 0.01), 4)
    end

    @testset "Discretised reporting delay" begin
        @test EpiBranch._discretize_report_delay(nothing) == [1.0]
        d = EpiBranch._discretize_report_delay(Gamma(2.0, 1.5))
        @test sum(d) ≈ 1.0
        # Infection uniform within its step puts some reports in the same step.
        @test d[1] > 0
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
        w0 = EpiBranch._discretize_generation_interval(gt)[1]
        infected(t) = t == 0 ? seeds / (1 - R * w0) : (t < 0 ? 0.0 : inf.mean[t])
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
                i = 0
                while (i += 1) <= cohort[k]
                    if rand(rng) < ρ
                        day = k + rand(rng, Categorical(d)) - 1
                        1 <= day <= n && (reports[day] += 1)
                    end
                    for _ in 1:rand(rng, Poisson(R))
                        day = k + rand(rng, Categorical(w)) - 1
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

    @testset "ForwardDiff through the generation-time distribution" begin
        f(μ) = sum(incidence_moments(Poisson(1.2), LogNormal(μ, 0.5), 4).mean)
        fd = (f(1.5 + 1.0e-6) - f(1.5 - 1.0e-6)) / 2.0e-6
        @test ForwardDiff.derivative(f, 1.5) ≈ fd rtol = 1.0e-4
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
        past(t) = t == 0 ? inv(one(M) - w[1] * M) * Float64.(seeds) : (t < 0 ? zeros(2) : μ[t, :])
        for t in 1:n
            # The same-step term puts μ_t on both sides.
            μ[t, :] = (one(M) - w[1] * M) \
                (M * sum(w[l + 1] * past(t - l) for l in 1:(length(w) - 1)))
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
            for k in 0:n
                done = zeros(Int, 2)
                while any(j -> done[j] < cohort[k + 1, j], 1:2)
                    j = findfirst(j -> done[j] < cohort[k + 1, j], 1:2)
                    done[j] += 1
                    k >= 1 && rand(rng) < ρ && (reports[k, j] += 1)
                    for _ in 1:rand(rng, Poisson(o.R_by_type[j]))
                        i = rand(rng, Categorical(o.alloc_probs[:, j]))
                        day = k + rand(rng, Categorical(w)) - 1
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

    @testset "Sub-steps approach the continuous-time process" begin
        # Continuous-time branching process with a generation time short
        # compared with a step, seeds infected uniformly within step 0 and
        # counts by (delayed) report time: the reference `substeps` converges to.
        R, gt, n, seeds, ρ = 0.6, Gamma(2.0, 0.1), 3, 2, 0.6
        delay = Gamma(2.0, 0.15)
        function simulate_continuous(rng)
            counts = zeros(n)
            report!(t) =
            if rand(rng) < ρ
                tr = t + rand(rng, delay)
                0 < tr <= n && (counts[ceil(Int, tr)] += 1)
            end
            stack = [rand(rng) - 1 for _ in 1:seeds]
            foreach(report!, stack)
            while !isempty(stack)
                t = pop!(stack)
                for _ in 1:rand(rng, Poisson(R))
                    tc = t + rand(rng, gt)
                    tc <= n || continue
                    report!(tc)
                    push!(stack, tc)
                end
            end
            return counts
        end
        nsim = 20000
        sims = reduce(hcat, (simulate_continuous(StableRNG(i)) for i in 1:nsim))'
        emp_mean = vec(sum(sims, dims = 1)) ./ nsim
        emp_cov = [
            sum((sims[:, t] .- emp_mean[t]) .* (sims[:, t′] .- emp_mean[t′])) / (nsim - 1)
                for t in 1:n, t′ in 1:n
        ]

        kw = (seeds = seeds, reporting_probability = ρ, reporting_delay = delay)
        coarse = incidence_moments(Poisson(R), gt, n; kw...)
        fine = incidence_moments(Poisson(R), gt, n; substeps = 24, kw...)
        @test maximum(abs.(emp_mean .- fine.mean)) < 0.06
        @test maximum(abs.(emp_cov .- fine.cov)) < 0.3
        # The default of one sub-step per step is far from the continuous-time
        # moments here.
        @test maximum(abs.(emp_mean .- coarse.mean)) > 0.25
        @test maximum(abs.(emp_cov .- coarse.cov)) > 1.0
    end

    @testset "Sub-steps in the multi-type method" begin
        gt = Gamma(2.0, 0.1)
        kw = (reporting_probability = 0.8, reporting_delay = Gamma(2.0, 0.2), substeps = 6)
        single = incidence_moments(NegBin(0.9, 0.5), gt, 4; seeds = 2, kw...)
        o = EpiBranch.MultiTypeOffspring(fill(0.9, 1, 1), R -> NegBin(R, 0.5))
        multi = incidence_moments(o, gt, 4; seeds = [2], kw...)
        @test vec(multi.mean) ≈ single.mean
        @test multi.cov ≈ single.cov
        @test_throws ArgumentError incidence_moments(Poisson(1.0), gt, 4; substeps = 0)
        @test_throws ArgumentError incidence_moments(o, gt, 4; seeds = [1], substeps = 0)
    end

    @testset "A count that can only be zero has mean and variance 0" begin
        # With every report delayed by two steps, no type has a report in the
        # first step: mean and variance 0 there.
        o = EpiBranch.MultiTypeOffspring([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5))
        gt, rd = LogNormal(1.6, 0.5), 2.0
        kw = (reporting_probability = 0.7, reporting_delay = rd)
        m = incidence_moments(o, gt, 4; seeds = [0, 1], kw...)
        @test all(m.mean[1, :] .== 0)
        @test m.cov[1, 1] == 0
    end

end

@testset "Missing ForwardDiff extension" begin
    # The core fallback, called directly because the extension is loaded here.
    @test_throws ArgumentError invoke(
        EpiBranch._gradient_and_hessian, Tuple{Any, Any}, identity, [0.0]
    )
end
