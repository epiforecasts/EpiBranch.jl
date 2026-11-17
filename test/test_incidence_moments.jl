using ForwardDiff

@testset "Incidence moments" begin
    @testset "Discretised generation interval" begin
        w = EpiBranch._discretize_generation_interval(LogNormal(1.6, 0.5))
        @test sum(w) ≈ 1.0
        @test all(>=(0), w)

        # A generation time with (almost) all its mass on lag 1 gives a pmf
        # concentrated there.
        w2 = EpiBranch._discretize_generation_interval(Dirac(1.0))
        @test w2[1] ≈ 1.0
        @test all(x -> x ≈ 0, w2[2:end])

        # The realised lag between the steps of an infector and an infectee
        # has the mean of the continuous distribution.
        # (Dropping the small same-step mass lengthens it slightly.)
        for gt in (LogNormal(1.6, 0.5), Gamma(4.0, 0.6))
            w = EpiBranch._discretize_generation_interval(gt)
            @test sum((1:length(w)) .* w) ≈ mean(gt) atol = 0.1
        end
        d = EpiBranch._discretize_report_delay(Gamma(4.0, 0.6))
        @test sum((0:(length(d) - 1)) .* d) ≈ 2.4 atol = 0.03
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
        # generation time into calendar steps.
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

    @testset "A count that can only be zero has mean and variance 0" begin
        # With no type-1 seed and every report delayed by a step, type 1 has
        # no report in the first step: mean and variance 0 there.
        o = EpiBranch.MultiTypeOffspring([1.2 0.4; 0.3 0.9], R -> NegBin(R, 0.5))
        gt, rd = LogNormal(1.6, 0.5), 1.0
        kw = (reporting_probability = 0.7, reporting_delay = rd)
        m = incidence_moments(o, gt, 4; seeds = [0, 1], kw...)
        @test m.mean[1, 1] == 0
        @test m.cov[1, 1] == 0
    end

end

@testset "Missing ForwardDiff extension" begin
    # The core fallback, called directly because the extension is loaded here.
    @test_throws ArgumentError invoke(
        EpiBranch._gradient_and_hessian, Tuple{Any, Any}, identity, [0.0]
    )
end
