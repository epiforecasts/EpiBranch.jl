@testset "Maximum-likelihood fitting" begin
    @testset "OffspringCounts — Poisson" begin
        data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 1, 4])
        f = fit(data, Poisson)

        # The Poisson MLE for iid counts is the sample mean, exactly.
        @test f.estimate.R ≈ mean(data.data) atol = 1.0e-6
        @test f.loglikelihood ≈ loglikelihood(data, Poisson(f.estimate.R))
        @test f.level == 0.95

        lo, hi = f.ci.R
        @test lo < f.estimate.R < hi
        # Both sides are finite for a well-identified single-parameter fit.
        @test isfinite(lo) && isfinite(hi)
    end

    @testset "OffspringCounts — NegativeBinomial" begin
        rng = StableRNG(1)
        true_R, true_k = 1.2, 0.4
        data = OffspringCounts(rand(rng, NegBin(true_R, true_k), 400))
        f = fit(data, NegativeBinomial)

        # The NegBin mean MLE is also the sample mean, exactly, regardless of k.
        @test f.estimate.R ≈ mean(data.data) atol = 1.0e-6
        @test f.estimate.k > 0
        @test f.loglikelihood ≈ loglikelihood(data, NegBin(f.estimate.R, f.estimate.k))
        @test isapprox(f.estimate.k, true_k; rtol = 0.5)

        R_lo, R_hi = f.ci.R
        @test R_lo < f.estimate.R < R_hi
        k_lo, k_hi = f.ci.k
        @test k_lo < f.estimate.k < k_hi
    end

    @testset "Unbounded upper CI on k when data show no overdispersion" begin
        # Exactly Poisson-distributed counts: the NegBin likelihood keeps
        # improving as k -> infty (the Poisson limit), so the profile
        # likelihood never drops far enough on the upper side.
        rng = StableRNG(2)
        data = OffspringCounts(rand(rng, Poisson(1.5), 500))
        f = fit(data, NegativeBinomial)

        k_lo, k_hi = f.ci.k
        @test isfinite(k_lo)
        @test k_hi == Inf
    end

    @testset "ChainSizes — Poisson" begin
        rng = StableRNG(3)
        true_R = 0.6
        model = BranchingProcess(Poisson(true_R))
        states = simulate(model, 300; rng = rng)
        sizes = Int[]
        for s in states
            append!(sizes, chain_statistics(s).size)
        end
        data = ChainSizes(sizes)
        f = fit(data, Poisson)

        @test isapprox(f.estimate.R, true_R; atol = 0.1)
        @test f.loglikelihood ≈ loglikelihood(data, Poisson(f.estimate.R))
        lo, hi = f.ci.R
        @test lo < f.estimate.R < hi
    end

    @testset "ChainSizes — NegativeBinomial with multi-seed clusters" begin
        rng = StableRNG(4)
        true_R, true_k = 0.6, 0.3
        cluster_law = chain_size_distribution(NegBin(true_R, true_k))
        seeds = rand(rng, [1, 1, 1, 2], 200)
        sizes = [sum(rand(rng, cluster_law) for _ in 1:s) for s in seeds]
        data = ChainSizes(sizes; seeds = seeds)

        f = fit(data, NegativeBinomial)
        @test isapprox(f.estimate.R, true_R; atol = 0.2)
        @test f.loglikelihood ≈ loglikelihood(data, NegBin(f.estimate.R, f.estimate.k))
    end

    @testset "ChainLengths — Poisson, subcritical search domain" begin
        rng = StableRNG(5)
        true_R = 0.5
        model = BranchingProcess(Poisson(true_R))
        states = simulate(model, 300; rng = rng)
        lengths = Int[]
        for s in states
            append!(lengths, chain_statistics(s).length)
        end
        data = ChainLengths(lengths)

        f = fit(data, Poisson)
        @test 0 < f.estimate.R < 1
        @test isapprox(f.estimate.R, true_R; atol = 0.15)
        @test f.loglikelihood ≈ loglikelihood(data, Poisson(f.estimate.R))
    end

    @testset "Parametric bootstrap" begin
        rng = StableRNG(6)
        true_R = 1.0
        data = OffspringCounts(rand(rng, Poisson(true_R), 300))

        f_no_boot = fit(data, Poisson)
        @test f_no_boot.bootstrap_ci === nothing

        f = fit(data, Poisson; bootstrap = 200, rng = StableRNG(7))
        @test f.bootstrap_ci !== nothing
        lo, hi = f.bootstrap_ci.R
        @test lo < f.estimate.R < hi
    end

    @testset "Parametric bootstrap over both Negative Binomial parameters" begin
        rng = StableRNG(11)
        data = OffspringCounts(rand(rng, NegativeBinomial(0.5, 0.5 / (0.5 + 2.0)), 400))

        f = fit(data, NegativeBinomial; bootstrap = 100, rng = StableRNG(12))
        @test f.bootstrap_ci !== nothing
        for (par, est) in ((:R, f.estimate.R), (:k, f.estimate.k))
            lo, hi = getproperty(f.bootstrap_ci, par)
            @test lo <= est <= hi
            @test isfinite(lo) && isfinite(hi)
        end
    end

    @testset "fit validates level" begin
        data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 2, 1, 4])
        for level in (1.0, 1.5, -0.1, 0.0)
            @test_throws ArgumentError fit(data, Poisson; level)
            @test_throws ArgumentError fit(data, NegativeBinomial; level)
        end
    end

    @testset "A profile side that never crosses is reported as unbounded" begin
        # `_profile_bound` walks outward by a fixed factor and gives up at
        # `hi_bound`. With no bound to reach and a profile that never drops, the
        # expansion runs out instead, which is the other way out of the search.
        flat = _ -> 0.0
        @test EpiBranch._profile_bound(
            flat, 1.0, -1.0, 1; lo_bound = 1.0e-8, hi_bound = Inf
        ) == Inf
        @test EpiBranch._profile_bound(
            flat, 1.0, -1.0, -1; lo_bound = 0.0, hi_bound = Inf
        ) == 0.0
    end

    @testset "MLEFit show method" begin
        data = OffspringCounts([0, 1, 2, 0, 3, 1, 0])
        f = fit(data, Poisson)
        @test occursin("R", sprint(show, f))
    end
end
