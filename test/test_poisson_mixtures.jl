using Distributions
using EpiBranch: quadgk
using StableRNGs

@testset "Poisson mixtures" begin
    lnorm = PoissonLognormal(-0.5, 1.0)
    weib = PoissonWeibull(0.7, 1.0)

    @testset "Mass function matches direct integration over the rate" begin
        for (d, rate) in ((lnorm, LogNormal(-0.5, 1.0)), (weib, Weibull(0.7, 1.0)))
            for x in (0, 1, 4, 20)
                ref, _ = quadgk(λ -> pdf(Poisson(λ), x) * pdf(rate, λ), 0, Inf; rtol = 1.0e-10)
                @test pdf(d, x) ≈ ref rtol = 1.0e-6
                @test logpdf(d, x) ≈ log(ref) rtol = 1.0e-6
            end
            @test pdf(d, -1) == 0.0
            @test sum(pdf(d, x) for x in 0:5000) ≈ 1.0 atol = 1.0e-3
            @test cdf(d, 7) ≈ sum(pdf(d, x) for x in 0:7) rtol = 1.0e-6
            @test ccdf(d, 7) ≈ 1 - cdf(d, 7) atol = 1.0e-8
        end
    end

    @testset "Mass function matches superspreading" begin
        # `dpoislnorm(x, meanlog, sdlog)` and `dpoisweibull(x, shape, scale)`
        # from superspreading 0.4.0. Its integration is accurate to about 1e-8
        # at these counts, except for the (-0.5, 1.0) lognormal at count 20,
        # where it returns 5.86e-5 and direct integration gives 6.404e-5; that
        # count is left out.
        xs = [0, 1, 5, 20]
        lnorm_ref = Dict(
            (-0.5, 1.0) => [
                5.12428646929183e-01, 2.58714852092794e-01,
                1.38899840898597e-02,
            ],
            (0.5, 0.8) => [
                2.40886906634488e-01, 2.48174179186483e-01,
                4.67428348258177e-02, 2.94270715342657e-04,
            ],
            (1.0, 1.5) => [
                2.11514514509127e-01, 1.59315164253183e-01,
                4.81145363376694e-02, 5.65462793828036e-03,
            ],
        )
        weibull_ref = Dict(
            (0.7, 1.0) => [
                0.523964215242873, 0.203107922890086,
                0.0231208801820102, 0.000147858917472714,
            ],
            (1.5, 2.0) => [
                2.76214636099870e-01, 2.56857542279831e-01,
                4.11242941834865e-02, 3.17729715601352e-07,
            ],
            (0.5, 3.0) => [
                0.379886912922477, 0.138267368129019,
                0.0348765095198392, 0.00497276338355947,
            ],
        )
        for (p, ref) in lnorm_ref
            @test [pdf(PoissonLognormal(p...), x) for x in xs[1:length(ref)]] ≈ ref rtol = 1.0e-6
        end
        for (p, ref) in weibull_ref
            @test [pdf(PoissonWeibull(p...), x) for x in xs[1:length(ref)]] ≈ ref rtol = 1.0e-6
        end
    end

    @testset "Large counts match direct integration over the log rate" begin
        # superspreading underflows to zero or loses accuracy beyond a count
        # of about 20, so these are checked against a fine grid in the log of
        # the rate instead.
        grid = range(-15, 15; length = 1001)
        function direct(rate, x)
            f = t -> (λ = exp(t); exp(logpdf(Poisson(λ), x) + logpdf(rate, λ) + t))
            return first(quadgk(f, grid...; rtol = 1.0e-12))
        end
        cases = (
            (PoissonLognormal(-0.5, 1.0), LogNormal(-0.5, 1.0)),
            (PoissonLognormal(1.0, 1.5), LogNormal(1.0, 1.5)),
            (PoissonWeibull(0.7, 1.0), Weibull(0.7, 1.0)),
            (PoissonWeibull(0.5, 3.0), Weibull(0.5, 3.0)),
        )
        for (d, rate) in cases, x in (100, 500)
            @test pdf(d, x) ≈ direct(rate, x) rtol = 1.0e-6
        end
        d = PoissonLognormal(1.0, 1.5)
        @test cdf(d, 100) + ccdf(d, 100) ≈ 1.0 atol = 1.0e-8
        @test cdf(d, 500) ≈ sum(pdf(d, x) for x in 0:500) rtol = 1.0e-6
    end

    @testset "Moments and sampling" begin
        @test mean(lnorm) ≈ exp(-0.5 + 0.5)
        @test mean(weib) ≈ mean(Weibull(0.7, 1.0))
        rng = StableRNG(1)
        for d in (lnorm, weib)
            draws = rand(rng, d, 100_000)
            @test mean(draws) ≈ mean(d) rtol = 0.05
            @test var(draws) ≈ var(d) rtol = 0.15
        end
    end

    @testset "Support, parameters and variance" begin
        for d in (lnorm, weib)
            @test minimum(d) == 0
            @test maximum(d) == Inf
            @test insupport(d, 3)
            @test !insupport(d, -1)
            @test logpdf(d, -1) == -Inf
            @test ccdf(d, -1) == 1.0
            @test cdf(d, -1) == 0.0
            @test partype(d) == Float64
        end
        @test params(lnorm) == (-0.5, 1.0)
        @test params(weib) == (0.7, 1.0)
        @test var(lnorm) ≈ mean(lnorm) + var(LogNormal(-0.5, 1.0))
        @test var(weib) ≈ mean(weib) + var(Weibull(0.7, 1.0))
    end

    @testset "Generating function and extinction" begin
        for d in (lnorm, weib)
            @test EpiBranch._pgf_derivative(d, 1.0) ≈ mean(d) rtol = 1.0e-6
            @test EpiBranch._pgf_derivative(d, 0.6) ≈
                sum(x * pdf(d, x) * 0.6^(x - 1) for x in 1:200) rtol = 1.0e-6
        end
        for d in (lnorm, weib)
            @test EpiBranch._pgf(d, 1.0) ≈ 1.0 atol = 1.0e-8
            @test EpiBranch._pgf(d, 0.0) ≈ pdf(d, 0) rtol = 1.0e-6
            @test EpiBranch._pgf(d, 0.6) ≈ sum(pdf(d, x) * 0.6^x for x in 0:200) rtol = 1.0e-6
        end
        sub = PoissonLognormal(-1.0, 0.5)
        @test reproduction_number(sub) < 1
        @test extinction_probability(sub) ≈ 1.0
        sup = PoissonLognormal(0.5, 1.0)
        q = extinction_probability(sup)
        @test 0 < q < 1
        @test EpiBranch._pgf(sup, q) ≈ q atol = 1.0e-6
        q = extinction_probability(PoissonWeibull(0.7, 2.0))
        @test 0 < q < 1
    end

    @testset "Chain sizes" begin
        d = chain_size_distribution(PoissonLognormal(-1.0, 0.5))
        # Size 1 occurs when the index case has no secondary cases.
        @test pdf(d, 1) ≈ pdf(PoissonLognormal(-1.0, 0.5), 0) rtol = 1.0e-6
        ll = loglikelihood(ChainSizes([1, 2, 1, 3]), PoissonWeibull(0.7, 0.4))
        @test isfinite(ll)
    end

    @testset "Argument checks" begin
        @test_throws ArgumentError PoissonLognormal(0.0, 0.0)
        @test_throws ArgumentError PoissonWeibull(-1.0, 1.0)
        @test_throws ArgumentError PoissonWeibull(1.0, 0.0)
    end
end
