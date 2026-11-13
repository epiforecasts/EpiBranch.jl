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

    @testset "Generating function and extinction" begin
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
