@testset "RenewalOffspring" begin
    ind = Individual(id = 1, infection_time = 10.0)
    draws(o, n; seed = 1) = begin
        rng = StableRNG(seed)
        [length(o(rng, ind).times) for _ in 1:n]
    end

    @testset "constant R gives NegBin(R, k) offspring" begin
        g = [0.2, 0.5, 0.3]
        for gt in (g, Gamma(4.0, 1.0))
            o = RenewalOffspring(
                2.0, gt; k = 0.5, R_max = gt isa Distribution ? 2.0 : nothing
            )
            x = draws(o, 40_000)
            @test mean(x) ≈ 2.0 rtol = 0.04
            @test var(x) ≈ var(NegBin(2.0, 0.5)) rtol = 0.08
        end
    end

    @testset "default k gives Poisson offspring" begin
        x = draws(RenewalOffspring(1.5, [1.0]), 40_000)
        @test mean(x) ≈ 1.5 rtol = 0.03
        @test var(x) ≈ 1.5 rtol = 0.05
    end

    @testset "a pmf places infections on whole-day lags at R(t + lag)" begin
        # R is zero except on day 12, which is lag 2 from an infection on day 10.
        o = RenewalOffspring(t -> t == 12.0 ? 5.0 : 0.0, [0.5, 0.5])
        times = reduce(vcat, [o(StableRNG(s), ind).times for s in 1:200])
        @test !isempty(times)
        @test all(==(2.0), times)
    end

    @testset "a continuous generation time thins by R(t)" begin
        # No infection can fall after day 15.
        o = RenewalOffspring(t -> t < 15 ? 1.0 : 0.0, Gamma(4.0, 1.0); R_max = 1.0)
        times = reduce(vcat, [o(StableRNG(s), ind).times for s in 1:2000])
        @test !isempty(times)
        @test all(<(5.0), times)
        @test !all(isinteger, times)
        # Mean count is R times the share of the generation time before day 15.
        @test length(times) / 2000 ≈ cdf(Gamma(4.0, 1.0), 5.0) rtol = 0.05
    end

    @testset "one case's infections on different days are correlated" begin
        o = RenewalOffspring(3.0, [0.5, 0.5]; k = 0.3)
        rng = StableRNG(2)
        a = Int[]
        b = Int[]
        for _ in 1:20_000
            t = o(rng, ind).times
            push!(a, count(==(1.0), t))
            push!(b, count(==(2.0), t))
        end
        @test cor(a, b) > 0.2
        p = RenewalOffspring(3.0, [0.5, 0.5])
        rng = StableRNG(2)
        a = Int[]
        b = Int[]
        for _ in 1:20_000
            t = p(rng, ind).times
            push!(a, count(==(1.0), t))
            push!(b, count(==(2.0), t))
        end
        @test abs(cor(a, b)) < 0.03
    end

    @testset "simulation follows the renewal mean" begin
        g = [0.3, 0.4, 0.3]
        Rfun(t) = t < 6 ? 2.0 : 0.5
        model = BranchingProcess(Infectiousness(RenewalOffspring(Rfun, g; k = 1.0)))
        daily(seed) = begin
            s = simulate(
                model; n_initial = 200, max_cases = nothing, max_time = 12.0,
                rng = StableRNG(seed)
            )
            counts = zeros(13)
            for c in s.individuals
                counts[Int(c.infection_time) + 1] += 1
            end
            counts
        end
        sim = sum(daily(s) for s in 1:10) ./ 10
        expected = zeros(13)
        expected[1] = 200
        for t in 1:12, (lag, w) in enumerate(g)
            t - lag >= 0 && (expected[t + 1] += expected[t - lag + 1] * Rfun(t) * w)
        end
        for t in (3, 6, 9)
            @test sim[t + 1] ≈ expected[t + 1] rtol = 0.12
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError RenewalOffspring(1.0, Gamma(2.0, 1.0))
        @test_throws ArgumentError RenewalOffspring(1.0, Normal(0, 1); R_max = 1.0)
        @test_throws ArgumentError RenewalOffspring(1.0, [1.0]; k = 0.0)
        @test_throws ArgumentError RenewalOffspring(1.0, [-1.0, 2.0])
        o = RenewalOffspring(2.0, Gamma(2.0, 1.0); R_max = 1.0)
        @test_throws ArgumentError o(StableRNG(1), ind)
    end
end
