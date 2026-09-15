using LinearAlgebra: eigvals

@testset "Multi-type analytics" begin
    M = [1.5 0.6;
         0.5 0.9]
    # Dominant eigenvalue of a 2×2 matrix in closed form.
    trace_M = M[1, 1] + M[2, 2]
    det_M = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
    ρ = (trace_M + sqrt(trace_M^2 - 4det_M)) / 2

    # Fraction of runs seeded by each type that die out before `max_cases`.
    function simulated_extinction(model, n; rng)
        extinct = zeros(2)
        seeded = zeros(2)
        for _ in 1:n
            state = simulate(model; max_cases = 200, rng)
            t = individual_type(state.individuals[1])
            seeded[t] += 1
            extinct[t] += state.extinct
        end
        return extinct ./ seeded
    end

    @testset "R* is the dominant eigenvalue of the offspring matrix" begin
        model = BranchingProcess(M, R -> NegBin(R, 0.5), Exponential(5.0))
        @test reproduction_number(model) ≈ ρ
        @test reproduction_number(ModelSpec(model)) ≈ ρ

        M3 = [2.0 0.5 0.1;
              0.5 1.5 0.3;
              0.1 0.3 0.8]
        model3 = BranchingProcess(M3, R -> Poisson(R), Exponential(5.0))
        @test reproduction_number(model3) ≈ maximum(abs, eigvals(M3))

        # Generic element types take the power-iteration path.
        @test EpiBranch._spectral_radius(big.(M3)) ≈ maximum(abs, eigvals(M3))
        @test EpiBranch._spectral_radius(big.([2.0 1.0; 0.0 0.5])) ≈ 2.0
    end

    @testset "One type reduces to the single-type results" begin
        for d in (NegBin(2.0, 0.3), NegBin(0.8, 0.5), Poisson(1.7), Poisson(0.6))
            model = BranchingProcess(fill(mean(d), 1, 1), R -> d, Exponential(5.0))
            @test reproduction_number(model) ≈ reproduction_number(d)
            q = extinction_probability(model)
            @test q isa Vector
            @test only(q) ≈ extinction_probability(d) atol = 1e-8
            @test only(epidemic_probability(model)) ≈ epidemic_probability(d) atol = 1e-8
        end
        @test reproduction_number(BranchingProcess(NegBin(2.5, 0.16))) ≈ 2.5
        # Subcritical: certain extinction for every type.
        sub = BranchingProcess([0.5 0.2; 0.2 0.5], R -> NegBin(R, 0.5), Exponential(5.0))
        @test extinction_probability(sub) == [1.0, 1.0]
        # Geometric has no `_pgf` method of its own, so it uses the truncated
        # series; it is NegBin with k = 1.
        geom = BranchingProcess([2.0;;], R -> Geometric(1 / (1 + R)), Exponential(5.0))
        @test only(extinction_probability(geom)) ≈ extinction_probability(2.0, 1.0) atol = 1e-8
    end

    @testset "Sink type" begin
        sink = BranchingProcess([2.0 0.0; 1.0 0.0], R -> NegBin(R, 0.5), Exponential(5.0))
        @test reproduction_number(sink) ≈ 2.0
        q = extinction_probability(sink)
        # A type-2 case infects no one; a type-1 case splits its offspring 2:1.
        @test q[2] == 1.0
        @test q[1] ≈ EpiBranch._pgf(NegBin(3.0, 0.5), (2q[1] + q[2]) / 3)
    end

    @testset "Poisson totals: analytic matches simulation" begin
        model = BranchingProcess(M, R -> Poisson(R), Exponential(1.0))
        q = extinction_probability(model)
        @test all(0 .< q .< 1)
        @test simulated_extinction(model, 3000; rng = StableRNG(1)) ≈ q atol = 0.05
    end

    @testset "Negative binomial totals: analytic matches simulation" begin
        k = 0.5
        model = BranchingProcess(M, R -> NegBin(R, k), Exponential(1.0))
        q = extinction_probability(model)
        # Treating each type's offspring count as an independent negative
        # binomial gives a different answer here, so agreement with simulation
        # checks the multinomial split.
        q_indep = zeros(2)
        for _ in 1:2000
            q_indep = [prod(EpiBranch._pgf(NegBin(M[i, j], k), q_indep[i]) for i in 1:2)
                       for j in 1:2]
        end
        @test all(abs.(q .- q_indep) .> 0.08)
        @test simulated_extinction(model, 3000; rng = StableRNG(2)) ≈ q atol = 0.05
    end

    @testset "Single-type-only helpers throw for a multi-type model" begin
        model = BranchingProcess(M, R -> Poisson(R), Exponential(5.0))
        @test_throws ArgumentError single_type_offspring(model)
        @test_throws ArgumentError probability_contain(model)
    end
end
