@testset "Types" begin
    @testset "NegBin convenience constructor" begin
        d = NegBin(2.5, 0.16)
        @test d isa NegativeBinomial
        @test mean(d) ≈ 2.5 atol=0.01
        @test_throws ArgumentError NegBin(-1.0, 0.5)
        @test_throws ArgumentError NegBin(2.0, -0.1)
    end

    @testset "Individual construction" begin
        ind = Individual(; id=1, infection_time=0.0)
        @test ind.id == 1
        @test ind.parent_id == 0
        @test ind.generation == 0
        @test !ind.isolated
        @test ind.isolation_time == Inf
        @test isempty(ind.secondary_case_ids)
    end

    @testset "BranchingProcess construction" begin
        model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        @test model.offspring isa NegativeBinomial
        @test model.generation_time isa Distribution
    end

    @testset "BranchingProcess with function generation time" begin
        gt_fn = inc -> truncated(Normal(inc, 2.0), 0.0, Inf)
        model = BranchingProcess(NegBin(2.5, 0.16), gt_fn)
        @test model.generation_time isa Function
    end

    @testset "scale_distribution" begin
        d = NegBin(2.0, 0.5)
        scaled = scale_distribution(d, 0.5)
        @test mean(scaled) ≈ 1.0 atol=0.01
        @test scaled.r ≈ 0.5  # dispersion preserved

        d_pois = Poisson(3.0)
        scaled_pois = scale_distribution(d_pois, 0.0)
        @test mean(scaled_pois) ≈ 0.0 atol=1e-10
    end
end
