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
        @test ind.susceptibility == 1.0
        @test ind.infectiousness == 1.0
        @test isempty(ind.secondary_case_ids)
        @test ind.state isa Dict{Symbol, Any}
    end

    @testset "Individual with state" begin
        s = Dict{Symbol, Any}(:age => 30, :isolated => true)
        ind = Individual(; id=1, state=s)
        @test ind.state[:age] == 30
        @test is_isolated(ind)
    end

    @testset "State accessors with defaults" begin
        ind = Individual(; id=1)
        @test !is_isolated(ind)
        @test isolation_time(ind) == Inf
        @test isnan(onset_time(ind))
        @test !is_traced(ind)
        @test !is_asymptomatic(ind)
        @test !is_test_positive(ind)
    end

    @testset "BranchingProcess construction" begin
        model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        @test model.offspring isa NegativeBinomial
        @test model.generation_time isa Distribution
        @test model.population_size === nothing
    end

    @testset "BranchingProcess with population_size" begin
        model = BranchingProcess(Poisson(2.0), Exponential(5.0); population_size=1000)
        @test model.population_size == 1000
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
        @test scaled.r ≈ 0.5

        d_pois = Poisson(3.0)
        scaled_pois = scale_distribution(d_pois, 0.0)
        @test mean(scaled_pois) ≈ 0.0 atol=1e-10
    end
end
