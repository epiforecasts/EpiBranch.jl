# A model whose `single_type_offspring` raises something other than
# `ArgumentError`/`MethodError`, so `has_single_type_law` must not swallow it.
struct _BrokenOffspringModel <: EpiBranch.TransmissionModel end
EpiBranch.single_type_offspring(::_BrokenOffspringModel) = error("broken")

@testset "Types" begin
    @testset "NegBin convenience constructor" begin
        d = NegBin(2.5, 0.16)
        @test d isa NegativeBinomial
        @test mean(d) ≈ 2.5 atol = 0.01
        @test_throws ArgumentError NegBin(-1.0, 0.5)
        @test_throws ArgumentError NegBin(2.0, -0.1)
    end

    @testset "Individual construction" begin
        ind = Individual(; id = 1, infection_time = 0.0)
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
        ind = Individual(; id = 1, state = s)
        @test ind.state[:age] == 30
        @test is_isolated(ind)
    end

    @testset "State accessors with defaults" begin
        ind = Individual(; id = 1)
        @test !is_isolated(ind)
        @test isolation_time(ind) == Inf
        @test isnan(onset_time(ind))
        @test !is_traced(ind)
        @test !is_asymptomatic(ind)
        @test !is_test_positive(ind)
    end

    @testset "BranchingProcess construction" begin
        model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        @test single_type_offspring(model) isa NegativeBinomial
        @test model.infectiousness[1].kernel isa Distribution
        @test model.population_size isa NoPopulation
    end

    @testset "has_single_type_law" begin
        # A single-type offspring law with a `chain_size_distribution` method.
        model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        @test EpiBranch.has_single_type_law(model)

        # A single-type law with no `chain_size_distribution` method.
        no_law_model = BranchingProcess(Binomial(5, 0.3))
        @test !EpiBranch.has_single_type_law(no_law_model)

        # `single_type_offspring` throwing `ArgumentError` (multi-type offspring).
        multi_type_model = BranchingProcess(
            [0.4 0.2; 0.1 0.3], R -> Poisson(R), Exponential(5.0)
        )
        @test !EpiBranch.has_single_type_law(multi_type_model)

        # Any other exception from `single_type_offspring` is not swallowed.
        @test_throws ErrorException EpiBranch.has_single_type_law(
            _BrokenOffspringModel()
        )
    end

    @testset "BranchingProcess with population_size" begin
        model = BranchingProcess(Poisson(2.0), Exponential(5.0); population_size = 1000)
        @test model.population_size == 1000
    end

    @testset "BranchingProcess with function generation time" begin
        gt_fn = inc -> truncated(Normal(inc, 2.0), 0.0, Inf)
        model = BranchingProcess(NegBin(2.5, 0.16), gt_fn)
        @test model.infectiousness[1].kernel isa Function
    end

    @testset "scale_distribution" begin
        d = NegBin(2.0, 0.5)
        scaled = scale_distribution(d, 0.5)
        @test mean(scaled) ≈ 1.0 atol = 0.01
        @test scaled.r ≈ 0.5

        d_pois = Poisson(3.0)
        scaled_pois = scale_distribution(d_pois, 0.0)
        @test mean(scaled_pois) ≈ 0.0 atol = 1.0e-10

        # An unsupported family gives a clear ArgumentError, not a MethodError.
        @test_throws ArgumentError scale_distribution(Geometric(0.3), 0.5)
    end
end
