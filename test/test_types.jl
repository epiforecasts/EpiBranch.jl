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
        @test ind.state isa AbstractDict{Symbol, Any}
    end

    @testset "Individual state store" begin
        s = EpiBranch.IndividualState{Float64}()
        s[:onset_time] = 2.5
        s[:age] = 30
        s[:traced] = true
        s[:label] = "a"
        # Each value comes back as the type it was stored as.
        @test s[:onset_time] === 2.5
        @test s[:age] === 30
        @test s[:traced] === true
        @test s[:label] == "a"
        @test length(s) == 4
        @test collect(keys(s)) == [:onset_time, :age, :traced, :label]
        # Overwriting moves a key between a time and any other value.
        s[:onset_time] = missing
        @test s[:onset_time] === missing
        s[:age] = 31.0
        @test s[:age] === 31.0
        @test get(s, :absent, :none) === :none
        @test get(() -> 0, s, :absent) == 0
        @test get!(s, :count, 1) == 1 && s[:count] == 1
        @test get!(() -> 2, s, :count) == 1
        @test_throws KeyError s[:absent]
        @test pop!(s, :count) == 1 && !haskey(s, :count)
        @test pop!(s, :absent, nothing) === nothing
        delete!(s, :label)
        @test !haskey(s, :label) && length(s) == 3
        # A copy is independent of the original.
        c = copy(s)
        c[:age] = 0.0
        @test s[:age] === 31.0
        @test isequal(Dict(s), Dict{Symbol, Any}(:onset_time => missing, :age => 31.0, :traced => true))
        empty!(s)
        @test isempty(s)
        # A value of another number type keeps its own type.
        d = EpiBranch.IndividualState{Float32}()
        d[:t] = 1.0
        @test d[:t] === 1.0
        d[:t] = 1.0f0
        @test d[:t] === 1.0f0
        @test EpiBranch._state_real(d, :t, Inf) === 1.0f0
        @test EpiBranch._state_real(d, :absent, Inf) === Inf32
        # A time is boxed at most once for reads through the dictionary
        # interface, however often it is read.
        firstread(s) = @allocated get(s, :t, 0.0)::Float64
        reread(s) = @allocated get(s, :t, 0.0)::Float64
        r = EpiBranch.IndividualState{Float64}()
        r[:t] = 1.5
        firstread(EpiBranch.IndividualState{Float64}(:t => 0.5))
        @test firstread(r) <= sizeof(Float64) + 8
        @test reread(r) == 0
        # The engine's own read takes the unboxed field even once a read
        # through the dictionary has boxed a copy beside it. The two are made
        # to differ here only so the test can tell which one is read.
        twofold = EpiBranch.IndividualState{Float64}([EpiBranch._Slot{Float64}(:t, 1.0, 2.0)])
        @test EpiBranch._state_real(twofold, :t, Inf) === 2.0
        @test r[:t] === 1.5 && EpiBranch._state_real(r, :t, Inf) === 1.5
        r[:t] = 2.5
        @test r[:t] === 2.5
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
