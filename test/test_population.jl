using Distributions
using StableRNGs

@testset "Population" begin
    @testset "construction" begin
        @test Population(5000) isa EpiBranch.SinglePopulation
        @test Population([10, 20]) isa EpiBranch.GroupedPopulation
        @test Population([10, 20]).sizes == [10, 20]
        @test Population([10, 20]; groups = ["a", "b"]).groups == ["a", "b"]
        @test EpiBranch.total_size(Population([10, 20])) == 30
        @test EpiBranch.total_size(NoPopulation()) isa NoPopulation

        @test_throws ArgumentError Population(0)
        @test_throws ArgumentError Population(Int[])
        @test_throws ArgumentError Population([10, -1])
        @test_throws ArgumentError Population([0, 0])
        @test_throws ArgumentError Population([10, 20]; groups = ["a"])
        @test_throws ArgumentError Population([10, 20]; groups = ["a", "a"])
    end

    @testset "population_size is shorthand for one group" begin
        a = BranchingProcess(Poisson(2.0), Exponential(5.0); population_size = 50)
        b = BranchingProcess(Poisson(2.0), Exponential(5.0); population = Population(50))
        @test EpiBranch.population(a) == EpiBranch.population(b)
        @test EpiBranch.population_size(a) == 50
        sa = simulate(a, 3; rng = StableRNG(1))
        sb = simulate(b, 3; rng = StableRNG(1))
        @test length(sa) == length(sb)
        @test_throws ArgumentError BranchingProcess(
            Poisson(2.0), Exponential(5.0); population_size = 50, population = Population(50)
        )
    end

    @testset "the number of groups must match the number of types" begin
        M = [1.2 0.4; 0.3 0.9]
        @test_throws ArgumentError BranchingProcess(
            M, R -> Poisson(R), Exponential(5.0); population = Population([10, 20, 30])
        )
        @test_throws ArgumentError BranchingProcess(
            Poisson(1.0), Exponential(5.0); population = Population([10, 20])
        )
        labelled = BranchingProcess(
            M, R -> Poisson(R), Exponential(5.0);
            population = Population([10, 20]; groups = ["child", "adult"])
        )
        @test labelled.type_labels == ["child", "adult"]
        @test_throws ArgumentError BranchingProcess(
            M, R -> Poisson(R), Exponential(5.0);
            population = Population([10, 20]; groups = ["child", "adult"]),
            type_labels = ["a", "b"]
        )
    end
end

@testset "Per-type depletion in simulate" begin
    # Every case infects two of each type, so a type is depleted only by its
    # own cases: a small type saturates while a large one carries on.
    M = [2.0 2.0; 2.0 2.0]
    model = BranchingProcess(
        M, R -> Poisson(R), Exponential(2.0); population = Population([5, 2000])
    )
    for seed in 1:5
        state = simulate(model; n_initial = 1, max_cases = 1500, rng = StableRNG(seed))
        by_type = [
            count(i -> EpiBranch.individual_type(i) == t && i.state[:infected], state.individuals)
                for t in 1:2
        ]
        @test state.group_cases == by_type
        @test by_type[1] <= 5
        @test by_type[2] <= 2000
        @test sum(by_type) == state.cumulative_cases
    end

    @testset "a type that is everyone's source and target depletes its own size" begin
        # Type 2 has 40 people: an outbreak cannot exceed them in that type
        # however many cases of type 1 there are.
        big = BranchingProcess(
            [3.0 3.0; 3.0 3.0], R -> Poisson(R), Exponential(2.0);
            population = Population([3000, 40])
        )
        states = [simulate(big; n_initial = 3, max_cases = 2500, rng = StableRNG(s)) for s in 1:5]
        @test all(s -> s.group_cases[2] <= 40, states)
        @test any(s -> s.group_cases[2] == 40, states)
    end

    @testset "a total alone is refused" begin
        total = BranchingProcess(
            M, R -> Poisson(R), Exponential(2.0); population_size = 100
        )
        @test_throws "size of each type" simulate(total; n_initial = 1, rng = StableRNG(1))
    end

    @testset "a continuation keeps counting each type" begin
        spec = ModelSpec(model)
        first_run = simulate(spec; n_initial = 1, max_time = 6.0, max_cases = 10_000, rng = StableRNG(2))
        longer = simulate(spec, first_run; max_time = 30.0)
        @test longer.group_cases == [
            count(i -> EpiBranch.individual_type(i) == t && i.state[:infected], longer.individuals)
                for t in 1:2
        ]
        @test longer.group_cases[1] <= 5
    end
end

@testset "Closed forms refuse a finite multi-type population" begin
    M = [1.2 0.4; 0.3 0.9]
    plain = BranchingProcess(M, R -> NegBin(R, 0.5), Exponential(5.0))
    finite = BranchingProcess(
        M, R -> NegBin(R, 0.5), Exponential(5.0); population = Population([500, 500])
    )
    @test !EpiBranch.offspring_law_unchanged(finite)
    @test EpiBranch.offspring_law_unchanged(plain)
    @test_throws ArgumentError extinction_probability(finite)
    @test 0 < extinction_probability(plain)[1] <= 1
end

@testset "MixingProcess takes sizes by group" begin
    prog = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)]
    M = [3.0 0.5; 0.5 0.5]
    rate = (from, to) -> M[to[1], from[1]] / 100
    sizes = [30, 70]

    model = ModelSpec(
        MixingProcess(; population = Population(sizes), mixing_by = (:band,), rate);
        progression = prog
    )
    @test EpiBranch.population_size(model) == 100
    for seed in 1:3
        state = simulate(model; n_initial = 2, rng = StableRNG(seed))
        bands = [i.state[:band] for i in state.individuals]
        @test count(==(1), bands) == 30
        @test count(==(2), bands) == 70
    end

    @testset "groups name the attribute values" begin
        named = MixingProcess(;
            population = Population(sizes; groups = [("young",), ("old",)]),
            mixing_by = (:band,), rate = (from, to) -> 0.01
        )
        state = simulate(ModelSpec(named; progression = prog); n_initial = 1, rng = StableRNG(1))
        bands = [i.state[:band] for i in state.individuals]
        @test count(==("young"), bands) == 30
        @test count(==("old"), bands) == 70

        # A bare value is a one-element tuple.
        bare = MixingProcess(;
            population = Population(sizes; groups = ["young", "old"]),
            mixing_by = (:band,), rate = (from, to) -> 0.01
        )
        state = simulate(ModelSpec(bare; progression = prog); n_initial = 1, rng = StableRNG(1))
        @test count(==("young"), [i.state[:band] for i in state.individuals]) == 30
    end

    @testset "attributes read the assigned group" begin
        seen = Int[]
        reads = (rng, ind) -> push!(seen, ind.state[:band])
        spec = ModelSpec(
            MixingProcess(; population = Population(sizes), mixing_by = (:band,), rate);
            progression = prog, attributes = reads
        )
        simulate(spec; n_initial = 1, rng = StableRNG(1))
        @test sort(seen) == sort(vcat(fill(1, 30), fill(2, 70)))
    end

    @testset "an attribute that writes the group is an error" begin
        band = (rng, ind) -> (ind.state[:band] = rand(rng, 1:2))
        spec = ModelSpec(
            MixingProcess(; population = Population(sizes), mixing_by = (:band,), rate);
            progression = prog, attributes = band
        )
        @test_throws "assigns from its group sizes" simulate(spec; n_initial = 1, rng = StableRNG(1))
    end

    @testset "without sizes the attributes draw the groups" begin
        band = (rng, ind) -> (ind.state[:band] = rand(rng, 1:2))
        spec = ModelSpec(
            MixingProcess(; population = Population(100), mixing_by = (:band,), rate);
            progression = prog, attributes = band
        )
        state = simulate(spec; n_initial = 1, rng = StableRNG(1))
        @test length(state.individuals) == 100
        @test EpiBranch.population_size(spec) == 100
    end

    @testset "sizes by group need mixing_by of matching length" begin
        @test_throws ArgumentError simulate(
            ModelSpec(
                MixingProcess(; population = Population(sizes), rate); progression = prog
            ); n_initial = 1, rng = StableRNG(1)
        )
        @test_throws ArgumentError simulate(
            ModelSpec(
                MixingProcess(;
                    population = Population(sizes), mixing_by = (:a, :b), rate
                ); progression = prog
            ); n_initial = 1, rng = StableRNG(1)
        )
    end

    @testset "a population is needed" begin
        @test_throws ArgumentError MixingProcess(; rate)
        @test_throws ArgumentError MixingProcess(;
            rate, population_size = 10, population = Population(10)
        )
    end
end

@testset "HomogeneousProcess takes a one-group population" begin
    a = HomogeneousProcess(; transmission_rate = 1.5, population = Population(80))
    b = HomogeneousProcess(; transmission_rate = 1.5, population_size = 80)
    @test EpiBranch.population(a) == EpiBranch.population(b)
    @test_throws ArgumentError HomogeneousProcess(;
        transmission_rate = 1.5, population = Population([40, 40])
    )
    @test_throws ArgumentError HomogeneousProcess(; transmission_rate = 1.5)
end
