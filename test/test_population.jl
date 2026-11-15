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

@testset "Mixing values" begin
    c = [10.0 1.0; 4.0 2.0]
    gt = Exponential(5.0)
    q, D = 0.05, 5.0
    offspring(m) = m.infectiousness[1].offspring.offspring_matrix

    @testset "construction" begin
        @test FrequencyDependent(c) isa Mixing
        @test DensityDependent(c) isa Mixing
        @test_throws ArgumentError FrequencyDependent([1.0 2.0])
        @test_throws ArgumentError FrequencyDependent([1.0 -2.0; 0.0 1.0])
        @test_throws ArgumentError DensityDependent(c; groups = [(1,)])
        flat = flatten_contacts(rand(2, 2, 2, 2); names = (:age, :setting))
        @test FrequencyDependent(flat).groups == flat.groups
        @test DensityDependent(flat).labels == flat.labels
    end

    @testset "branching process offspring means" begin
        freq = BranchingProcess(FrequencyDependent(c), q, D, R -> Poisson(R), gt)
        @test offspring(freq) == permutedims(c) .* q .* D
        # The contact-matrix form is the frequency-dependent one.
        @test offspring(BranchingProcess(c, q, D, R -> Poisson(R), gt)) == offspring(freq)

        r = [0.002 0.0004; 0.0004 0.0002]
        sizes = [1500, 3500]
        dens = BranchingProcess(
            DensityDependent(r), q, D, R -> Poisson(R), gt; population = Population(sizes)
        )
        @test offspring(dens) ≈ permutedims(r) .* sizes .* q .* D
        # Equal to the frequency form once the contacts are rate times size.
        same = BranchingProcess(
            FrequencyDependent(r .* permutedims(sizes)), q, D, R -> Poisson(R), gt
        )
        @test offspring(dens) ≈ offspring(same)

        @test_throws "size of each group" BranchingProcess(
            DensityDependent(r), q, D, R -> Poisson(R), gt
        )
        @test_throws "size of each group" BranchingProcess(
            DensityDependent(r), q, D, R -> Poisson(R), gt; population_size = 5000
        )

        labelled = BranchingProcess(
            FrequencyDependent(c; labels = ["child", "adult"]), q, D, R -> Poisson(R), gt
        )
        @test labelled.type_labels == ["child", "adult"]
    end

    @testset "a finite population is depleted by group under either form" begin
        model = BranchingProcess(
            FrequencyDependent([3.0 3.0; 3.0 3.0]), 0.5, 2.0, R -> Poisson(R), gt;
            population = Population([5, 2000])
        )
        state = simulate(model; n_initial = 1, max_cases = 1500, rng = StableRNG(1))
        @test state.group_cases[1] <= 5
    end

    @testset "MixingProcess pair rates" begin
        r = [0.002 0.0004; 0.0004 0.0002]
        freq = MixingProcess(;
            population = Population([100, 300]), mixing_by = (:band,),
            mixing = FrequencyDependent(c), q = 0.2
        )
        @test freq.rate((1,), (2,)) ≈ c[1, 2] * 0.2 / 300
        @test freq.rate((2,), (1,)) ≈ c[2, 1] * 0.2 / 100

        dens = MixingProcess(;
            population = Population([100, 300]), mixing_by = (:band,),
            mixing = DensityDependent(r), q = 0.2
        )
        @test dens.rate((1,), (2,)) ≈ r[1, 2] * 0.2
        # Density dependence needs no sizes for the rate.
        unsized = MixingProcess(;
            population_size = 400, mixing_by = (:band,), mixing = DensityDependent(r), q = 0.2
        )
        @test unsized.rate((2,), (2,)) ≈ r[2, 2] * 0.2

        @test_throws "size of each group" MixingProcess(;
            population_size = 400, mixing_by = (:band,), mixing = FrequencyDependent(c)
        )
        @test_throws ArgumentError MixingProcess(;
            population = Population([100, 300]), mixing_by = (:band,),
            mixing = FrequencyDependent(c), rate = (a, b) -> 0.1
        )
        @test_throws ArgumentError MixingProcess(; population = Population([100, 300]))
        @test_throws ArgumentError MixingProcess(;
            population = Population([100, 300]), mixing = FrequencyDependent(c)
        )
        @test_throws ArgumentError MixingProcess(;
            population = Population([100, 200, 300]), mixing_by = (:band,),
            mixing = FrequencyDependent(c)
        )
    end

    @testset "mixing groups and population groups agree" begin
        named = MixingProcess(;
            population = Population([100, 300]; groups = [("a",), ("b",)]),
            mixing_by = (:band,), mixing = FrequencyDependent(c; groups = [("a",), ("b",)])
        )
        @test named.rate(("a",), ("b",)) ≈ c[1, 2] / 300
        @test_throws ArgumentError MixingProcess(;
            population = Population([100, 300]; groups = [("a",), ("b",)]),
            mixing_by = (:band,), mixing = FrequencyDependent(c; groups = [("x",), ("y",)])
        )
        # The mixing value's groups label a population that names none.
        from_mixing = MixingProcess(;
            population = Population([100, 300]), mixing_by = (:band,),
            mixing = FrequencyDependent(c; groups = [("a",), ("b",)])
        )
        @test EpiBranch.population(from_mixing).groups == [("a",), ("b",)]
    end

    @testset "the mixing value gives the same run as the equivalent rate" begin
        prog = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)]
        n = [200, 300]
        contacts = [3.0 0.5; 0.5 0.5]
        by_mixing = MixingProcess(;
            population = Population(n), mixing_by = (:band,),
            mixing = FrequencyDependent(contacts)
        )
        by_rate = MixingProcess(;
            population = Population(n), mixing_by = (:band,), rate = mixing_rate(contacts, n)
        )
        a = simulate(ModelSpec(by_mixing; progression = prog); n_initial = 3, rng = StableRNG(4))
        b = simulate(ModelSpec(by_rate; progression = prog); n_initial = 3, rng = StableRNG(4))
        @test a.cumulative_cases == b.cumulative_cases
        @test isequal(
            [i.infection_time for i in a.individuals],
            [i.infection_time for i in b.individuals]
        )
    end
end

@testset "MassActionHazard takes the population" begin
    inf = [0.0, 1.0, 2.0, NaN, NaN]
    rem = [3.0, 4.0, 5.0, NaN, NaN]
    schedule(m) = EpiBranch.mass_action_schedule(m, inf, rem)

    # Without one it is the people in the data.
    @test schedule(MassActionHazard(; transmission_rate = 0.5)).N == 5
    @test schedule(MassActionHazard(; transmission_rate = 0.5, population = Population(800))).N == 800
    @test schedule(MassActionHazard(; transmission_rate = 0.5, population_size = 800)).N == 800
    @test_throws ArgumentError MassActionHazard(;
        transmission_rate = 0.5, population = Population([400, 400])
    )
    @test_throws ArgumentError MassActionHazard(;
        transmission_rate = 0.5, population = Population(8), population_size = 8
    )
end

@testset "Mean-field processes take a population" begin
    a = MeanFieldProcess(;
        population = Population(1000), transmission_rate = 0.4, removal_rate = 0.2
    )
    b = MeanFieldProcess(; population_size = 1000, transmission_rate = 0.4, removal_rate = 0.2)
    @test EpiBranch.population_size(a) == EpiBranch.population_size(b) == 1000
    @test EpiBranch.population_size(
        LinearNoiseProcess(;
            population = Population(1000), transmission_rate = 0.4, removal_rate = 0.2
        )
    ) == 1000
    @test_throws ArgumentError MeanFieldProcess(;
        population = Population([500, 500]), transmission_rate = 0.4, removal_rate = 0.2
    )

    # A population with sizes by group fixes the group sizes of the mixing
    # approximation, which reads them from the population it describes.
    prog = [Transition(:recovered; rate = 0.2, terminal = true)]
    mixing = MixingProcess(;
        population = Population([2000, 8000]), mixing_by = (:age_band,),
        mixing = FrequencyDependent([0.3 0.1; 0.1 0.3])
    )
    approx = MixingMeanFieldProcess(ModelSpec(mixing; progression = prog); removal_rate = 0.2)
    @test approx.group_sizes == [2000, 8000]
end
