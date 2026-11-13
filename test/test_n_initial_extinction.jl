using Distributions: DiscreteNonParametric, Poisson

@testset "extinction_probability n_initial" begin
    q = extinction_probability(2.0, 0.3)
    @test extinction_probability(2.0, 0.3; n_initial = 3) ≈ q^3
    @test extinction_probability(NegBin(2.0, 0.3); n_initial = 3) ≈ q^3
    @test extinction_probability(Poisson(2.0); n_initial = 2) ≈
        extinction_probability(Poisson(2.0))^2
    @test epidemic_probability(2.0, 0.3; n_initial = 3) ≈ 1 - q^3
    @test epidemic_probability(NegBin(2.0, 0.3); n_initial = 3) ≈
        probability_contain(2.0, 0.3; n_initial = 3) |> x -> 1 - x
    @test extinction_probability(0.5, 1.0; n_initial = 4) == 1.0
    @test_throws ArgumentError extinction_probability(2.0, 0.3; n_initial = 0)
end

@testset "n_initial with ClusterMixed" begin
    # θ straddles R = 1, so the extinction probability varies with θ.
    mixing = DiscreteNonParametric([0.5, 3.0], [0.5, 0.5])
    cm = ClusterMixed(Poisson, mixing)
    q = 0.5 * 1.0 + 0.5 * extinction_probability(Poisson(3.0))
    @test extinction_probability(cm) ≈ q
    @test extinction_probability(cm; n_initial = 3) ≈ q^3
    @test_throws ArgumentError extinction_probability(cm; n_initial = 0)
end

@testset "n_initial with models and multi-type offspring" begin
    model = BranchingProcess(NegBin(2.0, 0.3))
    @test extinction_probability(model; n_initial = 2) ≈
        extinction_probability(2.0, 0.3)^2
    @test epidemic_probability(model; n_initial = 2) ≈
        1 - extinction_probability(2.0, 0.3)^2
    mt = EpiBranch.MultiTypeOffspring([1.2 0.3; 0.3 1.2], R -> NegBin(R, 0.5))
    @test extinction_probability(mt; n_initial = 1) == extinction_probability(mt)
    @test_throws ArgumentError extinction_probability(mt; n_initial = 2)
    @test_throws ArgumentError epidemic_probability(mt; n_initial = 2)
end
