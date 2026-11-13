using Distributions: Poisson

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
