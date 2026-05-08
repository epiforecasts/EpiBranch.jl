@testset "Attributes builders" begin
    @testset "transmission_traits" begin
        @testset "constants" begin
            attrs = transmission_traits(susceptibility = 0.3, infectiousness = 0.7)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 0.3
            @test ind.infectiousness == 0.7
        end

        @testset "default leaves traits at 1.0" begin
            attrs = transmission_traits()
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 1.0
            @test ind.infectiousness == 1.0
        end

        @testset "Real argument coerced to Float64" begin
            attrs = transmission_traits(susceptibility = 1//2)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility === 0.5
        end

        @testset "Distribution sampled per individual" begin
            attrs = transmission_traits(susceptibility = Beta(2, 5))
            rng = StableRNG(42)
            samples = map(1:200) do _
                ind = Individual(id = 1)
                attrs(rng, ind)
                ind.susceptibility
            end
            @test all(0 .<= samples .<= 1)
            @test length(unique(samples)) > 100  # genuinely sampled, not constant
        end

        @testset "Function dispatch sees ind state set by earlier builders" begin
            attrs = compose(
                demographics(age_distribution = Uniform(0, 90)),
                transmission_traits(
                    susceptibility = (rng, ind) -> ind.state[:age] >= 65 ? 0.8 : 0.2,
                )
            )
            rng = StableRNG(7)
            for _ in 1:200
                ind = Individual(id = 1)
                attrs(rng, ind)
                expected = ind.state[:age] >= 65 ? 0.8 : 0.2
                @test ind.susceptibility == expected
            end
        end

        @testset "Independent fields" begin
            # only susceptibility specified — infectiousness stays at default
            attrs = transmission_traits(susceptibility = 0.4)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 0.4
            @test ind.infectiousness == 1.0
        end

        @testset "Integrates with simulate" begin
            model = BranchingProcess(Poisson(1.5), Exponential(5.0))
            attrs = transmission_traits(susceptibility = 0.5, infectiousness = 0.8)
            state = simulate(model;
                attributes = attrs,
                sim_opts = SimOpts(max_cases = 30),
                rng = StableRNG(1))
            @test all(ind.susceptibility == 0.5 for ind in state.individuals)
            @test all(ind.infectiousness == 0.8 for ind in state.individuals)
        end
    end
end
