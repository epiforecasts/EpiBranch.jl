# The RecipesBase extension draws the transmission tree of a SimulationState.

using Test, EpiBranch, Distributions, StableRNGs
using RecipesBase: RecipesBase

@testset "Transmission tree plot recipe" begin
    model = BranchingProcess(Poisson(2.0), Exponential(5.0))
    state = simulate(model; rng = StableRNG(1), max_cases = 40)
    people = individuals(state)
    series = RecipesBase.apply_recipe(Dict{Symbol, Any}(), state)

    edges, points = series[1], series[2]
    xs, ys = edges.args
    # One three-point elbow (plus a NaN break) per case with an infector.
    @test length(xs) == 4 * count(p -> p.parent_id != 0, people)
    # Edges leave the infector's infection time and end at the infectee's.
    @test xs[1] == people[findfirst(p -> p.id == people[2].parent_id, people)].infection_time
    @test xs[3] == people[2].infection_time

    px, py = points.args
    @test px == [p.infection_time for p in people]
    leaves = [!any(q -> q.parent_id == p.id, people) for p in people]
    @test allunique(py[leaves])

    @testset "uninfected contacts are not drawn" begin
        state = simulate(model; rng = StableRNG(1), max_cases = 40)
        leaf = last(individuals(state))
        leaf.state[:infected] = false
        leaf.infection_time = NaN
        series = RecipesBase.apply_recipe(Dict{Symbol, Any}(), state)
        xs, ys = series[1].args
        px, py = series[2].args
        @test !any(isnan, px)
        @test !any(isnan, py)
        @test length(px) == length(filter(is_infected, individuals(state)))
        @test length(xs) == 4 * count(p -> is_infected(p) && p.parent_id != 0, individuals(state))
        @test count(isnan, xs) == length(xs) ÷ 4
    end
end
