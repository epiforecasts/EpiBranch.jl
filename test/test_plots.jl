using EpiBranch, Distributions, StableRNGs, DataFrames, Dates, Test
using Makie

@testset "plots" begin
    @testset "arguments no method takes reach the fallback" begin
        @test_throws ArgumentError epicurve(1)
        @test_throws ArgumentError chain_size_plot(1)
    end

    model = ModelSpec(
        BranchingProcess(NegBin(1.5, 0.5), Exponential(5.0));
        attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
    )
    states = simulate(model, 30; rng = StableRNG(1), max_time = 60.0)

    @testset "epicurve" begin
        fig = epicurve(states)
        @test fig isa Makie.Figure
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        # Two bands and the median line.
        @test length(ax.scene.plots) == 3

        @test epicurve(states[1]; interval = 1) isa Makie.Figure
        @test epicurve(states; interval = 14, by = :infection) isa Makie.Figure
        @test_throws ArgumentError epicurve(states; probs = [0.1, 0.5, 0.9])
        @test_throws ArgumentError epicurve(EpiBranch.SimulationState[])
    end

    @testset "chain_size_plot" begin
        law = chain_size_distribution(Poisson(0.8))
        sizes = rand(StableRNG(2), law, 500)
        fig = chain_size_plot(sizes, law)
        @test fig isa Makie.Figure
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        # The bars are the observed proportions and the points the exact law.
        bars = only(filter(p -> p isa Makie.BarPlot, ax.scene.plots))
        heights = [p[2] for p in bars[1][]]
        @test sum(heights) ≈ sum(s <= maximum(sizes) for s in sizes) / 500
        @test chain_size_plot(sizes, law; max_size = 5) isa Makie.Figure
        @test_throws ArgumentError chain_size_plot(Int[], law)
    end
end
