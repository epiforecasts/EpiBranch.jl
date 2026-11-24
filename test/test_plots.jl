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
        @test length(ax.scene.plots) == 3

        @test epicurve(states[1]; interval = 1) isa Makie.Figure
        @test epicurve(states; interval = 14, by = :infection) isa Makie.Figure
        fig90 = epicurve(states; probs = (0.05, 0.25, 0.5, 0.75, 0.95))
        ax90 = only(filter(b -> b isa Makie.Axis, fig90.content))
        labels = [p.label[] for p in ax90.scene.plots if p isa Makie.Band]
        @test labels == ["90% band", "50% band"]
        @test_throws ArgumentError epicurve(states; probs = [0.1, 0.5, 0.9])
        @test_throws ArgumentError epicurve(EpiBranch.SimulationState[])
    end

    @testset "chain_size_plot" begin
        law = chain_size_distribution(Poisson(0.8))
        sizes = rand(StableRNG(2), law, 500)
        fig = chain_size_plot(sizes, law)
        @test fig isa Makie.Figure
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        bars = only(filter(p -> p isa Makie.BarPlot, ax.scene.plots))
        heights = [p[2] for p in bars[1][]]
        @test sum(heights) ≈ sum(s <= maximum(sizes) for s in sizes) / 500
        @test chain_size_plot(sizes, law; max_size = 5) isa Makie.Figure
        @test_throws ArgumentError chain_size_plot(Int[], law)
        @test_throws ArgumentError chain_size_plot([5, 6, 7], law; max_size = 3)
        @test_throws ArgumentError chain_size_plot(sizes, law; max_size = 0)
    end

    @testset "containment_plot" begin
        @test_throws ArgumentError containment_plot(1, :x)
        sweep = DataFrame(
            delay = [5.0, 1.0, 5.0, 1.0], R = [1.5, 1.5, 3.0, 3.0],
            containment_probability = [0.6, 0.9, 0.1, 0.4]
        )
        fig = containment_plot(sweep[1:2, :], :delay)
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        line = only(filter(p -> p isa Makie.ScatterLines, ax.scene.plots))
        @test collect.(line[1][]) == [[1.0, 0.9], [5.0, 0.6]]
        grouped = containment_plot(sweep, :delay; group = :R)
        ax = only(filter(b -> b isa Makie.Axis, grouped.content))
        lines = filter(p -> p isa Makie.ScatterLines, ax.scene.plots)
        @test length(lines) == 2
        @test [collect.(p[1][]) for p in lines] == [
            [[1.0, 0.9], [5.0, 0.6]], [[1.0, 0.4], [5.0, 0.1]],
        ]
        @test_throws ArgumentError containment_plot(sweep, :missing)
        @test_throws ArgumentError containment_plot(sweep, :delay; group = :missing)
        @test_throws ArgumentError containment_plot(sweep[1:0, :], :delay)
        @test_throws ArgumentError containment_plot(DataFrame(x = ["a"], containment_probability = [1.0]), :x)
    end
end
