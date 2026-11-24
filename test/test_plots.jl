using EpiBranch, Distributions, StableRNGs, DataFrames, Dates, Test
using Makie

@testset "plots" begin
    @testset "arguments no method takes reach the fallback" begin
        @test_throws ArgumentError epicurve(1)
        @test_throws ArgumentError chain_size_plot(1)
        @test_throws ArgumentError containment_plot(1)
        @test_throws ArgumentError end_of_outbreak_plot(1)
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
        @test epicurve(states; interval = 14, by = TransitionCount(:infection)) isa Makie.Figure
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
        results = DataFrame(
            R = repeat([0.5, 1.0, 2.0], 2), k = repeat([0.2, 0.2, 0.2, 1.0, 1.0, 1.0]; inner = 1),
            containment_probability = [0.9, 0.7, 0.3, 0.95, 0.6, 0.1]
        )
        fig = containment_plot(results, :R)
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        @test length(ax.scene.plots) == 1
        figg = containment_plot(results, :R; group = :k)
        axg = only(filter(b -> b isa Makie.Axis, figg.content))
        lines = filter(p -> p isa Makie.Lines, [q for p in axg.scene.plots for q in p.plots])
        @test length(lines) == 2
        @test_throws ArgumentError containment_plot(results, :missing)
        @test_throws ArgumentError containment_plot(results, :R; group = :missing)
        @test_throws ArgumentError containment_plot(DataFrame(x = ["a"], containment_probability = [0.5]), :x)

        swept = scenario_sweep(
            Dict(:offspring => [NegBin(0.5, 0.5), NegBin(2.0, 0.5)]); n_sim = 20, max_cases = 100,
            rng = StableRNG(3)
        )
        swept.R = [mean(o) for o in swept.offspring]
        @test containment_plot(swept, :R) isa Makie.Figure
    end

    @testset "end_of_outbreak_plot" begin
        offspring = NegBin(0.8, 0.5)
        gt = Gamma(2.0, 3.0)
        state = first(states)
        fig = end_of_outbreak_plot(state, offspring, gt)
        @test fig isa Makie.Figure
        ax = only(filter(b -> b isa Makie.Axis, fig.content))
        line = only(filter(p -> p isa Makie.Lines, ax.scene.plots))
        ts, ps = [p[1] for p in line[1][]], [p[2] for p in line[1][]]
        k = 7
        inputs = end_of_outbreak_inputs(state, ts[k]; by = TransitionCount(:onset))
        @test ps[k] ≈ end_of_outbreak_probability(offspring, gt, inputs.τs, inputs.secondary_cases)
        @test all(0 .<= ps .<= 1)
        @test ps[end] > 0.9
        @test end_of_outbreak_plot(state, offspring, gt; times = [10.0, 20.0]) isa Makie.Figure
        @test_throws ArgumentError end_of_outbreak_plot(state, offspring, gt; times = [-5.0])
    end
end
