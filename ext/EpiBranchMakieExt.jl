module EpiBranchMakieExt

using EpiBranch: EpiBranch, SimulationState, incidence, summarise
using DataFrames: AbstractDataFrame, groupby
using Dates: Date, Day
using Distributions: DiscreteUnivariateDistribution, pdf
using Makie: Makie, Figure, Axis, band!, lines!, barplot!, scatter!, scatterlines!, axislegend

# Bands cannot take dates, so dates are plotted as day numbers.
_day_numbers(dates) = Float64.((dates .- first(dates)) ./ Day(1))

function _date_axis(fig, dates, interval)
    start = first(dates)
    return Axis(
        fig[1, 1]; xlabel = "Date", ylabel = "Cases per $interval days",
        xtickformat = vals -> [string(start + Day(round(Int, v))) for v in vals]
    )
end

function EpiBranch.epicurve(
        states::Vector{<:SimulationState};
        interval::Integer = 7, by::Symbol = :onset,
        reference_date::Date = Date(2020, 1, 1), range = nothing,
        probs = (0.025, 0.25, 0.5, 0.75, 0.975)
    )
    length(probs) == 5 || throw(ArgumentError("epicurve: probs needs five probabilities"))
    counts = incidence(states; interval, by, reference_date, range)
    q = summarise(counts; by = :date, probs = collect(probs))
    isempty(q) && throw(ArgumentError("epicurve: no cases to plot"))
    cols = names(q)[3:end]
    fig = Figure()
    x = _day_numbers(q.date)
    ax = _date_axis(fig, q.date, interval)
    outer = round(Int, 100 * (probs[5] - probs[1]))
    inner = round(Int, 100 * (probs[4] - probs[2]))
    band!(ax, x, q[!, cols[1]], q[!, cols[5]]; color = (:steelblue, 0.2), label = "$outer% band")
    band!(ax, x, q[!, cols[2]], q[!, cols[4]]; color = (:steelblue, 0.4), label = "$inner% band")
    lines!(ax, x, q[!, cols[3]]; color = :steelblue, linewidth = 2, label = "Median")
    axislegend(ax; position = :rt)
    return fig
end

function EpiBranch.epicurve(
        state::SimulationState;
        interval::Integer = 7, by::Symbol = :onset,
        reference_date::Date = Date(2020, 1, 1), range = nothing
    )
    counts = incidence(state; interval, by, reference_date, range)
    isempty(counts) && throw(ArgumentError("epicurve: no cases to plot"))
    fig = Figure()
    ax = _date_axis(fig, counts.date, interval)
    barplot!(ax, _day_numbers(counts.date), counts.cases; color = :steelblue)
    return fig
end

function EpiBranch.chain_size_plot(
        sizes::AbstractVector{<:Integer}, law::DiscreteUnivariateDistribution;
        max_size::Union{Nothing, Integer} = nothing
    )
    isempty(sizes) && throw(ArgumentError("chain_size_plot: sizes is empty"))
    max_size === nothing && (max_size = maximum(sizes))
    max_size >= 1 || throw(ArgumentError("chain_size_plot: max_size must be at least 1"))
    grid = 1:max_size
    observed = [count(==(s), sizes) / length(sizes) for s in grid]
    fig = Figure()
    ax = Axis(
        fig[1, 1]; xlabel = "Chain size", ylabel = "Proportion of chains",
        yscale = Makie.log10
    )
    seen = findall(>(0), observed)
    isempty(seen) && throw(ArgumentError("chain_size_plot: no sizes lie within 1:max_size"))
    barplot!(
        ax, grid[seen], observed[seen];
        color = (:steelblue, 0.6), label = "Simulated", fillto = minimum(observed[seen]) / 10
    )
    scatter!(ax, collect(grid), pdf.(law, grid); color = :black, label = "Exact")
    axislegend(ax; position = :rt)
    return fig
end

function EpiBranch.containment_plot(
        results::AbstractDataFrame, x::Symbol; group::Union{Nothing, Symbol} = nothing
    )
    for col in (x, :containment_probability, group)
        col === nothing || col in propertynames(results) ||
            throw(ArgumentError("containment_plot: results has no column $col"))
    end
    isempty(results) && throw(ArgumentError("containment_plot: results is empty"))
    results = copy(results)
    for col in (x, :containment_probability)
        results[!, col] = identity.(results[!, col])
    end
    eltype(results[!, x]) <: Real ||
        throw(ArgumentError("containment_plot: column $x is not numeric"))
    fig = Figure()
    ax = Axis(
        fig[1, 1]; xlabel = string(x), ylabel = "Containment probability",
        limits = (nothing, (0, 1))
    )
    if group === nothing
        sorted = sort(results, x)
        scatterlines!(ax, sorted[!, x], sorted.containment_probability; color = :steelblue)
    else
        for (key, part) in pairs(groupby(results, group; sort = true))
            sorted = sort(part, x)
            scatterlines!(
                ax, sorted[!, x], sorted.containment_probability; label = string(key[1])
            )
        end
        axislegend(ax, string(group); position = :rt)
    end
    return fig
end

end
