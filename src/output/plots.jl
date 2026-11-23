# Plot entry points. The methods live in the `EpiBranchMakieExt` extension,
# which loads with Makie; this file holds the generics and the message shown
# before it has.

const _PLOT_FALLBACK_MESSAGE = "needs a Makie backend: run `using CairoMakie` (or another Makie backend) first"

"""
    epicurve(states::Vector{<:SimulationState}; interval=7, by=:onset,
             reference_date=Date(2020, 1, 1), range=nothing,
             probs=(0.025, 0.25, 0.5, 0.75, 0.975))
    epicurve(state::SimulationState; interval=7, by=:onset,
             reference_date=Date(2020, 1, 1), range=nothing)

Epidemic curve of simulated outbreaks as a Makie `Figure`, available after
`using Makie` or a backend such as `using CairoMakie`. Cases are counted with
[`incidence`](@ref), whose `interval`, `by`, `reference_date` and `range`
keywords it takes. For a vector of outbreaks the figure shows the median
across outbreaks as a line with the 50% and 95% bands around it, the
[`summarise`](@ref) quantiles; `probs` is the five probabilities (lower 95%,
lower 50%, median, upper 50%, upper 95%). A single outbreak is drawn as bars.
"""
function epicurve(args...; kwargs...)
    throw(ArgumentError("epicurve $_PLOT_FALLBACK_MESSAGE"))
end

"""
    chain_size_plot(sizes::AbstractVector{<:Integer}, law::DiscreteUnivariateDistribution;
                    max_size=maximum(sizes))

Chain sizes against their exact law as a Makie `Figure`, available after
`using Makie` or a backend such as `using CairoMakie`. Bars show the
proportion of `sizes` at each size from 1 to `max_size`, and points show
`pdf(law, size)`, for example the law from [`chain_size_distribution`](@ref).
`sizes` is typically the `size` column of [`chain_statistics`](@ref).
"""
function chain_size_plot(args...; kwargs...)
    throw(ArgumentError("chain_size_plot $_PLOT_FALLBACK_MESSAGE"))
end
