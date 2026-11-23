"""
    epicurve(states::Vector{<:SimulationState}; interval=7, by = TransitionCount(:onset),
             reference_date=Date(2020, 1, 1), range=nothing,
             probs=(0.025, 0.25, 0.5, 0.75, 0.975))
    epicurve(state::SimulationState; interval=7, by = TransitionCount(:onset),
             reference_date=Date(2020, 1, 1), range=nothing)

Epidemic curve of simulated outbreaks as a Makie `Figure`, available after
`using Makie` or a backend such as `using CairoMakie`. Cases are counted with
[`incidence`](@ref), and the keywords of `incidence` are accepted. For a vector of outbreaks it
shows the median with 50% and 95% bands from [`summarise`](@ref); `probs` is
the five probabilities (lower 95%, lower 50%, median, upper 50%, upper 95%).
A single outbreak is drawn as bars.
"""
function epicurve(args...; kwargs...)
    throw(ArgumentError("epicurve needs a Makie backend: run `using CairoMakie` (or another Makie backend) first"))
end

"""
    chain_size_plot(sizes::AbstractVector{<:Integer}, law::DiscreteUnivariateDistribution;
                    max_size=maximum(sizes))

Chain sizes against their exact law as a Makie `Figure`, available after
`using Makie` or a backend such as `using CairoMakie`. Bars show the
proportion of `sizes` at each size from 1 to `max_size`, and points show
`pdf(law, size)`.
"""
function chain_size_plot(args...; kwargs...)
    throw(ArgumentError("chain_size_plot needs a Makie backend: run `using CairoMakie` (or another Makie backend) first"))
end
