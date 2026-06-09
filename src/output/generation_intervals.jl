"""
    realised_generation_interval(ind::Individual, state::SimulationState)

The realised forward generation interval for `ind`: the time from its
infector's infection to its own, `ind.infection_time -
parent.infection_time`. Index cases and individuals that were never
infected return `NaN`.

The interval is shaped by the epidemic. Susceptible depletion and
interventions decide which transmissions happen, so realised intervals
are distributed differently from the intrinsic `generation_time` you
give the model. This is the per-individual counterpart of
[`onset_time`](@ref).
"""
function realised_generation_interval(ind::Individual, state::SimulationState)
    (is_infected(ind) && ind.parent_id != 0) || return NaN
    parent = state.individuals[ind.parent_id]
    return ind.infection_time - parent.infection_time
end

"""
    realised_generation_intervals(state::SimulationState)

The realised forward generation intervals of every infected non-index
case in one simulation, as a `Vector{Float64}`. See
[`realised_generation_interval`](@ref).

For a model with a fixed `generation_time`, the intrinsic interval is
that distribution itself. For a state-dependent `generation_time`, run
the model without interventions and the realised intervals coincide
with the intrinsic ones, since nothing blocks transmission.
"""
function realised_generation_intervals(state::SimulationState)
    gts = Float64[]
    for ind in state.individuals
        (is_infected(ind) && ind.parent_id != 0) || continue
        parent = state.individuals[ind.parent_id]
        push!(gts, ind.infection_time - parent.infection_time)
    end
    return gts
end

"""
    realised_generation_intervals(states::Vector{<:SimulationState})

The realised forward generation intervals across several simulations,
flattened into one `Vector{Float64}`.
"""
function realised_generation_intervals(states::Vector{<:SimulationState})
    reduce(vcat, (realised_generation_intervals(s) for s in states);
        init = Float64[])
end
