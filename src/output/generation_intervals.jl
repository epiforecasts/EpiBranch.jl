"""
    generation_interval(ind::Individual, state::SimulationState)

Realised forward generation interval for `ind`: the time from its
infector's infection to its own, `ind.infection_time -
parent.infection_time`. Returns `NaN` for index cases (no parent) and
for individuals that were not infected.

This is the realised interval, shaped by the epidemic — susceptible
depletion and interventions act on which transmissions occur, so the
distribution of realised intervals differs from the intrinsic
`generation_time` supplied to the model. It is the per-individual
mirror of [`onset_time`](@ref).
"""
function generation_interval(ind::Individual, state::SimulationState)
    (is_infected(ind) && ind.parent_id != 0) || return NaN
    parent = state.individuals[ind.parent_id]
    return ind.infection_time - parent.infection_time
end

"""
    realised_generation_intervals(state::SimulationState)

Collect the realised forward generation intervals across all infected
non-index cases in a single simulation (see
[`generation_interval`](@ref)). Returns a `Vector{Float64}`.

To recover the intrinsic generation interval for a model with a fixed
`generation_time`, read that distribution directly; for a
state-dependent `generation_time`, run the model without interventions
so that no transmissions are blocked.
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

Collect realised forward generation intervals across several
simulations into a single `Vector{Float64}`.
"""
function realised_generation_intervals(states::Vector{<:SimulationState})
    reduce(vcat, (realised_generation_intervals(s) for s in states);
        init = Float64[])
end
