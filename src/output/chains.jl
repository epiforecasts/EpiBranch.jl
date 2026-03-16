"""
    chain_statistics(state::SimulationState)

Compute chain size and length for each transmission chain in the simulation.
Returns a DataFrame with columns: chain_id, size, length.
"""
function chain_statistics(state::SimulationState)
    chain_ids = unique(ind.chain_id for ind in state.individuals)

    cids = Int[]
    sizes = Int[]
    lengths = Int[]

    for cid in chain_ids
        chain_inds = filter(i -> i.chain_id == cid, state.individuals)
        push!(cids, cid)
        push!(sizes, length(chain_inds))
        push!(lengths, maximum(i.generation for i in chain_inds))
    end

    DataFrame(chain_id=cids, size=sizes, length=lengths)
end

"""
    chain_statistics(states::Vector{SimulationState})

Compute chain statistics across multiple simulations.
Returns a DataFrame with columns: sim_id, chain_id, size, length.
"""
function chain_statistics(states::Vector{SimulationState})
    sim_ids = Int[]
    chain_ids = Int[]
    sizes = Int[]
    lengths = Int[]

    for (s, state) in enumerate(states)
        cs = chain_statistics(state)
        for row in eachrow(cs)
            push!(sim_ids, s)
            push!(chain_ids, row.chain_id)
            push!(sizes, row.size)
            push!(lengths, row.length)
        end
    end

    DataFrame(sim_id=sim_ids, chain_id=chain_ids, size=sizes, length=lengths)
end
