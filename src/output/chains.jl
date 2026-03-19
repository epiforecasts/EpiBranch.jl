"""
    chain_statistics(state::SimulationState)

Compute chain size and length for each transmission chain.
Only infected individuals are counted. Returns a DataFrame with
columns: chain_id, size, length.
"""
function chain_statistics(state::SimulationState)
    infected = filter(is_infected, state.individuals)
    chain_ids = unique(ind.chain_id for ind in infected)

    cids = Int[]
    sizes = Int[]
    lengths = Int[]

    for cid in chain_ids
        chain_inds = filter(i -> i.chain_id == cid, infected)
        push!(cids, cid)
        push!(sizes, length(chain_inds))
        push!(lengths, maximum(i.generation for i in chain_inds))
    end

    DataFrame(chain_id=cids, size=sizes, length=lengths)
end

"""
    chain_statistics(states::Vector{<:SimulationState})

Compute chain statistics across multiple simulations.
Returns a DataFrame with columns: sim_id, chain_id, size, length.
"""
function chain_statistics(states::Vector{<:SimulationState})
    sim_ids = Int[]
    chain_ids = Int[]
    sizes = Int[]
    lengths = Int[]

    for (s, state) in enumerate(states)
        cs = chain_statistics(state)
        # Issue 10: append column arrays directly instead of iterating eachrow
        n = nrow(cs)
        append!(sim_ids, fill(s, n))
        append!(chain_ids, cs.chain_id)
        append!(sizes, cs.size)
        append!(lengths, cs.length)
    end

    DataFrame(sim_id=sim_ids, chain_id=chain_ids, size=sizes, length=lengths)
end
