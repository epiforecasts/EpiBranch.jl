"""
    chain_statistics(state::SimulationState)

Compute chain size and length for each transmission chain.
Only infected individuals are counted. Returns a DataFrame with
columns: chain_id, size, length.
"""
function chain_statistics(state::SimulationState)
    # Single-pass aggregation: track size and max generation per chain
    chain_size = Dict{Int, Int}()
    chain_maxgen = Dict{Int, Int}()

    for ind in state.individuals
        is_infected(ind) || continue
        cid = ind.chain_id
        chain_size[cid] = get(chain_size, cid, 0) + 1
        prev = get(chain_maxgen, cid, -1)
        gen = ind.generation
        gen > prev && (chain_maxgen[cid] = gen)
    end

    cids = sort!(collect(keys(chain_size)))
    DataFrame(
        chain_id = cids,
        size = [chain_size[c] for c in cids],
        length = [chain_maxgen[c] for c in cids]
    )
end

"""
    chain_statistics(states::Vector{<:SimulationState})

Compute chain statistics across multiple simulations.
A DataFrame with columns sim_id, chain_id, size, length is returned.
"""
function chain_statistics(states::Vector{<:SimulationState})
    sim_ids = Int[]
    chain_ids = Int[]
    sizes = Int[]
    lengths = Int[]

    for (s, state) in enumerate(states)
        cs = chain_statistics(state)
        # append column arrays directly instead of iterating eachrow
        n = nrow(cs)
        append!(sim_ids, fill(s, n))
        append!(chain_ids, cs.chain_id)
        append!(sizes, cs.size)
        append!(lengths, cs.length)
    end

    DataFrame(sim_id = sim_ids, chain_id = chain_ids, size = sizes, length = lengths)
end
