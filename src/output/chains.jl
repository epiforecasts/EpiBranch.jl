"""
    chain_statistics(state::SimulationState)

Size and length of each transmission chain in one simulated outbreak, one row
per chain. A chain is all the cases descended from one index case. Only
infected people are counted.

Returns a DataFrame with columns `chain_id`, `size` (number of cases in the
chain) and `length` (number of generations of onward transmission: the
highest generation reached, so an index case that infects nobody has
`size = 1` and `length = 0`).

!!! note "Chain length is one less than in epichains"
    `length` here, as in [`ChainLengths`](@ref), counts generations of
    transmission after the index case. epichains' `chain_length` counts
    generations including the index case, so a single-case chain has length
    1 there and 0 here. Add 1 to compare with epichains.

# Examples

```julia
model = ModelSpec(BranchingProcess(NegBin(0.8, 0.5), Gamma(2.0, 3.0)))
state = simulate(model; n_initial = 20)
chain_statistics(state)   # one row per index case
```
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
    return DataFrame(
        chain_id = cids,
        size = [chain_size[c] for c in cids],
        length = [chain_maxgen[c] for c in cids]
    )
end

"""
    chain_statistics(states::Vector{<:SimulationState})

Chain sizes and lengths across several simulated outbreaks, such as the
output of `simulate(model, n)`. Returns a DataFrame with columns `sim_id`
(which simulation), `chain_id`, `size` and `length`, defined as for a single
outbreak above.
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

    return DataFrame(sim_id = sim_ids, chain_id = chain_ids, size = sizes, length = lengths)
end
