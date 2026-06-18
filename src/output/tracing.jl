"""
    derive_trace_level!(state::SimulationState) -> state

Post-run enrichment: walk each individual's `:traced_by` back to the index
case and stamp `:trace_level` (distance from the index case) onto its
`state`. The ring anchor — the seeded, eligible case a trace started from —
is level `0`; its directly traced contacts are `1`, contacts-of-contacts
`2`, and so on. Cases that were never traced (and never anchored a realised
ring) are left without a `:trace_level`.

`:trace_level` then flows into [`linelist`](@ref) automatically, and any
other `state` consumer can read it with `get(ind.state, :trace_level,
missing)`. The engine carries no level during simulation — this is a
deliberate post-run step, so the cost stays out of the hot loop.

!!! note "First-traced, not nearest"
    `:traced_by` records the *first / earliest-exposure* tracer, because the
    engine makes one trace attempt per node. On a tree (`BranchingProcess`)
    that is the unique parent, so the level is exact. On a cyclic
    `NetworkProcess` it is the depth along the first-traced path, which is
    **not** guaranteed to be the shortest distance to the nearest index —
    do not read it as one. (A true nearest-index distance would need the
    engine to record every successful tracer; see issue #150.)
"""
function derive_trace_level!(state::SimulationState)
    individuals = state.individuals

    # id → position lookup (don't assume id == index).
    byid = Dict{Int, Int}()
    for (i, ind) in pairs(individuals)
        byid[ind.id] = i
    end

    levels = Dict{Int, Int}()   # id → level, for traced nodes (memoised)
    anchors = Set{Int}()        # untraced ids a traced chain terminates on

    # Level of `id`, following `:traced_by` to the first untraced ancestor
    # (the anchor, level 0). `visiting` guards against the cycle that should
    # never occur — `:traced_by` points to earlier-infected nodes.
    function level_of(id::Int, visiting::Set{Int})
        idx = get(byid, id, 0)
        idx == 0 && return 0                       # dangling reference
        src = get(individuals[idx].state, :traced_by, nothing)
        if src === nothing                          # untraced terminus
            push!(anchors, id)
            return 0
        end
        haskey(levels, id) && return levels[id]
        id in visiting && return 0                  # defensive cycle guard
        push!(visiting, id)
        lvl = level_of(src::Int, visiting) + 1
        delete!(visiting, id)
        levels[id] = lvl
        return lvl
    end

    for ind in individuals
        get(ind.state, :traced_by, nothing) === nothing && continue
        level_of(ind.id, Set{Int}())
    end

    for ind in individuals
        if haskey(levels, ind.id)
            ind.state[:trace_level] = levels[ind.id]
        elseif ind.id in anchors
            ind.state[:trace_level] = 0
        end
    end
    return state
end

"""
    derive_trace_level!(states::AbstractVector{<:SimulationState}) -> states

Apply [`derive_trace_level!`](@ref) to each state in turn (mirrors
`chain_statistics` over a batch of runs).
"""
function derive_trace_level!(states::AbstractVector{<:SimulationState})
    for state in states
        derive_trace_level!(state)
    end
    return states
end
