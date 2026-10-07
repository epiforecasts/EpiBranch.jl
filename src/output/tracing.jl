"""
    compute_trace_level!(state::SimulationState) -> state

Record how far along a chain of contact tracing each traced person was found:
`trace_level` 0 for the case a round of tracing started from, 1 for its
traced contacts, 2 for contacts of contacts, and so on. People never traced
(and cases from which no tracing started) get no level. Call it after
[`simulate`](@ref); the `!` means it changes `state` in place, adding the
level to each person's record. [`linelist`](@ref) then shows it as a
`trace_level` column.

!!! note "Level along the first tracing path, not the shortest"
    Each person is traced at most once, from the earliest exposure that led to
    them. In a `BranchingProcess` that is their infector, so the level is
    exact. On a `NetworkProcess`, where a person can have several infectious
    neighbours, it is the depth along the path by which they were first
    traced, which may be longer than the shortest path to an index case.
"""
function compute_trace_level!(state::SimulationState)
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
    compute_trace_level!(states::AbstractVector{<:SimulationState}) -> states

Add trace levels to each of several simulated outbreaks, as
[`compute_trace_level!`](@ref) does for one.
"""
function compute_trace_level!(states::AbstractVector{<:SimulationState})
    for state in states
        compute_trace_level!(state)
    end
    return states
end
