"""
    compute_trace_level!(state::SimulationState) -> state

Record how far along a chain of contact tracing each traced person was found:
`trace_level` 0 for the case a round of tracing started from, 1 for its
traced contacts, 2 for contacts of contacts, and so on. People never traced
(and cases from which no tracing started) get no level. Call it after
[`simulate`](@ref); the `!` means it changes `state` in place, adding the
level to each person's record. [`linelist`](@ref) then shows it as a
`trace_level` column.

The level is the smallest number of tracing steps from any case that started
tracing without being traced itself. `ContactTracing` records every case whose
tracing reached a person, so on a network with cycles the level is the
distance to the nearest such case, found by a breadth-first search from all of
them at once, and not the length of whichever path happened to reach the person
last. A person traced only along a cycle that no such case leads into is
given no level.
"""
function compute_trace_level!(state::SimulationState)
    individuals = state.individuals

    # id → position lookup (don't assume id == index).
    byid = Dict{Int, Int}()
    for (i, ind) in pairs(individuals)
        byid[ind.id] = i
    end

    # Edges tracer → traced, and the untraced tracers the search starts from.
    # A hand-set `:traced_by` with no `:tracers` counts as the one tracer.
    children = Dict{Int, Vector{Int}}()
    traced = Set{Int}()
    for ind in individuals
        tracers = get(ind.state, :tracers, nothing)
        if tracers === nothing
            src = get(ind.state, :traced_by, nothing)
            src === nothing && continue
            tracers = (src::Int,)
        end
        push!(traced, ind.id)
        for t in tracers
            haskey(byid, t) || continue             # dangling reference
            push!(get!(() -> Int[], children, t), ind.id)
        end
    end

    levels = Dict{Int, Int}()
    queue = Int[]
    for id in keys(children)
        id in traced && continue
        levels[id] = 0
        push!(queue, id)
    end

    head = 1
    while head <= length(queue)
        id = queue[head]
        head += 1
        for child in get(children, id, ())
            haskey(levels, child) && continue
            levels[child] = levels[id] + 1
            push!(queue, child)
        end
    end

    for ind in individuals
        lvl = get(levels, ind.id, nothing)
        lvl === nothing || (ind.state[:trace_level] = lvl)
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
