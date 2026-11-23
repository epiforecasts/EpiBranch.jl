module EpiBranchRecipesBaseExt

# `plot(state)` draws the transmission tree of a `SimulationState`: infection
# time on the horizontal axis, one row per case, and a line from each case to
# their infector.

using EpiBranch: EpiBranch, SimulationState, individuals
using RecipesBase: RecipesBase, @recipe, @series

# Rows are assigned depth-first so that a case sits between its descendants,
# with each infector placed at the mean row of their infectees.
function _tree_rows(people)
    ids = [p.id for p in people]
    index = Dict(id => k for (k, id) in enumerate(ids))
    children = [Int[] for _ in people]
    roots = Int[]
    for (k, p) in enumerate(people)
        parent = get(index, p.parent_id, 0)
        parent == 0 ? push!(roots, k) : push!(children[parent], k)
    end
    rows = zeros(Float64, length(people))
    next = Ref(0)
    stack = [(r, false) for r in Iterators.reverse(roots)]
    while !isempty(stack)
        k, done = pop!(stack)
        if isempty(children[k])
            rows[k] = (next[] += 1)
        elseif done
            rows[k] = sum(rows[c] for c in children[k]) / length(children[k])
        else
            push!(stack, (k, true))
            for c in Iterators.reverse(children[k])
                push!(stack, (c, false))
            end
        end
    end
    return rows, index
end

@recipe function f(state::SimulationState)
    people = individuals(state)
    rows, index = _tree_rows(people)
    xs = Float64[]
    ys = Float64[]
    for (k, p) in enumerate(people)
        parent = get(index, p.parent_id, 0)
        parent == 0 && continue
        t_parent = Float64(people[parent].infection_time)
        t = Float64(p.infection_time)
        append!(xs, (t_parent, t_parent, t, NaN))
        append!(ys, (rows[parent], rows[k], rows[k], NaN))
    end
    legend --> false
    yticks --> false
    xguide --> "Infection time (days)"
    yguide --> "Case"
    @series begin
        seriestype := :path
        linecolor --> :grey
        label := ""
        xs, ys
    end
    seriestype := :scatter
    markercolor --> :black
    label := ""
    Float64[p.infection_time for p in people], rows
end

end
