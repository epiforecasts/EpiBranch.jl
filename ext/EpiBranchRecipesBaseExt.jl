module EpiBranchRecipesBaseExt

using EpiBranch: EpiBranch, SimulationState, individuals
using RecipesBase: RecipesBase, @recipe, @series

@recipe function f(state::SimulationState)
    people = individuals(state)
    index = Dict(p.id => k for (k, p) in enumerate(people))
    children = [Int[] for _ in people]
    roots = Int[]
    for (k, p) in enumerate(people)
        parent = get(index, p.parent_id, 0)
        parent == 0 ? push!(roots, k) : push!(children[parent], k)
    end

    # Depth-first rows, each infector at the mean row of their infectees.
    rows = zeros(Float64, length(people))
    next = 0
    stack = [(r, false) for r in Iterators.reverse(roots)]
    while !isempty(stack)
        k, done = pop!(stack)
        if isempty(children[k])
            rows[k] = (next += 1)
        elseif done
            rows[k] = sum(rows[c] for c in children[k]) / length(children[k])
        else
            push!(stack, (k, true))
            for c in Iterators.reverse(children[k])
                push!(stack, (c, false))
            end
        end
    end

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
