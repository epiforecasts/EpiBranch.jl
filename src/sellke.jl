# ── Sellke/Dijkstra continuous-time race ─────────────────────────────
#
# The generic continuous-time competing-risks primitive shared by the
# structure-driven models whose contacts are a finite, depleting set of
# existing nodes (a household clique, say). Members are processed in
# increasing infection-time order; popping the earliest pending infection
# makes it final, because any not-yet-processed infector has a later
# infection time, becomes infectious no earlier, and so reaches it later.

# When the infector becomes infectious: the `from` state's time (the infection
# time itself when the kernel times from :infection, otherwise a state key).
function _window_open(ind::Individual{T}, from::Symbol) where {T}
    from === :infection ? ind.infection_time :
    convert(T, get(ind.state, Symbol(from, :_time), T(Inf)))
end

# Earliest of the `until` removal states' times (Inf if none reached).
function _window_close(ind::Individual{T}, until::Tuple) where {T}
    isempty(until) ? T(Inf) :
    minimum(convert(T, get(ind.state, Symbol(s, :_time), T(Inf))) for s in until)
end

"""
    _sellke_race!(state, members, rng; seed!, targets, from, until)

Run the Sellke/Dijkstra continuous-time competing-risks race over the individuals
`members` (global ids). `seed!(best, members, rng)` fills the candidate infection
times `best` (indexed `1:length(members)`) with exogenous introductions or a seed.
`targets(infective_id, state)` yields `(target_id, kernel)` for the contacts an
infective can reach, each with its pairwise contact-interval kernel. Members are
processed in increasing infection time (each pop is final); on processing, the
case's natural history is stamped and it exposes still-susceptible targets with a
`from`-timed contact interval accepted inside its infectious window.

The contact-interval `kernel` must be a **non-negative** distribution: the
"each pop is final" invariant relies on a candidate time `open_t + dt` never
preceding the infector's own window-open (`dt ≥ 0`). A kernel with support on
the negatives would break the shortest-path race with no error.
"""
function _sellke_race!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; seed!, targets, from::Symbol, until::Tuple)
    m = length(members)
    best = fill(Inf, m)
    src = zeros(Int, m)
    processed = falses(m)
    pos = Dict{Int, Int}(id => k for (k, id) in enumerate(members))

    seed!(best, members, rng)

    while true
        j = 0
        bt = Inf
        for k in 1:m
            if !processed[k] && best[k] < bt
                bt = best[k]
                j = k
            end
        end
        j == 0 && break
        processed[j] = true

        ind = state.individuals[members[j]]
        ind.infection_time = best[j]
        ind.state[:infected] = true
        ind.state[:index] = src[j] == 0
        if src[j] != 0
            infector = state.individuals[src[j]]
            ind.parent_id = infector.id
            ind.generation = infector.generation + 1
            ind.chain_id = infector.chain_id
        end
        resolve_transitions!(state, ind)

        open_t = _window_open(ind, from)
        isfinite(open_t) || continue
        close_t = _window_close(ind, until)

        for (target_id, kernel) in targets(members[j], state)
            k = get(pos, target_id, 0)
            (k == 0 || processed[k]) && continue
            dt = rand(rng, kernel)
            cand = open_t + dt
            (cand <= close_t && cand < best[k]) || continue
            best[k] = cand
            src[k] = members[j]
        end
    end
    return nothing
end
