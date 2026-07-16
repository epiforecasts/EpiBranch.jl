# ── Sellke fixed-size population pool ────────────────────────────────
#
# A closed population of N pre-allocated individuals, bucketed by mixing type.
# Each susceptible carries a fixed resistance threshold `Q_j ~ Exponential(1)`
# and accumulates infection pressure `Λ(t) = ∫ λ(s) ds` at the force of
# infection `λ` felt by its own mixing type; it is infected the instant `Λ`
# crosses its threshold. A model names *which real attributes define mixing*
# (`mixing_by`, e.g. `(:age_band, :ses)`); an individual's mixing type is the
# tuple of those attribute values it already carries, and susceptibles that share
# a type feel a common force. The force is supplied by the model as a function
# `force(type, counts)` of the mixing type and the per-type infectious counts —
# so a homogeneous pool names no attributes (`mixing_by = ()`, one type `()`) and
# passes `β/N·Σ counts`, while an age- or space-structured model passes a contact
# matrix keyed on the attribute values. This is the Sellke threshold
# construction; the homogeneous case reproduces the exact stochastic SIR
# final-size law, with `R0 = β·E[infectious period]`.
#
# The integration is event-driven so that infection *times* — not just the
# final size — fall out exactly: between events each group's pressure grows
# linearly at its own force `λ`, so its next threshold crossing has a
# closed-form time, and the candidate events (each group's next infection, the
# next window-open, the next window-close) are raced. Every other concern is the
# shared engine: each infected member's natural history is stamped through
# `resolve_transitions!` and its interventions resolved, so a removing
# intervention (isolation) acts by shortening the infectious window through the
# window-close, exactly as in `_sellke_race!`.

# A minimal binary min-heap over `(time, id)` pairs — tuples compare
# lexicographically, so ties break deterministically on id. Avoids a
# DataStructures dependency for the two pending-event queues.
function _pool_heap_push!(h::Vector{Tuple{T, Int}}, x::Tuple{T, Int}) where {T <: Real}
    push!(h, x)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        h[p] <= h[i] && break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return h
end

function _pool_heap_peek(h::Vector{Tuple{T, Int}}) where {T <: Real}
    isempty(h) ? (T(Inf), 0) : @inbounds h[1]
end

function _pool_heap_pop!(h::Vector{Tuple{T, Int}}) where {T <: Real}
    n = length(h)
    n == 0 && return (T(Inf), 0)
    @inbounds top = h[1]
    @inbounds last = h[n]
    pop!(h)
    n -= 1
    if n > 0
        @inbounds h[1] = last
        i = 1
        @inbounds while true
            l = 2i
            r = 2i + 1
            s = i
            (l <= n && h[l] < h[s]) && (s = l)
            (r <= n && h[r] < h[s]) && (s = r)
            s == i && break
            h[i], h[s] = h[s], h[i]
            i = s
        end
    end
    return top
end

"""
    _sellke_pool!(state, members, rng; mixing_by = (), force, n_initial, from, until)

Run the Sellke threshold construction over the `members` (global ids) of a
closed population, infecting individuals in continuous time and stamping each
case's natural history.

A model names **which real attributes define mixing** through `mixing_by`, a
tuple of attribute keys, for example `(:age_band, :ses)`. An individual's
**mixing type** is the tuple of those attribute values read off its own state,
`Tuple(get(ind.state, k, missing) for k in mixing_by)`. These are the real
attributes an individual already carries: age band, patch, risk group, read
straight off its state. With `mixing_by = ()` every individual has the empty type
`()`, a single homogeneous group with no tagging needed. The one model-level
input is the mixing rule between types:

  - `force(type, counts)::Float64` is the per-susceptible force of infection on a
    susceptible of mixing type `type`, given `counts`, a `Dict` mapping each
    mixing type to the number currently infectious of that type. It must be
    piecewise-constant
    between events, which it is: `counts` only changes at an infection, a window
    opening or a window closing. Homogeneous mixing is
    `force = (type, counts) -> beta / N * sum(values(counts))`.

`n_initial` is the number of index cases seeded at time 0, `from` the state the
infectious window opens at and `until` the removal states that close it. Each
susceptible carries a fixed `Exponential(1)` resistance threshold and is infected
the instant its accumulated pressure crosses it; pressure accumulates at the
force felt by its mixing type. Writes per-individual state directly; the caller
reconciles aggregate bookkeeping and applies observation.
"""
function _sellke_pool!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; mixing_by::Tuple = (), force, n_initial::Integer,
        from::Symbol, until::Tuple, interventions = ())
    N = length(members)
    N == 0 && return nothing

    # The pool carries the state's timing type `T` (Float64 by default, a dual
    # or stochastic-triple type under automatic differentiation): pressures,
    # forces, event times and thresholds promote to it so gradients flow through
    # the crossing times. The Exponential(1) resistance thresholds are drawn as
    # constants and stay Float64 — they are the reparameterisation, held fixed
    # while parameters vary.
    T = _timetype(state)

    # Stamp an individual infected at time τ with infector id `src` (0 = index),
    # matching `_sellke_race!`'s conventions, then run its natural history.
    stamp! = function (ind, τ, src)
        ind.infection_time = τ
        ind.state[:infected] = true
        ind.state[:index] = src == 0
        if src != 0
            infector = state.individuals[src]
            ind.parent_id = infector.id
            ind.generation = infector.generation + 1
            ind.chain_id = infector.chain_id
        end
        resolve_transitions!(state, ind)
        _resolve_interventions!(state, ind, interventions)
        return nothing
    end

    # Mixing type of each member, read once as the tuple of its `mixing_by`
    # attribute values (`missing` for any it lacks). With `mixing_by = ()` every
    # member has the empty type `()` — a single homogeneous group.
    typ = Dict{Int, Any}()
    for id in members
        s = state.individuals[id].state
        typ[id] = Tuple(get(s, k, missing) for k in mixing_by)
    end

    # Current infectious count per mixing type: one persistent Dict, mutated in
    # place and handed to `force` (never reallocated). Every type present starts
    # at 0, so index cases and susceptibles of any type key in without a miss.
    counts = Dict{Any, Int}()
    for id in members
        counts[typ[id]] = 0
    end

    open_heap = Tuple{T, Int}[]         # pending window-open (becomes infectious)
    close_heap = Tuple{T, Int}[]        # pending window-close (recovers/isolates)
    infectious_ids = Int[]              # ids currently infectious (window open)
    slot = Dict{Int, Int}()             # id → its index in `infectious_ids`

    # Push an infected individual's infectious window onto the queues. A never-
    # infectious case (window-open is Inf) contributes nothing to the force; a
    # never-closing window (Inf close, e.g. no removal state) simply stays open.
    # A case removed at or before its infectious onset (e.g. isolated during a
    # latent period, so `close_t <= open_t`) is never infectious: skip both queues
    # so no window opens and no close is popped against an id that was never added.
    push_windows! = function (ind)
        open_t = _window_open(ind, from)
        isfinite(open_t) || return nothing
        close_t = min(_window_close(ind, until),
            _intervention_removal_time(ind, interventions))
        (isfinite(close_t) && close_t <= open_t) && return nothing
        _pool_heap_push!(open_heap, (open_t, ind.id))
        isfinite(close_t) && _pool_heap_push!(close_heap, (close_t, ind.id))
        return nothing
    end

    # Split the pool into index cases and susceptibles by a random shuffle.
    order = shuffle(rng, collect(members))
    for k in 1:n_initial
        ind = state.individuals[order[k]]
        stamp!(ind, zero(T), 0)
        push_windows!(ind)
    end

    # The distinct susceptible types are the groups. Assign each a group index on
    # first appearance and keep `types[g]` = the type value of group `g`.
    # Susceptibles carry an Exponential(1) resistance threshold; within each group
    # consume thresholds in ascending order (a sorted vector with a front
    # pointer), so each group tracks its own next crossing.
    type_group = Dict{Any, Int}()    # type value → group index
    types = Any[]                       # group index → type value
    sus_by_group = Vector{Int}[]
    Q_by_group = Vector{Float64}[]
    for id in @view order[(n_initial + 1):end]
        tp = typ[id]
        g = get(type_group, tp, 0)
        if g == 0
            push!(types, tp)
            push!(sus_by_group, Int[])
            push!(Q_by_group, Float64[])
            g = length(types)
            type_group[tp] = g
        end
        push!(sus_by_group[g], id)
        push!(Q_by_group[g], rand(rng, Exponential(1.0)))
    end
    G = length(types)                   # number of susceptible groups in play
    for g in 1:G
        perm = sortperm(Q_by_group[g])
        sus_by_group[g] = sus_by_group[g][perm]
        Q_by_group[g] = Q_by_group[g][perm]
    end
    ptr = ones(Int, G)                  # front pointer into each group's queue
    nsus = [length(v) for v in sus_by_group]
    Λ = zeros(T, G)                     # accumulated pressure per group
    λ = zeros(T, G)                     # current force per group (scratch)

    t = zero(T)

    while true
        # Force on a susceptible in each group at the current infectious counts.
        # Piecewise-constant between events, so it is evaluated once per group.
        @inbounds for g in 1:G
            λ[g] = ptr[g] <= nsus[g] ? convert(T, force(types[g], counts)) : zero(T)
        end

        # Next infection per group: the lowest remaining threshold in the group,
        # reached at the group's own force. The soonest across groups wins.
        t_inf = T(Inf)
        gstar = 0
        @inbounds for g in 1:G
            (λ[g] > 0 && ptr[g] <= nsus[g]) || continue
            tg = t + (Q_by_group[g][ptr[g]] - Λ[g]) / λ[g]
            if tg < t_inf
                t_inf = tg
                gstar = g
            end
        end

        t_open, _ = _pool_heap_peek(open_heap)
        t_close, _ = _pool_heap_peek(close_heap)

        t_event = min(t_open, t_close, t_inf)
        isfinite(t_event) || break

        # Advance every group's pressure over [t, t_event] at the force that held
        # during the interval. The min event includes each group's next crossing,
        # so Λ[g] never overshoots its next threshold.
        dt = t_event - t
        @inbounds for g in 1:G
            Λ[g] += λ[g] * dt
        end
        t = t_event

        # Ties are resolved close → open → infection, so an equal-time window-open
        # (e.g. a case that becomes infectious at its infection time when
        # `from == :infection`) updates `counts` before the next infection is timed.
        if t_close == t_event
            _, id = _pool_heap_pop!(close_heap)
            i = slot[id]
            lastid = infectious_ids[end]
            infectious_ids[i] = lastid
            slot[lastid] = i
            pop!(infectious_ids)
            delete!(slot, id)
            counts[typ[id]] -= 1
        elseif t_open == t_event
            _, id = _pool_heap_pop!(open_heap)
            push!(infectious_ids, id)
            slot[id] = length(infectious_ids)
            counts[typ[id]] += 1
        else
            # Infection: the lowest-threshold susceptible in group `gstar` crosses
            # now. Its infector is drawn uniformly from all currently-infectious
            # individuals across groups — a valid parent label whose epidemic
            # dynamics are exact regardless; mixing-weighted attribution (weighting
            # by each infective's contribution to `force`) is a possible refinement
            # that would only sharpen the parent label, not the dynamics.
            id = sus_by_group[gstar][ptr[gstar]]
            ptr[gstar] += 1
            # If the infectious pool is empty — a custom `force` with a positive
            # count-independent hazard, e.g. external importation — there is no
            # infector to attribute to, so fall back to the index-case label 0.
            src = isempty(infectious_ids) ? 0 :
                  infectious_ids[rand(rng, 1:length(infectious_ids))]
            ind = state.individuals[id]
            stamp!(ind, t, src)
            push_windows!(ind)
        end
    end

    return nothing
end
