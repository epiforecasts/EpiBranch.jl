# ── Sellke fixed-size population pool ────────────────────────────────
#
# A closed, homogeneously-mixing population of N pre-allocated individuals.
# Each susceptible carries a fixed resistance threshold `Q_j ~ Exponential(1)`;
# every currently-infectious individual exerts force of infection at rate
# `β/N` on each susceptible, so the cumulative pressure a susceptible has
# received is the same value `Λ(t) = (β/N)·∫ Y(s) ds` for all of them, with
# `Y` the number currently infectious. A susceptible is infected the instant
# `Λ(t)` crosses its threshold. This is the Sellke threshold construction; it
# reproduces the exact stochastic SIR final-size law, with `R0 = β·E[infectious
# period]`.
#
# The integration is event-driven so that infection *times* — not just the
# final size — fall out exactly: between events `Λ` grows linearly at rate
# `(β/N)·Y`, so the next threshold crossing has a closed-form time, and the
# three candidate events (next infection, next window-open, next window-close)
# are raced in O(N log N). Every other concern is the shared engine: each
# infected member's natural history is stamped through `resolve_transitions!`,
# so interventions act only by shortening the infectious window through
# `_window_close`, exactly as in `_sellke_race!`.

# A minimal binary min-heap over `(time, id)` pairs — tuples compare
# lexicographically, so ties break deterministically on id. Avoids a
# DataStructures dependency for the two pending-event queues.
function _pool_heap_push!(h::Vector{Tuple{Float64, Int}}, x::Tuple{Float64, Int})
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

_pool_heap_peek(h::Vector{Tuple{Float64, Int}}) = isempty(h) ? (Inf, 0) : @inbounds h[1]

function _pool_heap_pop!(h::Vector{Tuple{Float64, Int}})
    n = length(h)
    n == 0 && return (Inf, 0)
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
    _sellke_pool!(state, members, rng; beta, n_initial, from, until)

Run the Sellke threshold construction over the `members` (global ids) of a
closed, homogeneously-mixing population, infecting individuals in continuous
time and stamping each case's natural history. `beta` is the transmission
rate (`R0 = beta·E[infectious period]`), `n_initial` the number of index cases
seeded at time 0, `from` the state the infectious window opens at and `until`
the removal states that close it. Writes per-individual state directly; the
caller reconciles aggregate bookkeeping and applies observation.
"""
function _sellke_pool!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; beta::Real, n_initial::Integer, from::Symbol, until::Tuple)
    N = length(members)
    N == 0 && return nothing
    βN = Float64(beta) / N

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
        return nothing
    end

    open_heap = Tuple{Float64, Int}[]   # pending window-open (becomes infectious)
    close_heap = Tuple{Float64, Int}[]  # pending window-close (recovers/isolates)
    infectious_ids = Int[]              # ids currently infectious (window open)
    slot = Dict{Int, Int}()             # id → its index in `infectious_ids`

    # Push an infected individual's infectious window onto the queues. A never-
    # infectious case (window-open is Inf) contributes nothing to Λ; a never-
    # closing window (Inf close, e.g. no removal state) simply stays open.
    push_windows! = function (ind)
        open_t = _window_open(ind, from)
        isfinite(open_t) || return nothing
        _pool_heap_push!(open_heap, (open_t, ind.id))
        close_t = _window_close(ind, until)
        isfinite(close_t) && _pool_heap_push!(close_heap, (close_t, ind.id))
        return nothing
    end

    # Split the pool into index cases and susceptibles by a random shuffle.
    order = shuffle(rng, collect(members))
    for k in 1:n_initial
        ind = state.individuals[order[k]]
        stamp!(ind, 0.0, 0)
        push_windows!(ind)
    end

    # Susceptibles carry an Exponential(1) resistance threshold; consumed in
    # ascending order (sorted vector with a front pointer).
    sus_ids = order[(n_initial + 1):end]
    sus_Q = [rand(rng, Exponential(1.0)) for _ in sus_ids]
    perm = sortperm(sus_Q)
    sus_ids = sus_ids[perm]
    sus_Q = sus_Q[perm]
    ptr = 1
    nsus = length(sus_ids)

    t = 0.0
    Λ = 0.0

    while true
        Y = length(infectious_ids)

        t_open, _ = _pool_heap_peek(open_heap)
        t_close, _ = _pool_heap_peek(close_heap)
        # Next threshold crossing: pressure rises at rate βN·Y toward the lowest
        # remaining threshold. No susceptibles or no infectives ⇒ no infection.
        t_inf = (Y > 0 && ptr <= nsus) ? t + (sus_Q[ptr] - Λ) / (βN * Y) : Inf

        t_event = min(t_open, t_close, t_inf)
        isfinite(t_event) || break

        # Accumulate pressure over [t, t_event] at the current infectious count.
        Λ += βN * Y * (t_event - t)
        t = t_event

        # Ties are resolved close → open → infection, so an equal-time window-open
        # (e.g. a case that becomes infectious at its infection time when
        # `from == :infection`) increments Y before the next infection is timed.
        if t_close == t_event
            _, id = _pool_heap_pop!(close_heap)
            i = slot[id]
            lastid = infectious_ids[end]
            infectious_ids[i] = lastid
            slot[lastid] = i
            pop!(infectious_ids)
            delete!(slot, id)
        elseif t_open == t_event
            _, id = _pool_heap_pop!(open_heap)
            push!(infectious_ids, id)
            slot[id] = length(infectious_ids)
        else
            # Infection: the lowest-threshold susceptible crosses now. Its
            # infector is drawn uniformly from the currently-open infectives.
            id = sus_ids[ptr]
            ptr += 1
            src = infectious_ids[rand(rng, 1:Y)]
            ind = state.individuals[id]
            stamp!(ind, t, src)
            push_windows!(ind)
        end
    end

    return nothing
end
