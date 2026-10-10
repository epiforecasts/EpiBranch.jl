# ── Mass action on the race ──────────────────────────────────────────
#
# Under mass action every infectious case meets every other member of a closed
# population, each pair at a constant rate set by the two members' mixing
# groups, so a pair's contact interval is exponential at that rate. That is a
# clique of the race, and it runs there unchanged, but asking a route for every
# pair costs a draw per pair on every opening: O(N) per case. Only the pairs
# whose first contact falls inside the infector's window lead to a proposal, and
# with an exponential interval each pair does so independently, with probability
# `1 - exp(-r·L)` over a window of length `L`. So a mass-action route samples
# just those pairs, by geometric skips through each group, and draws each one's
# contact time from the exponential conditioned on falling inside the window.
# That is the same process as drawing every pair and discarding the late ones,
# at a cost per contact rather than per pair.
#
# Everything else is the race's: blocked contacts redraw on the pair (the pair
# kernel below), per-individual traits scale the pair's rate, interventions and
# risks resolve when a proposal is popped, and the infector of a case is the
# member whose contact reached it first.

"""
    _MassAction(state, members; mixing_by = (), rate)

Mass-action contacts among `members` (global ids) of `state`, as the `targets`
of a `_sellke_race!` route.

A member's mixing group is the tuple of its `mixing_by` attribute values, read
off its state when the route is built (`missing` for any it lacks), so with
`mixing_by = ()` everyone shares the group `()`. `rate(infector_group,
target_group)` is the contact rate from one infective in the first group to
one member of the second: `β/N` for homogeneous mixing, or `M[b, a]/n[b]` for a
contact matrix `M[b, a]` giving the rate at which one infective in band `a`
contacts the `n[b]` members of band `b` between them. The rate may be any real type,
a dual under automatic differentiation included, and is fixed for the run.
"""
struct _MassAction{T <: Real}
    groups::Vector{Vector{Int}}  # member ids in each mixing group
    group_of::Dict{Int, Int}     # member id → index of its group
    rate::Matrix{T}              # rate[a, b]: one infective in a to one member of b
end

function _MassAction(state::SimulationState, members; mixing_by::Tuple = (), rate)
    keys = Any[]
    index = Dict{Any, Int}()
    groups = Vector{Int}[]
    group_of = Dict{Int, Int}()
    for id in members
        record = state.individuals[id].state
        key = Tuple(get(record, k, missing) for k in mixing_by)
        g = get!(index, key) do
            push!(keys, key)
            push!(groups, Int[])
            length(keys)
        end
        push!(groups[g], id)
        group_of[id] = g
    end
    rates = [rate(keys[a], keys[b]) for a in eachindex(keys), b in eachindex(keys)]
    all(r -> isfinite(r) && r >= 0, rates) || throw(
        ArgumentError("mass-action contact rates must be finite and non-negative")
    )
    return _MassAction(groups, group_of, float.(rates))
end

# The route as any other: every susceptible member with its pair kernel. The race
# itself proposes through `_propose_along!` below and asks for single pairs through
# `_pair_kernel`; this is for anything that walks a route's targets.
function (m::_MassAction)(infector_id, state)
    a = m.group_of[infector_id]
    return (
        (id, Exponential(inv(m.rate[a, b])))
            for b in eachindex(m.groups) if m.rate[a, b] > 0
            for id in m.groups[b]
            if id != infector_id && !is_infected(state.individuals[id])
    )
end

function _pair_kernel(m::_MassAction, infector_id, target_id, state)
    r = m.rate[m.group_of[infector_id], m.group_of[target_id]]
    return r > 0 ? Exponential(inv(r)) : nothing
end

function _propose_along!(
        m::_MassAction, race, infector_id, opening_id, open_t, close_t, traits, live
    )
    live && throw(
        ArgumentError("a mass-action route reads no host records, so none can be watched")
    )
    infector = race.state.individuals[infector_id]
    a = m.group_of[infector_id]
    window = close_t - open_t
    window > 0 || return nothing
    for (b, group) in enumerate(m.groups)
        r = m.rate[a, b]
        r > 0 || continue
        if traits
            _propose_scaled!(race, group, infector, r, opening_id, open_t, window)
        else
            _propose_sampled!(race, group, infector_id, r, opening_id, open_t, window)
        end
    end
    return nothing
end

# Every pair in `group` at rate `r`, sampled by the pairs with a contact in the
# window. With an unbounded window every pair has one, so each draws its time
# directly. Otherwise the gap to the next pair that does is geometric with the
# per-pair probability `p`, and its time is the exponential conditioned on
# `dt ≤ window`, `-log(1 - u·p)/r`.
function _propose_sampled!(race, group, infector_id, r, opening_id, open_t, window)
    (; rng) = race
    if !isfinite(window)
        for id in group
            _propose_contact!(race, id, infector_id, opening_id, open_t + randexp(rng) / r)
        end
        return nothing
    end
    p = -expm1(-r * window)
    logq = log1p(-p)
    n = length(group)
    i = 0
    while true
        gap = log1p(-rand(rng)) / logq
        gap < n - i || break
        i += 1 + floor(Int, gap)
        dt = -log1p(-rand(rng) * p) / r
        _propose_contact!(race, group[i], infector_id, opening_id, open_t + dt)
    end
    return nothing
end

# With per-individual traits each pair has its own rate, so every pair in the
# group draws, as the race's default route does.
function _propose_scaled!(race, group, infector, r, opening_id, open_t, window)
    (; state, rng) = race
    for id in group
        rate = r * infector.infectiousness * state.individuals[id].susceptibility
        rate > 0 || continue
        dt = randexp(rng) / rate
        dt <= window && _propose_contact!(race, id, infector.id, opening_id, open_t + dt)
    end
    return nothing
end

# A contact with a member who is the infector or already infected is wasted, as
# it would be under any route.
function _propose_contact!(race, id, infector_id, opening_id, t)
    id == infector_id && return nothing
    k = race.pos[id]
    race.processed[k] && return nothing
    return _propose!(race, k, opening_id, t)
end
