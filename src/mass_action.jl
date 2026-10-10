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
# Per-individual traits give each pair its own rate, the group's rate scaled by
# the infector's infectiousness and the member's susceptibility. The route then
# samples at a bound on those rates, the group's largest susceptibility, and keeps
# each selected pair with the ratio of its own contact probability to the
# bound's, which leaves exactly the pairs with a contact at their own rate, still
# at a cost per contact.
#
# Everything else is the race's: blocked contacts redraw on the pair (the pair
# kernel below), interventions and risks resolve when a proposal is popped, and
# the infector of a case is the member whose contact reached it first.

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

Each group's largest susceptibility is also read when the route is built, as the
bound its proposals are sampled at. A member's susceptibility may fall during the
run but not rise above that bound, which is how the race already reads it: as set
when the member is created.
"""
struct _MassAction{T <: Real, S <: Real}
    groups::Vector{Vector{Int}}  # member ids in each mixing group
    group_of::Dict{Int, Int}     # member id → index of its group
    rate::Matrix{T}              # rate[a, b]: one infective in a to one member of b
    max_susceptibility::Vector{S}  # largest susceptibility in each group
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
    max_susceptibility = [
        maximum(id -> state.individuals[id].susceptibility, group) for group in groups
    ]
    return _MassAction(groups, group_of, float.(rates), max_susceptibility)
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
        r = traits ? m.rate[a, b] * infector.infectiousness : m.rate[a, b]
        bound = traits ? r * m.max_susceptibility[b] : r
        bound > 0 || continue
        _propose_sampled!(
            race, group, infector_id, r, bound, traits, opening_id, open_t, window
        )
    end
    return nothing
end

# Every pair in `group`, at rate `r` scaled by the member's susceptibility when
# `traits` is set, sampled by the pairs with a contact in the window. With an
# unbounded window every pair has one, so each draws its time directly.
# Otherwise pairs are selected at the `bound` rate: the gap to the next pair with
# a contact is geometric with the bound's per-pair probability `p`. A pair at a
# lower rate, with probability `pk`, is kept with probability `pk/p`, and its
# time is the exponential at its own rate conditioned on `dt ≤ window`,
# `-log(1 - u·pk)/rate`.
function _propose_sampled!(
        race, group, infector_id, r, bound, traits, opening_id, open_t, window
    )
    (; rng, state) = race
    pair_rate(id) = traits ? r * state.individuals[id].susceptibility : r
    if !isfinite(window)
        for id in group
            rate = pair_rate(id)
            rate > 0 || continue
            _propose_contact!(race, id, infector_id, opening_id, open_t + randexp(rng) / rate)
        end
        return nothing
    end
    p = -expm1(-bound * window)
    logq = log1p(-p)
    n = length(group)
    i = 0
    while true
        gap = log1p(-rand(rng)) / logq
        gap < n - i || break
        i += 1 + floor(Int, gap)
        id = group[i]
        rate, pk = bound, p
        if traits
            rate = pair_rate(id)
            rate <= bound || throw(
                ArgumentError(
                    "a member's susceptibility rose above its group's largest when " *
                        "the mass-action route was built; set it when the member is created"
                )
            )
            pk = -expm1(-rate * window)
            rand(rng) * p < pk || continue
        end
        dt = -log1p(-rand(rng) * pk) / rate
        _propose_contact!(race, id, infector_id, opening_id, open_t + dt)
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
