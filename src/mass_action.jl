# ── Mass action on the race ──────────────────────────────────────────
#
# Under mass action every infectious case meets every other member of a closed
# population, each pair at a constant rate set by the two members' mixing
# groups, so a pair's contact interval is exponential at that rate. That is a
# clique of the race, and it runs there unchanged, but asking a route for every
# pair costs a draw per pair on every opening: O(N) per case. Only the pairs
# whose first contact falls inside the infector's window lead to a proposal, and
# with an exponential interval each pair does so independently, with probability
# `1 - exp(-r·L)` over a window of length `L`, so a mass-action route samples
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
    _MassAction(state, members; mixing_by = (), rate, calendar = nothing, from)

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

`calendar`, a [`Steps`](@ref) or [`Seasonal`](@ref) schedule or a callable
`t -> multiplier`, scales every pair's rate by the schedule's value on the
calendar day, exactly as a [`PairKernel`](@ref)'s `calendar` does. `from`
names the state each member's window opens at (`:infection` unless the model
gives it a later one), read back when a blocked pair's next contact needs the
calendar day its window opened on. A `calendar` with a declared
[`EpiBranch.calendar_bound`](@ref) (`Steps` and `Seasonal` both have one)
folds into the bound described below, keeping the cost per contact; a plain
callable has none, so with it in force every pair in a group is drawn
directly, at O(N) per case, the cost an unbounded window already incurs.

Each group's largest susceptibility is also read when the route is built, as the
bound its proposals are sampled at. A member's susceptibility may fall during the
run but not rise above that bound, which is how the race already reads it: as set
when the member is created. The route also records what it left past the run's
time limit, so build one for each run.
"""
struct _MassAction{T <: Real, S <: Real, U <: Real, C}
    groups::Vector{Vector{Int}}  # member ids in each mixing group
    group_of::Dict{Int, Int}     # member id → index of its group
    rate::Matrix{T}              # rate[a, b]: one infective in a to one member of b
    max_susceptibility::Vector{S}  # largest susceptibility in each group
    # cut[a, b]: an unbounded window of an infective in a was sampled only up to
    # the run's time limit, while it could still reach b.
    cut::Matrix{Bool}
    # Reused across every `propose_along` call on this route, which otherwise
    # runs once per infectious window, since one race never calls it
    # concurrently with itself; reuse spares an allocation on each call.
    contacts::Vector{Tuple{Int, U}}
    calendar::C                  # `nothing`, or a schedule scaling every rate
    from::Symbol                 # the state each member's window opens at
end

function _MassAction(
        state::SimulationState, members; mixing_by::Tuple = (), rate,
        calendar = nothing, from::Symbol = :infection
    )
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
    return _MassAction(
        groups, group_of, float.(rates), max_susceptibility,
        zeros(Bool, length(groups), length(groups)),
        Tuple{Int, _timetype(state)}[], calendar, from
    )
end

# The route as any other: every susceptible member with its pair kernel. This
# is for anything that walks a route's targets; the race itself proposes
# through `propose_along` below and asks for single pairs through
# `route_pair_kernel`.
function (m::_MassAction)(infector_id, state)
    a = m.group_of[infector_id]
    return (
        (id, Exponential(inv(m.rate[a, b])))
            for b in eachindex(m.groups) if m.rate[a, b] > 0
            for id in m.groups[b]
            if id != infector_id && !is_infected(state.individuals[id])
    )
end

proposes_own_contacts(::_MassAction) = true

function route_pair_kernel(m::_MassAction, infector_id, target_id, state)
    r = m.rate[m.group_of[infector_id], m.group_of[target_id]]
    r > 0 || return nothing
    kernel = Exponential(inv(r))
    m.calendar === nothing && return kernel
    # The race scales this kernel's redraw by elapsed time since the window
    # opened (see `_next_contact`), so the opening it needs is this infector's
    # own, recovered from `from` exactly as it was then.
    opening = _window_open(state.individuals[infector_id], m.from)
    return _CalendarScaledKernel(kernel, m.calendar, opening)
end

function propose_along(m::_MassAction, infector_id, state, open_t, close_t, capped, traits)
    infector = state.individuals[infector_id]
    a = m.group_of[infector_id]
    window = close_t - open_t
    contacts = empty!(m.contacts)
    for (b, group) in enumerate(m.groups)
        r = traits ? m.rate[a, b] * infector.infectiousness : m.rate[a, b]
        bound = traits ? r * m.max_susceptibility[b] : r
        bound > 0 || continue
        # A window that never closes would have this case contact every
        # member. A contact after the run's time limit is never used, so with
        # one in force the window is sampled only up to it, and the route
        # remembers that contacts were left beyond it (`pending_beyond`). A
        # window that closes on its own is sampled whole, which keeps a run
        # cut at the time limit the same draw as the full run.
        capped && (m.cut[a, b] = true)
        window > 0 || continue
        _sample_contacts!(
            contacts, state, group, r, bound, traits, open_t, window, m.calendar
        )
    end
    return contacts
end

# Every pair in `group`, at rate `r` scaled by the member's susceptibility when
# `traits` is set, sampled by the pairs with a contact in the window. A
# `calendar` schedule with no `calendar_bound` (a plain function of time)
# cannot be bounded in advance, so with one in force every pair has a contact
# to try and each draws its time directly, at O(N) per case; the geometric-skip
# sampling below, which needs a bound on the rate, does not apply to it. An
# unbounded window with no calendar is drawn the same way, since every pair is
# certain to contact eventually.
#
# Otherwise pairs are selected at a `bound` rate (`bound` itself, or `bound`
# scaled by a `calendar`'s own bound where one is declared): the gap to the next
# pair with a contact is geometric with the envelope's per-pair probability
# `p`. A pair at its own, lower probability `pk` is kept with probability
# `pk/p`, and its time is drawn from its own hazard conditioned on `dt ≤
# window`. With no calendar that hazard is the plain exponential at `rate`;
# with a bounded calendar it is the pair's own `_CalendarScaledKernel`, so a
# `Steps` or `Seasonal` schedule keeps the per-contact cost a fixed rate
# allows.
function _sample_contacts!(
        contacts, state, group, r, bound, traits, open_t, window, calendar
    )
    rng = state.rng
    pair_rate(id) = traits ? r * state.individuals[id].susceptibility : r
    check_bound(rate) = rate <= bound || throw(
        ArgumentError(
            "a member's susceptibility rose above its group's largest when " *
                "the mass-action route was built; set it when the member is created"
        )
    )
    cal_bound = calendar === nothing ? nothing : calendar_bound(calendar)
    if calendar !== nothing && cal_bound === nothing
        for id in group
            rate = pair_rate(id)
            rate > 0 || continue
            dt = rand(rng, _CalendarScaledKernel(Exponential(inv(rate)), calendar, open_t))
            isfinite(dt) && dt <= window && push!(contacts, (id, open_t + dt))
        end
        return nothing
    end
    if calendar === nothing && !isfinite(window)
        for id in group
            rate = pair_rate(id)
            rate > 0 || continue
            push!(contacts, (id, open_t + randexp(rng) / rate))
        end
        return nothing
    end
    envelope = cal_bound === nothing ? bound : bound * cal_bound
    p = -expm1(-envelope * window)
    logq = log1p(-p)
    n = length(group)
    i = 0
    while true
        gap = log1p(-rand(rng)) / logq
        gap < n - i || break
        i += 1 + floor(Int, gap)
        id = group[i]
        if cal_bound === nothing
            rate, pk = bound, p
            if traits
                rate = pair_rate(id)
                check_bound(rate)
                pk = -expm1(-rate * window)
                rand(rng) * p < pk || continue
            end
            dt = -log1p(-rand(rng) * pk) / rate
        else
            rate = traits ? pair_rate(id) : r
            traits && check_bound(rate)
            rate > 0 || continue
            kernel = _CalendarScaledKernel(Exponential(inv(rate)), calendar, open_t)
            pk = -expm1(-cumhazard(kernel, window))
            rand(rng) * p < pk || continue
            dt = _time_at_log_survival(kernel, log1p(-rand(rng) * pk))
        end
        push!(contacts, (id, open_t + dt))
    end
    return nothing
end

# Redraw the group pairs `propose_along` left capped (`m.cut`) from
# `old_max_time` over the extended window: each still-infectious member with
# an opening on this route whose window never closes (`close_t == Inf`) gets
# one fresh call to `propose_along`, over `[old_max_time, Inf)`, exactly as
# though their window had just opened there. The contact interval is
# exponential and so memoryless, which is what makes this the same draw as
# sampling the whole window at once would have been; a member whose window
# did close before the race stopped was never capped and needs no redraw.
# Only one redraw per infector: several resumes can each leave their own
# now-superseded opening recorded for the same member, and resampling for
# each would draw that member's remaining contacts more than once.
_resumable_route(::_MassAction) = true
function _resume_route!(m::_MassAction, race, ri, old_max_time, traits)
    any(m.cut) || return nothing
    (; buffers, members) = race
    T = typeof(old_max_time)
    seen = Set{Int}()
    for idx in 1:length(buffers.openings)
        opening = buffers.openings[idx]
        (opening.route == ri && isinf(opening.close_t)) || continue
        infector_id = opening.infector
        (infector_id == 0 || infector_id in seen) && continue
        push!(seen, infector_id)
        opening_id = _store_opening!(
            members, buffers.openings, infector_id, ri, old_max_time, T(Inf)
        )
        _propose_along!(
            m, race, infector_id, opening_id, old_max_time, T(Inf), traits, false, ri
        )
    end
    return nothing
end

# Whether a window sampled only up to the time limit could still have reached a
# member left uninfected, in which case the run was cut off rather than extinct.
function pending_beyond(m::_MassAction, state, reinfects::Bool)
    any(m.cut) || return false
    return any(CartesianIndices(m.cut)) do ab
        a, b = Tuple(ab)
        m.cut[a, b] && any(m.groups[b]) do id
            ind = state.individuals[id]
            (!is_infected(ind) || (reinfects && isfinite(susceptible_again_time(ind)))) &&
                ind.susceptibility > 0
        end
    end
end
