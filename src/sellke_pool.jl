# ── Sellke fixed-size population pool ────────────────────────────────
#
# A closed population of N pre-allocated individuals, bucketed by mixing type.
# Each susceptible carries a resistance threshold `Q_j ~ Exponential(1)` and
# accumulates infection pressure `Λ(t) = ∫ λ(s) ds` at the force of infection
# `λ` felt by its own mixing type; it is contacted the instant `Λ` crosses its
# threshold, and infected unless a competing risk blocks that contact. A model
# names *which real attributes define mixing*
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
#
# ── Per-contact competing risks ──────────────────────────────────────
#
# A threshold crossing is the arrival of one infectious contact, and like any
# other potential transmission it is put to the composed competing risks (see
# `_proposal_blocked`): whatever block the model's own risk sources or an
# intervention contributes against that pair at that time. The two
# per-individual traits are not among them here: they are rate multipliers, and
# the construction carries them already — infectiousness as each infective's
# weight in the counts the force reads, susceptibility as the scaling of each
# susceptible's threshold.
#
# A blocked contact does not infect, and the susceptible goes back into its
# group with a fresh `Exponential(1)` resistance above the pressure it has
# already absorbed. Mass action offers a *stream* of contacts, not one per pair:
# the pressure keeps arriving whether or not any given contact transmits, so
# surviving one contact says nothing about the next. The exponential resistance
# is memoryless, so re-drawing it is exactly the residual of the same process,
# and blocking a fraction `p` of contacts thins the force of infection to
# `(1-p)·λ` — the continuous-time reading of a leaky vaccine, and the same
# realised reduction in cases as blocking a fraction `p` of a parent's contacts
# on the generation engine.
#
# `_sellke_race!` reads a block the same way on an edge of a graph, where the
# pair's next contact is drawn from its kernel conditioned on falling later. Both
# thin the hazard by the same factor, so a clique of the race and a pool with the
# matching force of infection are the same process.

# The infector of a pool infection, drawn from `infectious_ids` with probability
# proportional to infectiousness. When every member has the default
# infectiousness the draw is uniform, which keeps seeded runs without the trait
# unchanged. With no infectious weight at all — an empty pool under a custom
# `force` with a count-independent hazard, e.g. external importation, or only
# zero-infectiousness cases — there is no infector to attribute to, so fall back
# to the index-case label 0.
function _draw_infector(rng::AbstractRNG, state::SimulationState,
        infectious_ids::AbstractVector{Int}, equal_infectiousness::Bool)
    isempty(infectious_ids) && return 0
    equal_infectiousness && return infectious_ids[rand(rng, 1:length(infectious_ids))]
    u = rand(rng) * sum(id -> state.individuals[id].infectiousness, infectious_ids)
    # The last positive-weight case stands in when rounding leaves `u` a hair
    # above zero after the final subtraction; with no positive weight at all it
    # stays 0.
    infector = 0
    for id in infectious_ids
        weight = state.individuals[id].infectiousness
        weight > 0 || continue
        infector = id
        u -= weight
        u < 0 && break
    end
    return infector
end

# ── Infector-side risks under structured mixing ──────────────────────
#
# A contact's infector is drawn in proportion to infectiousness, which is a
# uniform draw while every infective is at the default. That is exact for one
# mixing type, where every infective of the same infectiousness adds the same to
# the force on every susceptible. With several types an infective adds to
# a group's force according
# to its own type, so the exact draw weights each infective by its contribution.
# `force` is an arbitrary function of the per-type counts and is not required to
# be linear in them, so that contribution is not defined in general; recovering
# it by differencing `force` would assume linearity, cost one call per infectious
# type on every contact, and change the random stream and parent labels of
# structured pools that carry no risks at all. The infectiousness-weighted draw
# is kept, and it leaves the dynamics exact as long as no risk depends on who the
# infector is:
# the contact's susceptibility or a vaccine's protection of the contact are
# fine, whereas a leaky isolation would weight blocks by the wrong infectors.
# Those are refused. Per-individual infectiousness is not among them: it is
# carried by the force itself, as each infective's weight in the counts, so it
# needs no attribution to be exact.

# Whether a risk source can block a contact differently depending on its
# infector. A model's own risk source is opaque, so it is assumed to; an
# intervention without a `competing_risk` method of its own contributes no risk.
_blocks_by_infector(source) = true
function _blocks_by_infector(iv::AbstractIntervention)
    _has_own_method(competing_risk, typeof(iv), AbstractIntervention)
end
# Perfect isolation's block starts when the infector's window closes, so the
# infector is never drawn once it could apply; only a leaky residual can bite.
_blocks_by_infector(iso::Isolation) = iso.post_isolation_transmission > 0
# A vaccination's own risk protects the contact and reads the infector only for
# a ring's onward effect — but that holds for the ones this package writes. A
# subtype of its own is taken to read the infector, as any other intervention is.
function _blocks_by_infector(v::AbstractVaccination)
    _has_own_method(competing_risk, typeof(v), AbstractVaccination)
end
_blocks_by_infector(rv::RingVaccination) = rv.onward_efficacy > 0
_blocks_by_infector(s::Scheduled) = _blocks_by_infector(s.intervention)

function _refuse_infector_side_risks(state, members, risks, interventions)
    culprits = String[]
    for source in (risks..., interventions...)
        _blocks_by_infector(source) || continue
        push!(culprits, string(nameof(typeof(_unwrap_scheduled(source)))))
    end
    isempty(culprits) && return nothing
    throw(ArgumentError(
        "the fixed-size pool with more than one mixing type draws each contact's " *
        "infector in proportion to infectiousness rather than by its share of " *
        "the force, which gives the right " *
        "dynamics only while the infector cannot change whether a contact " *
        "transmits. These can: $(join(unique(culprits), ", ")). Fold differences " *
        "in infectiousness between types into `force`, or run a single mixing " *
        "type (`mixing_by = ()`). Risks that act on the contact alone, such as " *
        "susceptibility, are supported."))
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
    mixing type to the infectiousness-weighted number currently infectious of
    that type (each infective contributes its own `infectiousness`, 1 by
    default, rather than a flat 1). It must be piecewise-constant
    between events, which it is: `counts` only changes at an infection, a window
    opening or a window closing. Homogeneous mixing is
    `force = (type, counts) -> beta / N * sum(values(counts))`.

`n_initial` is the number of index cases seeded at time 0, `from` the state the
infectious window opens at and `until` the removal states that close it. Each
susceptible carries an `Exponential(1)` resistance threshold and is contacted the
instant its accumulated pressure, scaled by its own `susceptibility` (1 by
default), crosses it; pressure accumulates at the force felt by its mixing type. Writes per-individual state directly; the caller
reconciles aggregate bookkeeping and applies observation.

Each contact is put to the composed competing risks — the model's own `risks`
(what [`transmission_risks`](@ref) reports) and the interventions; the two
per-individual traits are already in the construction, as the weights in
`counts` and the scaling of each threshold. A blocked contact
does not infect, and the susceptible draws a fresh resistance and waits for the
next one. With more than one mixing type, a risk that depends on the infector —
a leaky isolation, a model risk source, or any other intervention defining its
own `competing_risk` apart from those known to act on the contact alone — is
refused with an `ArgumentError`, because the infector a contact is attributed to
is not weighted by its share of the force. Per-individual infectiousness is not
refused: it reaches the force through the weighted `counts`.
"""
function _sellke_pool!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; mixing_by::Tuple = (), force, n_initial::Integer,
        from::Symbol, until::Tuple, interventions = (), risks = ())
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
        # A pre-created member has no infection time, and so no onset, until
        # now. Derive the onset from the infection time before transitions and
        # interventions read it.
        _set_onset_from_incubation!(ind)
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

    # Current infectiousness-weighted infectious count per mixing type: one
    # persistent Dict, mutated in place and handed to `force` (never
    # reallocated). Every type present starts at 0, so index cases and
    # susceptibles of any type key in without a miss. Weighting by each
    # infective's own `infectiousness` (1 by default, carrying `T` like every
    # other pressure term) is the pool's reading of that trait: a case with half
    # the infectiousness contributes half the force a default case would.
    counts = Dict{Any, T}()
    # Headcount per mixing type. Adding and then subtracting fractional
    # infectiousness leaves a rounding residual, which as a positive force would
    # infect the remaining susceptibles at absurd times once nobody is
    # infectious, so a type's weighted count is reset to zero exactly when its
    # headcount is.
    n_infectious = Dict{Any, Int}()
    for id in members
        counts[typ[id]] = zero(T)
        n_infectious[typ[id]] = 0
    end

    # Every infective contributing the same weight makes the infector draw
    # uniform, which is what a run without the trait did before it was honoured.
    equal_infectiousness = all(id -> state.individuals[id].infectiousness == 1, members)

    route = _shorthand_window(from, until)
    risk_interventions = filter(iv -> risk_applies(iv, route), interventions)
    length(counts) > 1 &&
        _refuse_infector_side_risks(state, members, risks, risk_interventions)

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
        _heap_push!(open_heap, (open_t, ind.id))
        isfinite(close_t) && _heap_push!(close_heap, (close_t, ind.id))
        return nothing
    end

    # Split the pool into index cases and susceptibles by a random shuffle.
    order = shuffle(rng, collect(members))
    for k in 1:n_initial
        ind = state.individuals[order[k]]
        stamp!(ind, zero(T), 0)
        equal_infectiousness &= ind.infectiousness == 1
        push_windows!(ind)
    end

    # The distinct susceptible types are the groups. Assign each a group index on
    # first appearance and keep `types[g]` = the type value of group `g`.
    # Susceptibles carry an Exponential(1) resistance threshold; within each group
    # consume thresholds in ascending order (a sorted vector with a front
    # pointer), so each group tracks its own next crossing.
    #
    # A susceptible's own `susceptibility` (1 by default) scales the pressure it
    # feels, `Λ(t)·susceptibility ≥ Q`, so its *effective* threshold is
    # `Q/susceptibility` — the group's shared pressure crosses it later exactly
    # in proportion to how resistant this individual is, which is the trait as a
    # rate multiplier rather than a per-contact block. Dividing once here (and
    # sorting on the result) keeps the rest of the loop, which only ever compares
    # against the group's pressure, unchanged; a susceptibility of 0 is an
    # infinite threshold, never crossed.
    type_group = Dict{Any, Int}()    # type value → group index
    types = Any[]                       # group index → type value
    sus_by_group = Vector{Int}[]
    # Carries `T` (e.g. a dual under AD), not hardcoded `Float64`: the effective
    # threshold is arithmetic on susceptibility, which does too.
    Q_by_group = Vector{T}[]
    for id in @view order[(n_initial + 1):end]
        tp = typ[id]
        g = get(type_group, tp, 0)
        if g == 0
            push!(types, tp)
            push!(sus_by_group, Int[])
            push!(Q_by_group, T[])
            g = length(types)
            type_group[tp] = g
        end
        push!(sus_by_group[g], id)
        susceptibility = state.individuals[id].susceptibility
        push!(Q_by_group[g],
            susceptibility > 0 ? rand(rng, Exponential(1.0)) / susceptibility : T(Inf))
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
    qn = zeros(T, G)                    # next threshold per group (scratch)

    # A susceptible whose contact a competing risk blocks comes back with a fresh
    # resistance, which can land anywhere in the order and so cannot go back into
    # the sorted queue. Each group keeps a min-heap of `(threshold, id)` for those,
    # and its next crossing is the lower of the queue's front and the heap's root.
    # The heaps stay empty for a model with no risks in play, which is why the
    # sorted queue is kept rather than everything moved to a heap: that path walks
    # a pointer, as it did before per-contact risks existed. A re-drawn threshold
    # carries the timing type `T` because it sits above the pressure already
    # absorbed and so inherits that pressure's sensitivity under automatic
    # differentiation, while the drawn ones stay `Float64` constants — the
    # reparameterisation.
    redrawn = [Tuple{T, Int}[] for _ in 1:G]

    t = zero(T)

    while true
        # Each group's next threshold, and the force on a susceptible in it at the
        # current infectious counts. The force is piecewise-constant between
        # events, so it is evaluated once per group.
        @inbounds for g in 1:G
            front = ptr[g] <= nsus[g] ? convert(T, Q_by_group[g][ptr[g]]) : T(Inf)
            qn[g] = isempty(redrawn[g]) ? front : min(front, redrawn[g][1][1])
            λ[g] = isfinite(qn[g]) ? convert(T, force(types[g], counts)) : zero(T)
        end

        # Next contact per group: the lowest remaining threshold in the group,
        # reached at the group's own force. The soonest across groups wins.
        t_inf = T(Inf)
        gstar = 0
        @inbounds for g in 1:G
            (λ[g] > 0 && isfinite(qn[g])) || continue
            tg = t + (qn[g] - Λ[g]) / λ[g]
            if tg < t_inf
                t_inf = tg
                gstar = g
            end
        end

        t_open, _ = _heap_peek(open_heap)
        t_close, _ = _heap_peek(close_heap)

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
            _, id = _heap_pop!(close_heap)
            i = slot[id]
            lastid = infectious_ids[end]
            infectious_ids[i] = lastid
            slot[lastid] = i
            pop!(infectious_ids)
            delete!(slot, id)
            tp = typ[id]
            n_infectious[tp] -= 1
            counts[tp] = n_infectious[tp] == 0 ? zero(T) :
                         counts[tp] - state.individuals[id].infectiousness
        elseif t_open == t_event
            _, id = _heap_pop!(open_heap)
            push!(infectious_ids, id)
            slot[id] = length(infectious_ids)
            n_infectious[typ[id]] += 1
            counts[typ[id]] += state.individuals[id].infectiousness
        else
            # Contact: the lowest-threshold susceptible in group `gstar` is
            # reached now. Take it from whichever holds that threshold — the
            # front of the group's sorted queue, or the root of its re-drawn heap.
            q = qn[gstar]
            local id
            if !isempty(redrawn[gstar]) && redrawn[gstar][1][1] == q
                _, id = _heap_pop!(redrawn[gstar])
            else
                id = sus_by_group[gstar][ptr[gstar]]
                ptr[gstar] += 1
            end
            # Its infector is drawn from all currently-infectious individuals
            # across groups, in proportion to their infectiousness — each one's
            # share of the pooled force. With one mixing type that is the exact
            # attribution. With several it is only a parent label, because it
            # takes no account of which types mix with which, and the dynamics
            # stay exact because no risk the pool resolves reads the infector:
            # those are refused before the run
            # (`_refuse_infector_side_risks`).
            src = _draw_infector(rng, state, infectious_ids, equal_infectiousness)
            ind = state.individuals[id]
            # An introduction with no infector has no pair to resolve risks over,
            # as an index case on the generation engine has none either.
            blocked = src != 0 && _proposal_blocked(
                state, state.individuals[src], ind, t, risks, risk_interventions)
            if blocked
                # With an opaque risk, an immortal infectious source can keep
                # generating rejected contacts forever. Require every active
                # source to have a finite removal time instead of guessing from
                # how many rejections have occurred.
                all(infectious_ids) do source_id
                    source = state.individuals[source_id]
                    isfinite(min(_window_close(source, until),
                        _intervention_removal_time(source, interventions)))
                end || throw(ArgumentError(
                    "repeated contacts after a blocked pool proposal require " *
                    "finite infectious windows. Add a removal transition, or " *
                    "encode static protection in host susceptibility or force."))
                # The contact did not transmit. Put the susceptible back with the
                # residual of its resistance: a fresh Exponential(1) above the
                # threshold this contact consumed.
                # The residual is in the group's pressure, so it is the
                # individual's own fresh `Exponential(1)` over its
                # susceptibility, exactly as its first threshold was.
                _heap_push!(redrawn[gstar],
                    (
                        q +
                        convert(T, rand(rng, Exponential(1.0))) /
                        state.individuals[id].susceptibility,
                        id))
            else
                stamp!(ind, t, src)
                equal_infectiousness &= ind.infectiousness == 1
                push_windows!(ind)
            end
        end
    end

    return nothing
end
