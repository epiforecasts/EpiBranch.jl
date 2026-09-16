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

# ── Interventions on the continuous-time (Sellke) models ─────────────
# These models run their own event loop rather than the generation engine, so
# the engine's per-generation hook passes never fire. Two seams carry an
# intervention's effect instead. The first is the infectious window: an
# intervention that removes a case from onward transmission (isolation)
# shortens it. After a case's natural history is stamped, run each
# intervention's per-individual resolution (so `Isolation` writes its isolation
# time), then close the window at the earliest removal across interventions as
# well as the `until` states. The second is the per-contact competing risk,
# resolved against each candidate infection time the race proposes — see the
# next block.

# Run each intervention's per-individual resolution on a freshly-stamped case.
function _resolve_interventions!(state::SimulationState, ind, interventions)
    isempty(interventions) && return nothing
    # Expose the running simulation clock and case count so a `Scheduled`
    # intervention's `start_time` / `end_time` / `start_after_cases` gate
    # evaluates correctly during the continuous-time loop. Cases are processed in
    # increasing infection time, so this case's infection time is the current
    # clock and it is the next case in sequence; the post-loop reconcile
    # (`_reconcile_sellke_bookkeeping!`) sets the final values.
    state.max_infection_time = ind.infection_time
    state.cumulative_cases += 1
    for iv in interventions
        resolve_individual!(iv, ind, state)
    end
    return nothing
end

# ── Per-contact competing risks on the continuous-time path ──────────
#
# A competing risk blocks one infector → contact transmission with some
# probability, conditional on the risk having arrived before that transmission's
# time. On the race, a potential transmission *is* a drawn time: the contact
# interval from the infector's window opening to the moment it would infect that
# neighbour. So the risks are resolved where that time is proposed and against
# the time itself (see the proposal loop in `_sellke_race!`), exactly as the
# generation engine resolves them against a contact's transmission time.
#
# A block declines the proposal and does nothing else. The pair's candidate time
# is not relaxed, the neighbour keeps whatever candidate it already had, stays
# susceptible to everyone else, and the race carries on. The kernel draw is not
# repeated, because the kernel is that pair's *contact interval* — the time of
# the first infectious contact along the edge, not the first of a stream — so
# there is no second contact to offer. Declining is also what keeps the two
# engines agreeing: on the generation engine a blocked contact is a lost
# transmission and the parent gets no replacement for it, so an efficacy of 0.5
# halves that pair's transmissions on either engine, and a susceptibility of 0
# blocks every one of them.
#
# This does depart from the pure Sellke reading, in which a susceptible draws a
# single resistance and every infector races against that one threshold; here
# each ordered pair is blocked independently. That is the generation engine's
# semantics and it is coherent on a graph, whose edges are already independent —
# but it means a run carrying per-contact risks is no longer the exact
# generative model of the pairwise likelihood, which has no term for a declined
# proposal. A model with risks and a likelihood fit to it will disagree.
#
# An intervention's risks are resolved per route, and only on a route that opted
# into intervention removal — see the proposal loop for why. The model's own
# risks and the per-individual multipliers apply on every route.
#
# Risks are resolved only for a proposal that would otherwise win the race. One
# that is already later than the neighbour's current best could not have
# infected anyone whatever the risks said, and candidate times only ever fall,
# so skipping it changes no outcome and keeps the cost at nothing for a model
# with no risks in play: the built-in sources return no risk at all when every
# multiplier is 1, so nothing is drawn from the rng and a run without risks
# reproduces the same seeded results as before.

# Earliest time any intervention removes `ind` from onward transmission.
function _intervention_removal_time(ind, interventions)
    t = Inf
    for iv in interventions
        t = min(t, infectious_removal_time(iv, ind))
    end
    return t
end

"""
    INTERVENTION_REMOVAL

The reserved state a [`RouteWindow`](@ref) lists in its `until` to be cut by
whatever the composed interventions remove the case at.

Route censoring is otherwise expressed in states the natural history writes, so
a route ends when the case recovers, dies, or is buried. An intervention
removal cannot be read off a state key alone, because whether it removes at all
depends on the intervention: perfect isolation takes a case out of transmission
entirely, whereas leaky isolation only reduces it and an infectious window
cannot express that. `infectious_removal_time` is what resolves this, and this
pseudo-state is how a window opts into it.

Listing it is what makes a route one that control measures can cut. A community
route lists it, so isolating a case ends its community transmission; a
household route does not, so the case goes on infecting the people it lives
with. That difference is the whole reason routes are separated.
"""
const INTERVENTION_REMOVAL = :intervention_removal

# Close a window: the earliest of its `until` states' times, plus the
# intervention removal when the window opted into it.
function _route_close(ind, w::RouteWindow, interventions)
    t = _window_close(ind, w.until)
    if INTERVENTION_REMOVAL in w.until
        t = min(t, _intervention_removal_time(ind, interventions))
    end
    return t
end

# Whether a continuous-time model honours an intervention. Between the two seams
# these models have — the infectious window and the per-contact competing risk —
# an intervention is honoured when its effect is a removal (perfect isolation
# shortens the window), a per-contact block (leaky isolation, a vaccine's
# efficacy), or per-individual state written as each individual is initialised or
# as each case is resolved. That covers most of what an intervention does, which
# is why the default is that one is honoured. `Scheduled` delegates to its
# wrapped intervention — the loop exposes the running clock/count (see
# `_resolve_interventions!`), so its time/count gate is honoured whenever the
# wrapped intervention is.
#
# What has no continuous-time representation is a *generation-shaped* hook.
# `apply_post_transmission!` and `keep_active` act on a batch of freshly created
# contact objects, and a race that settles one pre-existing node at a time never
# builds those. An intervention reaching its targets that way declares itself
# unhonoured here, as `MassVaccination` does: its rollout doses each new contact
# as the generation engine creates it, so on the continuous-time path nobody is
# ever dosed and the efficacy risk it contributes never fires. The model warns
# for the unhonoured ones rather than silently ignoring them.
#
# Tracing needs one thing more: the model has to be able to name the contacts a
# case reached, which is what `supplies_contacts` reports. A graph names a node's
# neighbours and a household its members, but the mass-action pool has no
# pairwise contact structure, so tracing has nothing to act along there and stays
# unhonoured — and ring vaccination, which doses along the trace, with it.
_sellke_honours(model, ::AbstractIntervention) = true
_sellke_honours(model, ::ContactTracing) = supplies_contacts(model)
_sellke_honours(model, ::RingVaccination) = supplies_contacts(model)
_sellke_honours(model, ::MassVaccination) = false
_sellke_honours(model, s::Scheduled) = _sellke_honours(model, s.intervention)

"""
    supplies_contacts(model) -> Bool

Whether a continuous-time model can name the contacts each case reached, so
that [`trace_contacts!`](@ref EpiBranch.trace_contacts!) has something to act
on. True for the structure-driven processes, whose contacts are a node's
neighbours or a household's members; false by default, and in particular for
the mass-action pool, which has no pairwise contact structure. A model that
returns `true` must pass a `contacts` closure to `_sellke_race!`.
"""
supplies_contacts(::TransmissionModel) = false

# Tracing on the continuous-time path. A case's trace time follows from its own
# timeline, so it can only be stamped once the race has finalised that case —
# which is also the point at which its contacts are known. Only contacts that
# are not yet final can be affected: popping a case fixes its infectious window
# and its onward proposals, so a trace arriving later has nothing left to
# shorten. The generation engine behaves the same way, tracing a case's contacts
# and never an earlier generation, so the two paths agree; tracing *backwards*
# to an already-final infector is a separate capability neither engine has.
function _trace_from!(state, infector, interventions, contacts, pos, processed)
    contacts === nothing && return nothing
    any(traces_contacts, interventions) || return nothing
    pending = Individual[]
    not_before = typeof(infector.infection_time)[]
    timed = false
    for c in contacts(infector.id, state)
        cid, t0 = c isa Tuple ? (c[1], c[2]) : (c, -Inf)
        timed |= c isa Tuple
        k = get(pos, cid, 0)
        (k == 0 || processed[k]) && continue
        push!(pending, state.individuals[cid])
        push!(not_before, t0)
    end
    isempty(pending) && return nothing
    for iv in interventions
        if timed
            trace_contacts!(iv, state, infector, pending, not_before)
        else
            trace_contacts!(iv, state, infector, pending)
        end
    end
    return nothing
end

# Warn once (per `simulate` call) when a continuous-time model is handed
# interventions it cannot honour, so the limitation is loud rather than silent.
# Gated on `_honours_termination_controls`, which is `false` for exactly the
# structure-driven models that run their own Sellke loop.
function _warn_unhonoured_interventions(model, interventions)
    _honours_termination_controls(model) && return nothing
    unhonoured = unique(String[string(nameof(typeof(iv)))
                               for iv in interventions if !_sellke_honours(model, iv)])
    isempty(unhonoured) && return nothing
    @warn "$(nameof(typeof(model))) is a continuous-time model that settles one " *
          "pre-existing case at a time, so it never creates the batches of new " *
          "contacts the generation engine's post-transmission hooks act on; it " *
          "does not honour these, which will have no effect: " *
          "$(join(unhonoured, ", ")). Express such control as a removal " *
          "`Transition` in the progression, or through an intervention that acts " *
          "when each individual is initialised, resolved, or traced."
    return nothing
end

"""
    _sellke_race!(state, members, rng; seed!, targets, from, until, routes, risks)

Run the Sellke/Dijkstra continuous-time competing-risks race over the individuals
`members` (global ids). `seed!(best, members, rng)` fills the candidate infection
times `best` (indexed `1:length(members)`) with exogenous introductions or a seed.
`targets(infective_id, state)` yields `(target_id, kernel)` for the contacts an
infective can reach, each with its pairwise contact-interval kernel. Members are
processed in increasing infection time (each pop is final); on processing, the
case's natural history is stamped and it exposes still-susceptible targets with a
`from`-timed contact interval accepted inside its infectious window. Each case's
`interventions` are resolved after its natural history, and any that remove it
from transmission (isolation, quarantine on being traced) shorten that window.

Each proposed infection is then put to the composed competing risks — the
built-in per-individual susceptibility and infectiousness, the model's own
`risks` (what [`transmission_risks`](@ref) reports), and the interventions — and
declined if any of them blocks it. The target keeps the candidate it already had
and stays susceptible to its other neighbours.

A model with several transmission routes passes `routes`, a collection of
`(RouteWindow, targets)` pairs, in place of `from`/`until`/`targets`. Each route
opens and closes on its own window, and only a route listing
`INTERVENTION_REMOVAL` in its `until` is cut by the interventions.

`contacts(infective_id, state)` yields the ids of everyone that case was in
contact with, whether or not transmission followed, which is what contact
tracing acts on; it is therefore usually wider than `targets`, which yields only
those still susceptible. Omit it when the model has no interventions that trace.
A model whose contacts can come about later than the case's infection, such as
at a funeral, yields `(id, time)` pairs instead, where `time` is when that
person became a contact (`-Inf` for a standing relationship); it reaches the
interventions as `trace_contacts!`'s `not_before`.

The contact-interval `kernel` must be a **non-negative** distribution: the
"each pop is final" invariant relies on a candidate time `open_t + dt` never
preceding the infector's own window-open (`dt ≥ 0`). A kernel with support on
the negatives would break the shortest-path race with no error.
"""
function _sellke_race!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; seed!, targets = nothing,
        from::Union{Symbol, Nothing} = nothing, until::Union{Tuple, Nothing} = nothing,
        routes = nothing, interventions = (), contacts = nothing, risks = ())
    # A model either passes `routes`, a collection of `(RouteWindow, targets)`
    # pairs, or the single-route shorthand `from`/`until`/`targets`. The
    # shorthand's one window opts into intervention removal, which is what a
    # model with no route structure of its own means by isolation. Passing both
    # is an error, because a routed model's windows would silently drop the
    # shorthand's censoring, including intervention removal.
    if routes === nothing
        targets === nothing && throw(ArgumentError(
            "_sellke_race! needs either `routes` or the `targets` shorthand"))
        rts = ((
            RouteWindow(:transmission; from = something(from, :infection),
                until = (something(until, ())..., INTERVENTION_REMOVAL),
                kernel = nothing),
            targets),)
    else
        (targets === nothing && from === nothing && until === nothing) ||
            throw(ArgumentError(
                "_sellke_race! takes either `routes` or `from`/`until`/`targets`, " *
                "not both; list the censoring states in each route's `until`"))
        rts = routes
    end
    m = length(members)
    best = fill(Inf, m)
    src = zeros(Int, m)
    processed = falses(m)
    pos = Dict{Int, Int}(id => k for (k, id) in enumerate(members))

    seed!(best, members, rng)

    # The heap orders pending candidates by `(time, k)`, so equal times settle in
    # member order. Candidate times only ever decrease, so a relaxation pushes a
    # new entry and leaves the old one in the heap. On pop, the loop skips an
    # entry as stale if its member has already settled or its time is later than
    # that member's current best.
    pending = Tuple{eltype(best), Int}[]
    for k in 1:m
        best[k] < Inf && _heap_push!(pending, (best[k], k))
    end

    while !isempty(pending)
        bt, j = _heap_pop!(pending)
        (processed[j] || bt > best[j]) && continue
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
        # Onset was first drawn when the node was created at time 0. Recompute
        # it from the infection time before transitions and interventions, such
        # as onset-triggered isolation, read it.
        _set_onset_from_incubation!(ind)
        resolve_transitions!(state, ind)
        _resolve_interventions!(state, ind, interventions)
        _trace_from!(state, ind, interventions, contacts, pos, processed)

        # Each route opens and closes on its own states, so a case can still be
        # transmitting on one while another has been cut. A route whose `from`
        # state was never reached contributes nothing, which is how a survivor
        # never materialises funeral contacts.
        for (w, route_targets) in rts
            open_t = window_open(ind, w)
            isfinite(open_t) || continue
            close_t = _route_close(ind, w, interventions)
            # A route the interventions cannot cut is not cut by their per-contact
            # risks either. Listing `INTERVENTION_REMOVAL` is a route's whole
            # statement about whether the response reaches it, and a route that
            # withholds it means the case goes on infecting along it: a household
            # route runs on through an isolation, and blocking every proposal it
            # makes would cut it just as surely. The model's own risks and the
            # per-individual multipliers always apply — those belong to the
            # people and the edge rather than to the response.
            route_interventions = INTERVENTION_REMOVAL in w.until ? interventions : ()

            for (target_id, kernel) in route_targets(members[j], state)
                k = get(pos, target_id, 0)
                (k == 0 || processed[k]) && continue
                dt = rand(rng, kernel)
                cand = open_t + dt
                (cand <= close_t && cand < best[k]) || continue
                _composed_risks_block(state, ind, state.individuals[target_id],
                    cand, risks, route_interventions,
                    _SELLKE_RISK_SOURCES) && continue
                best[k] = cand
                src[k] = members[j]
                _heap_push!(pending, (best[k], k))
            end
        end
    end
    return nothing
end
