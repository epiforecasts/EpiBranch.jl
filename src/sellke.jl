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
# neighbour. So the risks are resolved against the time itself, exactly as the
# generation engine resolves them against a contact's transmission time.
#
# They are resolved when the proposal reaches the front of the queue, not when
# the infector makes it. By then every case infected before the proposed time
# has settled and run its trace, so a dose given along another case's trace
# before that time is already on the target, as it would be on the generation
# engine; resolving at the proposal would read the target as it stood when the
# infector settled, and miss it.
#
# A block declines the proposal and does nothing else. The neighbour stays
# susceptible to everyone else, falls back to its next proposal in the queue,
# and the race carries on. The kernel draw is not
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
# On a model with several routes, which interventions' risks a route resolves is
# each intervention's `risk_scope`: a removal's risk only on the routes that opted
# into intervention removal, anything else on every route — see the proposal
# loop for why. The model's own risks and the per-individual multipliers apply on
# every route.
#
# Risks are resolved only for a proposal that reaches the front of the queue
# while its neighbour is still susceptible. Any other could not have infected
# anyone whatever the risks said, so skipping it changes no outcome and keeps
# the cost at nothing for a model with no risks in play: the built-in sources
# return no risk at all when every multiplier is 1, so nothing is drawn from the
# rng.

# Whether `f` has a method for `argtypes` more specific than the fallback defined
# on `base`, i.e. whether a type has implemented a hook itself.
function _has_own_method(f, argtypes::Tuple, base::Type)
    fallback = (base, argtypes[2:end]...)
    return which(f, argtypes) !== which(f, fallback)
end

# Whether an intervention implements a hook that only the generation engine calls.
function _has_generation_hook(iv::AbstractIntervention)
    T = typeof(iv)
    _has_own_method(apply_post_transmission!, (T, Any, Any), AbstractIntervention) ||
        _has_own_method(keep_active, (T, Any, Any, Any), AbstractIntervention)
end

# Earliest time any intervention removes `ind` from onward transmission.
function _intervention_removal_time(ind, interventions)
    t = Inf
    for iv in interventions
        t = min(t, infectious_removal_time(iv, ind))
    end
    return t
end

# The interventions whose risks apply on a route that has not opted into
# intervention removal.
function _every_route_interventions(interventions)
    filter(iv -> risk_scope(iv) isa EveryRoute, interventions)
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
with. That difference is the whole reason routes are separated. The same holds
for the per-contact risk of a leaky isolation, but not for a vaccine's
protection, which applies on every route; see [`risk_scope`](@ref
EpiBranch.risk_scope).
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
# as each case is resolved. That covers most of what an intervention does.
# `Scheduled` delegates to its wrapped intervention — the loop exposes the running
# clock/count (see `_resolve_interventions!`), so its time/count gate is honoured
# whenever the wrapped intervention is.
#
# What has no continuous-time representation is a *generation-shaped* hook.
# `apply_post_transmission!` and `keep_active` act on a batch of freshly created
# contact objects, and a race that settles one pre-existing node at a time never
# builds those. So an intervention with a method of its own for either hook is
# taken to reach its targets that way, and reported as unhonoured, unless it
# also traces contacts: `trace_contacts!` is then its continuous-time
# counterpart, which needs a model that can name a case's contacts. The check
# reads the methods themselves, so an intervention written outside the package
# is reported without declaring anything. `MassVaccination`'s rollout, for one,
# doses each new contact as the generation engine creates it, so on the
# continuous-time path nobody is ever dosed and the efficacy risk it contributes
# never fires; `GroupVaccination` doses whole groups as their members are
# created, and goes the same way.
#
# Tracing needs one thing more: the model has to be able to name the contacts a
# case reached, which is what `supplies_contacts` reports. A graph names a node's
# neighbours and a household its members, but the mass-action pool has no
# pairwise contact structure, so tracing has nothing to act along there and stays
# unhonoured — and ring vaccination, which doses along the trace, with it.
#
# Two parts of ring vaccination are timed from the contact's own exposure, which
# a contact the race has not settled does not yet have: the `eligibility_window`,
# and the post-exposure abort, drawn against the exposure when the dose is given.
# Either would be measured from the zero an unsettled node was created with, so a
# ring that sets one is reported as unhonoured, and does not dose at all, rather
# than applying it wrongly.
function _sellke_honours(model, iv::AbstractIntervention)
    _has_generation_hook(iv) || return true
    return traces_contacts(iv) && supplies_contacts(model)
end
function _sellke_honours(model, rv::RingVaccination)
    supplies_contacts(model) && _ring_doses_on_race(rv)
end
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

Each proposed infection is put to the composed competing risks — the built-in
per-individual susceptibility and infectiousness, the model's own `risks` (what
[`transmission_risks`](@ref) reports), and the interventions — when it reaches
the front of the queue, and declined if any of them blocks it. The target then
falls back to its next proposal and stays susceptible to its other neighbours.

A model with several transmission routes passes `routes`, a collection of
`(RouteWindow, targets)` pairs, in place of `from`/`until`/`targets`. Each route
opens and closes on its own window, and only a route listing
`INTERVENTION_REMOVAL` in its `until` is cut by the interventions' removals and
blocked by the risks of those whose [`risk_scope`](@ref) is `RemovalRoutes()`.
The risks of every other intervention apply on every route.

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
    every_route_interventions = _every_route_interventions(interventions)

    seed!(best, members, rng)

    # The heap holds every pending proposal as `(time, k, infector, route)`, the
    # seeds with infector 0. A proposal's risks are resolved only when it is
    # popped (see the block comment above), so a neighbour keeps all of its
    # proposals rather than the earliest: a blocked one falls through to the
    # next. Equal times settle in member order, and a seed before any proposal.
    # `best` holds only the seeds, which no risk blocks, so a proposal no earlier
    # than its neighbour's seed can never win and is not queued.
    pending = Tuple{eltype(best), Int, Int, Int}[]
    for k in 1:m
        best[k] < Inf && _heap_push!(pending, (best[k], k, 0, 0))
    end

    while !isempty(pending)
        bt, j, infector_id, route = _heap_pop!(pending)
        processed[j] && continue
        if infector_id != 0
            # A route the interventions cannot cut is not cut by the per-contact
            # risks that stand in for a removal either: a household route runs on
            # through an isolation, and blocking every proposal it makes would cut
            # it just as surely. A vaccine's protection is no removal, and a
            # vaccinated person is protected at home too, so risks scoped to
            # every route still apply. The model's own risks and the
            # per-individual multipliers always apply, since they belong to the
            # people and the edge.
            w, _ = rts[route]
            route_interventions = INTERVENTION_REMOVAL in w.until ? interventions :
                                  every_route_interventions
            _composed_risks_block(state, state.individuals[infector_id],
                state.individuals[members[j]], bt, risks, route_interventions,
                _sellke_builtin_risk_blocks) && continue
        end
        processed[j] = true
        best[j] = bt
        src[j] = infector_id

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
        for (route, (w, route_targets)) in enumerate(rts)
            open_t = window_open(ind, w)
            isfinite(open_t) || continue
            close_t = _route_close(ind, w, interventions)
            for (target_id, kernel) in route_targets(members[j], state)
                k = get(pos, target_id, 0)
                (k == 0 || processed[k]) && continue
                dt = rand(rng, kernel)
                cand = open_t + dt
                (cand <= close_t && cand < best[k]) || continue
                _heap_push!(pending, (convert(eltype(best), cand), k, members[j], route))
            end
        end
    end
    return nothing
end
