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
# the engine's per-generation hook passes never fire. The one intervention seam
# is the infectious window: an intervention that removes a case from onward
# transmission (isolation) shortens it. After a case's natural history is
# stamped, run each intervention's per-individual resolution (so `Isolation`
# writes its isolation time), then close the window at the earliest removal
# across interventions as well as the `until` states.

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

# Whether a continuous-time model honours an intervention — i.e. can express it
# through the infectious window. Perfect isolation shortens the window; a leaky
# isolation (`post_isolation_transmission > 0`) only reduces transmission, which
# the window cannot express, so it is not honoured. Contact tracing is honoured:
# quarantining a traced contact removes it from transmission, which is a window
# close (see `trace_contacts!` and `infectious_removal_time(::ContactTracing,…)`).
# Interventions whose effect is purely a per-contact competing risk against the
# infection event itself, such as leaky vaccination, still have no window
# representation. `Scheduled` delegates to its wrapped intervention — the loop
# exposes the running clock/count (see `_resolve_interventions!`), so its
# time/count gate is honoured whenever the wrapped intervention is. The model
# warns for the unhonoured ones rather than silently ignoring them.
#
# Tracing needs one thing more than a window: the model has to be able to name
# the contacts a case reached, which is what `supplies_contacts` reports. A
# graph names a node's neighbours and a household its members, but the
# mass-action pool has no pairwise contact structure, so tracing has nothing to
# act along there and stays unhonoured.
_sellke_honours(model, ::AbstractIntervention) = false
_sellke_honours(model, iso::Isolation) = iso.post_isolation_transmission == 0
_sellke_honours(model, ::ContactTracing) = supplies_contacts(model)
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
    for cid in contacts(infector.id, state)
        k = get(pos, cid, 0)
        (k == 0 || processed[k]) && continue
        push!(pending, state.individuals[cid])
    end
    isempty(pending) && return nothing
    for iv in interventions
        trace_contacts!(iv, state, infector, pending)
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
    @warn "$(nameof(typeof(model))) is a continuous-time model that expresses " *
          "interventions only through the infectious window; it does not honour " *
          "these, which will have no effect: $(join(unhonoured, ", ")). Express " *
          "such control as a removal `Transition` in the progression instead."
    return nothing
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
`from`-timed contact interval accepted inside its infectious window. Each case's
`interventions` are resolved after its natural history, and any that remove it
from transmission (isolation, quarantine on being traced) shorten that window.

`contacts(infective_id, state)` yields the ids of everyone that case was in
contact with, whether or not transmission followed, which is what contact
tracing acts on; it is therefore usually wider than `targets`, which yields only
those still susceptible. Omit it when the model has no interventions that trace.

The contact-interval `kernel` must be a **non-negative** distribution: the
"each pop is final" invariant relies on a candidate time `open_t + dt` never
preceding the infector's own window-open (`dt ≥ 0`). A kernel with support on
the negatives would break the shortest-path race with no error.
"""
function _sellke_race!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; seed!, targets = nothing,
        from::Symbol = :infection, until::Tuple = (), routes = nothing,
        interventions = (), contacts = nothing)
    # A model either passes `routes`, a collection of `(RouteWindow, targets)`
    # pairs, or the single-route shorthand `from`/`until`/`targets`. The
    # shorthand's one window opts into intervention removal, which is what a
    # model with no route structure of its own means by isolation.
    rts = routes === nothing ?
          ((
        RouteWindow(:transmission; from = from,
            until = (until..., INTERVENTION_REMOVAL), kernel = nothing),
        targets),) : routes
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
        _resolve_interventions!(state, ind, interventions)
        _trace_from!(state, ind, interventions, contacts, pos, processed)

        # Each route opens and closes on its own states, so a case can still be
        # transmitting on one while another has been cut. A route whose `from`
        # state was never reached contributes nothing, which is how a survivor
        # never materialises funeral contacts.
        for (w, route_targets) in rts
            open_t = _window_open(ind, w.from)
            isfinite(open_t) || continue
            close_t = _route_close(ind, w, interventions)

            for (target_id, kernel) in route_targets(members[j], state)
                k = get(pos, target_id, 0)
                (k == 0 || processed[k]) && continue
                dt = rand(rng, kernel)
                cand = open_t + dt
                (cand <= close_t && cand < best[k]) || continue
                best[k] = cand
                src[k] = members[j]
            end
        end
    end
    return nothing
end
