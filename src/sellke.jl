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
# neighbour. So the risks are resolved against that time, exactly as the
# generation engine resolves them against a contact's transmission time — but
# when the candidate is popped rather than when it was proposed, so that a dose
# or an isolation that arrived in between is in force (see the loop in
# `_sellke_race!`).
#
# A blocked contact does not transmit, and the pair goes on meeting: its next
# contact is a draw from the same kernel conditioned on falling later, offered if
# the infector's window is still open for it. The points of a sequence drawn that
# way are the points of the kernel's own hazard `h(t)`, so blocking each of them
# with probability `p` thins that hazard to `(1-p)·h(t)` and the first contact
# that gets through arrives with survival `S(t)^(1-p)`. That is the per-exposure
# reading of a leaky vaccine and the rate-multiplier reading of a relative
# susceptibility, and it is what the mass-action pool does with the stream of
# contacts it delivers: a clique of this race and an equivalent pool are then the
# same process. A degenerate kernel (`Dirac`) has one contact and no more, so a
# block ends that pair.
#
# It is not what a block means on the generation engine, where a parent draws a
# fixed set of contacts and a blocked one is a transmission lost with nothing to
# follow it: an efficacy of 0.5 halves that pair's transmissions there, and here
# it leaves `1 - exp(-∫h/2)` of them.
#
# Thinning a hazard keeps a pair's contact process in the family the pairwise
# likelihood is written in, with its hazard scaled, so a susceptibility that is
# in force throughout stays representable wherever that family is closed under
# proportional hazards — an exponential contact interval, for one. A risk that
# arrives partway through the window, such as an isolation or a dose given by a
# trace, is not: the likelihood has no term for a blocked contact.
#
# On a model with several routes, which interventions' risks a route resolves is
# each intervention's `risk_scope`: a removal's risk only on the routes that opted
# into intervention removal, anything else on every route — see the proposal
# loop for why. The model's own risks and the per-individual multipliers apply on
# every route.
#
# The built-in sources return no risk at all when every multiplier is 1, so
# nothing is drawn from the rng and a run without risks reproduces the same
# seeded results, through the same heap, as it did before per-contact risks
# existed.

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

# One case's window on one route: who it is, which route, and when that window
# opened and closes. Every proposal a case makes along a route shares one of
# these, and names it by its index, so the race remembers each proposal in a
# little over a word plus its time.
struct _RouteOpening{T}
    infector::Int
    route::Int
    open_t::T
    close_t::T
end

# Record a proposal to member `k` from opening `w`, due at `t`, and put it in the
# heap when it is that member's earliest.
function _propose!(pending, proposals, head, best, represents, k, w, t, may_block)
    # With nothing to block the contact there is no fallback to keep, so the
    # entry names the opening itself and only an improvement is pushed.
    pid = w
    if may_block
        push!(proposals, _Pending(w, head[k], t, false))
        pid = length(proposals)
        head[k] = pid
    end
    if t < best[k]
        best[k] = t
        represents[k] = pid
        may_block && (proposals[pid] = _queue(proposals[pid]))
        _heap_push!(pending, (t, k, pid))
    end
    return nothing
end

# After a member's earliest contact was blocked, hand its place to the next
# earliest. That one may still have an entry in the heap, from before a later
# proposal overtook it, in which case there is nothing to push.
function _requeue!(pending, proposals, head, best, represents, k)
    pid = 0
    t = oftype(best[k], Inf)
    q = Int(head[k])
    while q != 0
        if proposals[q].time < t
            t = proposals[q].time
            pid = q
        end
        q = proposals[q].chain
    end
    best[k] = t
    represents[k] = pid
    if pid != 0 && !proposals[pid].queued
        proposals[pid] = _queue(proposals[pid])
        _heap_push!(pending, (t, k, pid))
    end
    return nothing
end

# One proposal: the opening it was made from, the next proposal to the same
# member, when its next contact falls, and whether it has an entry in the heap.
struct _Pending{T}
    opening::Int
    chain::Int
    time::T
    queued::Bool
end
_queue(p::_Pending) = _Pending(p.opening, p.chain, p.time, true)
_dequeue(p::_Pending) = _Pending(p.opening, p.chain, p.time, false)
_at(p::_Pending, t) = _Pending(p.opening, p.chain, t, p.queued)

# The kernel of one pair on the route numbered `route`.
function _route_pair_kernel(rts, route, infector_id, target_id, state)
    for (ri, (_, route_targets)) in enumerate(rts)
        ri == route && return _pair_kernel(route_targets, infector_id, target_id, state)
    end
    return nothing
end

# The kernel of one pair on one route, asked of the model again. Only a blocked
# contact needs it.
function _pair_kernel(route_targets, infector_id, target_id, state)
    for (tid, kernel) in route_targets(infector_id, state)
        tid == target_id && return kernel
    end
    return nothing
end

# The infector's infectiousness and the target's susceptibility as a rate
# multiplier `m` on the pair kernel, rather than a Bernoulli thin: scaling a
# hazard by `m` turns its survival function S(t) into S(t)^m, which is drawn by
# inverse-transform on the *survival* scale, at the survival `U^(1/m)` for
# `U ~ Uniform(0, 1)`. That survival goes to `cquantile`, the quantile of the
# complement, rather than to `quantile` as `1 - U^(1/m)`: a small multiplier puts
# much of the mass at a survival below `eps()/2`, where subtracting from 1 rounds
# the argument to 1 and loses the contact altogether. `rand(rng, kernel)` is the
# `m == 1` case of the same draw, kept as a fast path since every pair without
# either trait set takes it. `m <= 0` (either trait exactly zero) never
# transmits, and draws nothing.
function _traits_scaled_draw(rng::AbstractRNG, kernel, m::Real)
    m == 1 && return rand(rng, kernel)
    m <= 0 && return oftype(float(m), Inf)
    return cquantile(kernel, rand(rng)^(1 / m))
end

# The pair's next contact after the one at `dt`, as a time from the window
# opening: the same `m`-scaled hazard, conditioned on falling later than `dt`.
# Its survival above `dt` is `(S(t)/S(dt))^m`, so one uniform `U` gives it at the
# survival `S(dt)·U^(1/m)`, for any kernel, at one `ccdf` and one `cquantile`
# call — both on the survival scale, for the reason above. A kernel whose support
# ends at or before `dt` has no later contact to give: a degenerate (`Dirac`)
# contact interval is one such, offering exactly one contact.
_next_contact(::AbstractRNG, ::Nothing, ::Real, dt) = Inf
function _next_contact(rng::AbstractRNG, kernel, m::Real, dt)
    m <= 0 && return Inf
    s = ccdf(kernel, dt)
    s > 0 || return Inf
    nxt = cquantile(kernel, s * rand(rng)^(1 / m))
    return nxt > dt ? nxt : Inf
end

# Earliest time any intervention removes `ind` from onward transmission.
function _intervention_removal_time(ind, interventions)
    t = Inf
    for iv in interventions
        t = min(t, infectious_removal_time(iv, ind))
    end
    return t
end

# Whether the composed risks block `parent` infecting `contact` at the proposed
# `transmission_time` on a continuous-time model. On the generation engine a
# contact's `infection_time` already holds its transmission time when the risks
# are resolved, and a risk may read the exposure from there. A contact these
# models propose is still susceptible, so its `infection_time` holds nothing
# yet: set it to the proposed time for the resolution, and put it back if the
# contact is blocked, leaving it as it was.
function _proposal_blocked(state::SimulationState, parent, contact, transmission_time,
        model_risks, interventions)
    previous = contact.infection_time
    contact.infection_time = transmission_time
    blocked = _composed_risks_block(state, parent, contact, transmission_time,
        model_risks, interventions, _sellke_builtin_risk_blocks)
    blocked && (contact.infection_time = previous)
    return blocked
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
_sellke_honours(model, ::ContactTracing) = supplies_contacts(model)
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

Each proposed infection is then put to the composed competing risks — the
built-in per-individual susceptibility and infectiousness, the model's own
`risks` (what [`transmission_risks`](@ref) reports), and the interventions — and
declined if any of them blocks it. The target keeps the candidate it already had
and stays susceptible to its other neighbours.

A model with several transmission routes passes `routes`, a collection of
`(RouteWindow, targets)` pairs, in place of `from`/`until`/`targets`. Each route
opens and closes on its own window, and only a route listing
`INTERVENTION_REMOVAL` in its `until` is cut by the interventions' removals and
blocked by the risks of those whose [`risk_scope`](@ref) is `RemovalRoutes()`.
The risks of every other intervention apply on every route.

`introduction`, when given, is the `(kernel, until)` of the community hazard the
model seeded its members from: the contact-interval distribution of an
introduction from outside the population, and the time the introduction window
closes. It says that a seeded time is a community introduction rather than an
index case, so the composed risks are resolved against it — a vaccinated person
is protected from the community as from a neighbour — and a blocked introduction
is followed by the next one from the same hazard. Omit it for a model whose
seeds are index cases.

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
        routes = nothing, interventions = (), contacts = nothing, risks = (),
        introduction = nothing)
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
    processed = falses(m)
    pos = Dict{Int, Int}(id => k for (k, id) in enumerate(members))
    # A route the interventions cannot cut is not cut by the per-contact risks
    # that stand in for a removal either: a household route runs on through an
    # isolation, and blocking every proposal it makes would cut it just as
    # surely. A vaccine's protection is no removal, and a vaccinated person is
    # protected at home too, so risks scoped to every route still apply. The
    # model's own risks and the per-individual multipliers always apply, since
    # they belong to the people and the edge.
    every_route_interventions = _every_route_interventions(interventions)
    route_interventions = [INTERVENTION_REMOVAL in w.until ? interventions :
                           every_route_interventions for (w, _) in rts]

    seed!(best, members, rng)

    T = eltype(best)
    # A popped entry is final unless the risks block it: every other pending
    # proposal is at a later time, a blocked pair's next contact is later again,
    # and a proposal still to be made opens no earlier than the infection time of
    # a case that has not settled. So the risks are resolved on the pop rather
    # than when the infector settled, against the state as it stands at the
    # candidate time — which is what a contact traced, and dosed, in between
    # depends on.
    #
    # A member whose earliest contact is blocked falls back on its other
    # proposals, so when something can block, every proposal is kept: `proposals`
    # says which opening each was made from, when its next contact falls, and
    # which proposal to the same member comes next in the list `head` starts.
    # Only the member's earliest is in the heap, named by `represents`, and when
    # a contact is blocked the next earliest takes its place.
    #
    # Nothing can block unless the model or an intervention says so — the two
    # per-individual traits are in the draw, not here — and then no member ever
    # falls back, so the lists are not built at all and a heap entry names its
    # opening directly. That path pushes, pops and draws exactly what the race
    # did before per-contact risks existed.
    # Whether any per-individual trait is set. Both are constants of the two
    # people, so a race where none is set can leave the target's record alone
    # when it draws — a saving worth having on a dense graph, where that record
    # is a pointer chase per proposal. A trait an intervention writes as a case
    # is resolved turns it on from there.
    traits = any(
        id -> (ind = state.individuals[id];
            ind.susceptibility != 1 || ind.infectiousness != 1),
        members)
    may_block = !isempty(risks) ||
                any(
        iv -> _has_own_method(competing_risk, (typeof(iv), Any, Any, Any),
            AbstractIntervention),
        interventions)
    openings = _RouteOpening{T}[] # one per case and route it transmits along
    proposals = _Pending{T}[]  # every proposal made, when something can block
    head = zeros(Int, may_block ? m : 0)  # first proposal to each member
    represents = zeros(Int, m) # what each member's heap entry names
    pending = Tuple{T, Int, Int}[]

    # The seeds' own opening: no infector, so nothing about it is ever read.
    push!(openings, _RouteOpening(0, 0, zero(T), T(Inf)))
    for k in 1:m
        best[k] < Inf || continue
        seeded = best[k]
        best[k] = T(Inf)
        _propose!(pending, proposals, head, best, represents, k, 1, seeded, may_block)
    end

    while !isempty(pending)
        bt, j, p = _heap_pop!(pending)
        may_block && (proposals[p] = _dequeue(proposals[p]))
        (processed[j] || represents[j] != p) && continue
        opening = openings[may_block ? proposals[p].opening : p]
        infector_id = opening.infector

        ind = state.individuals[members[j]]
        # A seeded time is a community introduction when the model gave the race
        # an `introduction`, and an index case otherwise. An introduction arrives
        # from outside the population, so it is put to the risks like any other
        # contact, with the person standing in for the infector the model does
        # not have; an index case is where an outbreak is defined to start and is
        # put to none, as on the generation engine.
        source = infector_id == 0 ? ind : state.individuals[infector_id]
        if may_block && (infector_id != 0 || introduction !== nothing) &&
           _proposal_blocked(
               state, source, ind, bt, risks,
               opening.route == 0 ? interventions : route_interventions[opening.route])
            # The contact did not transmit, and the source goes on meeting the
            # person: the next contact is a draw from the same hazard conditioned
            # on falling later, kept while the window is still open for it.
            if opening.route == 0
                kernel, close_t = introduction
                open_t = zero(T)
                mult = ind.susceptibility
            else
                # Which route's targets to ask is known only at run time, so the
                # routes are walked rather than indexed: indexing a tuple of
                # routes with a running value would put the whole tuple on the
                # heap, once per race.
                kernel = _route_pair_kernel(rts, opening.route, infector_id,
                    members[j], state)
                open_t = opening.open_t
                close_t = opening.close_t
                mult = source.infectiousness * ind.susceptibility
            end
            nxt = open_t + _next_contact(rng, kernel, mult, bt - open_t)
            proposals[p] = _at(proposals[p], nxt <= close_t ? nxt : T(Inf))
            _requeue!(pending, proposals, head, best, represents, j)
            continue
        end
        processed[j] = true

        ind.infection_time = bt
        ind.state[:infected] = true
        ind.state[:index] = infector_id == 0
        if infector_id != 0
            infector = state.individuals[infector_id]
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
        traits |= ind.susceptibility != 1 || ind.infectiousness != 1

        # Each route opens and closes on its own states, so a case can still be
        # transmitting on one while another has been cut. A route whose `from`
        # state was never reached contributes nothing, which is how a survivor
        # never materialises funeral contacts.
        for (ri, (w, route_targets)) in enumerate(rts)
            open_t = window_open(ind, w)
            isfinite(open_t) || continue
            close_t = _route_close(ind, w, interventions)
            push!(openings, _RouteOpening(members[j], ri, open_t, close_t))
            opening_id = length(openings)

            for (target_id, kernel) in route_targets(members[j], state)
                k = get(pos, target_id, 0)
                (k == 0 || processed[k]) && continue
                # Both per-individual traits are rate multipliers on this
                # pair's contact interval, folded into the draw rather than
                # resolved contact by contact. A pair at the default 1 draws
                # exactly as it did before they were honoured here.
                dt = traits ?
                     _traits_scaled_draw(rng, kernel,
                    ind.infectiousness *
                    state.individuals[target_id].susceptibility) :
                     rand(rng, kernel)
                cand = open_t + dt
                cand <= close_t || continue
                _propose!(pending, proposals, head, best, represents, k, opening_id,
                    cand, may_block)
            end
        end
    end
    return nothing
end
