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
    return from === :infection ? ind.infection_time :
        convert(T, get(ind.state, Symbol(from, :_time), T(Inf)))
end

# Earliest of the `until` removal states' times (Inf if none reached).
function _window_close(ind::Individual{T}, until::Tuple) where {T}
    return isempty(until) ? T(Inf) :
        minimum(convert(T, get(ind.state, Symbol(s, :_time), T(Inf))) for s in until)
end

# ── Interventions on the continuous-time (Sellke) models ─────────────
# These models run their own event loop rather than the generation engine, so
# the engine's per-generation hook passes never run. Two seams carry an
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
# Rejection continuations require finite remaining integrated hazard; otherwise
# an opaque risk could reject contacts forever and the model is refused.
#
# A risk whose block is certain and does not fade — an `AllOrNothingMode`
# responder blocked from its immunity time, an aborted infection's infector —
# answers every later proposal on the pair the same way it just answered this
# one, so nothing is left to gain from asking the model or the kernel again:
# the pair is dropped from the race instead of drawing towards a foregone
# conclusion, which is also what spares an unbounded window from the
# rejection-continuation guard above. The pair still stands in each other's
# contacts; tracing and ring construction read that, not the proposals the
# race no longer makes, so they are unaffected either way.
#
# An output that wants every contact event — to count how many exposures a
# vaccine averted, say — needs the draws the race would otherwise skip. A
# `recorder` ([`ContactRecorder`](@ref)) attached to the composed model is
# asked, every time a standing block would end a pair's draws, whether they
# still matter ([`records_contacts`](@ref EpiBranch.records_contacts)); if it
# says yes the race keeps drawing exactly as it does for a block that is not
# certain, which puts the pair back under the rejection-continuation guard
# rather than needing one of its own. The default `NoContactRecorder` answers
# no to every pair, so a run with no recorder attached drops the pair exactly
# as above.
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
# selected by `risk_applies(intervention, route)`: removal effects use the
# route's censoring states, while protection defaults to every route. The model's own risks and the per-individual multipliers apply on
# every route.
#
# The built-in sources return no risk at all when every multiplier is 1, so
# nothing is drawn from the rng and a run without risks reproduces the same
# seeded results, through the same heap, as it did before per-contact risks
# existed.

# Whether `f` has a method for `argtypes` more specific than the fallback defined
# on `base`, i.e. whether a type has implemented a hook itself.
function _has_own_method(f, T::Type, base::Type)
    # `methods` rather than `which`, which finds only a method whose parameters
    # accept `Any`: an intervention that types its hook's arguments, as the style
    # guide asks, has one `which` looks straight past.
    # Julia 1.10 also lists the less specific methods `T` falls back on, so only
    # a method narrower than `base` counts.
    return any(methods(f, Tuple{T, Vararg{Any}})) do mm
        p = Base.unwrap_unionall(mm.sig).parameters[2]
        p isa TypeVar && (p = p.ub)
        p !== base && p <: base
    end
end

# The generation-only hooks an intervention implements a method of its own
# for, named so a warning can say exactly which ones the race skips rather
# than claiming the intervention itself does nothing.
function _generation_hooks(iv::AbstractIntervention)
    T = typeof(iv)
    hooks = Symbol[]
    _has_own_method(apply_post_transmission!, T, AbstractIntervention) &&
        push!(hooks, :apply_post_transmission!)
    _has_own_method(keep_active, T, AbstractIntervention) && push!(hooks, :keep_active)
    return hooks
end
_generation_hooks(s::Scheduled) = _generation_hooks(s.intervention)

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

# After a member's earliest contact was blocked, or its proposals were redrawn,
# hand its place to the earliest remaining. Contacts at the same time go to the
# opening made first, as `_propose!` orders them, so the infector does not
# depend on the order the chain was built in. The chosen proposal may still have
# an entry in the heap, from before a later proposal overtook it, in which case
# there is nothing to push.
function _requeue!(pending, proposals, head, best, represents, k)
    pid = 0
    t = oftype(best[k], Inf)
    q = Int(head[k])
    while q != 0
        time = proposals[q].time
        if time < t ||
                (time == t && pid != 0 && proposals[q].opening < proposals[pid].opening)
            t = time
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
# hazard by `m` turns its survival function S(t) into S(t)^m, so the draw is the
# time whose survival is `U^(1/m)` for `U ~ Uniform(0, 1)`. Both the survival and
# its inverse are taken in logs, at `log(U)/m`: a small multiplier sends
# `U^(1/m)` itself to zero below about 1e-324, and puts it into the range where
# the inverse of a `Gamma` survival raises a `DomainError` long before that,
# while `log(U)/m` stays an ordinary number. `rand(rng, kernel)` is the `m == 1`
# case of the same draw, kept as a fast path since every pair without either
# trait set takes it. `m <= 0` (either trait exactly zero) never transmits, and
# draws nothing.
function _traits_scaled_draw(rng::AbstractRNG, kernel, m::Real)
    m == 1 && return rand(rng, kernel)
    m <= 0 && return oftype(float(m), Inf)
    return _time_at_log_survival(kernel, log(rand(rng)) / m)
end

# The time whose log-survival under `kernel` is `lp`. A kernel with an
# `invlogccdf` of its own answers directly. One without gets Distributions'
# generic method, which rebuilds the argument as `-expm1(lp)` and so reaches
# exactly 1 below `lp ≈ -37`, handing back the top of the support and losing a
# contact that may be well inside the window: `Rayleigh`, a `MixtureModel`, a
# `truncated` or shifted distribution and anything written outside Distributions
# are all in that position. Rather than ask which kernel is which, the answer is
# checked against the kernel's own `logccdf` — specialised far more widely, and
# generic to `log(ccdf(...))` otherwise — and inverted by bisection on it when
# the check fails. That costs one `logccdf` on each scaled draw, and the
# bisection only where a direct answer would be wrong.
function _time_at_log_survival(kernel, lp)
    isfinite(lp) || return oftype(float(lp), Inf)
    t = invlogccdf(kernel, lp)
    isfinite(t) && isapprox(logccdf(kernel, t), lp; rtol = 1.0e-6, atol = 1.0e-12) && return t
    return _bisect_log_survival(kernel, lp)
end

# Bisect `logccdf`, which decreases in `t`, for the earliest time at or past the
# log-survival `lp`. The bracket starts at the time for a survival the direct
# inverse does get right and grows until the kernel's survival has fallen far
# enough, which a bounded support reaches at once. The loop ends when the two
# ends are adjacent floats, so it is bounded by their exponent range.
function _bisect_log_survival(kernel, lp)
    lo = float(invlogccdf(kernel, max(lp, -30)))
    (isfinite(lo) && logccdf(kernel, lo) >= lp) || (lo = float(minimum(kernel)))
    isfinite(lo) || return oftype(lo, Inf)
    # A kernel whose mass sits at its lower bound, an atom or a censored law,
    # has already fallen past `lp` there, so the bracket lies beyond the answer
    # and bisecting it would land a float late.
    logccdf(kernel, lo) <= lp && return lo
    hi = lo + max(one(lo), abs(lo))
    steps = 0
    while logccdf(kernel, hi) > lp && steps < 2000
        hi = lo + 2 * (hi - lo)
        steps += 1
        isfinite(hi) || return oftype(hi, Inf)
    end
    while true
        mid = lo + (hi - lo) / 2
        (lo < mid < hi) || return hi
        logccdf(kernel, mid) > lp ? (lo = mid) : (hi = mid)
    end
    return
end

# The pair's next contact after the one at `dt`, as a time from the window
# opening: the same `m`-scaled hazard, conditioned on falling later than `dt`.
# Its survival above `dt` is `(S(t)/S(dt))^m`, so one uniform `U` puts the next
# contact at the survival `S(dt)·U^(1/m)` — in logs, `logccdf(kernel, dt) +
# log(U)/m`, for the reason above, a sum of two ordinary numbers whatever the
# multiplier. A kernel whose support ends at or before `dt` has no survival left
# and so no later contact to give: a degenerate (`Dirac`) contact interval is one
# such, offering exactly one contact.
_next_contact(::AbstractRNG, ::Nothing, ::Real, dt, end_dt, certain_until = nothing) = Inf
function _next_contact(
        rng::AbstractRNG, kernel, m::Real, dt, end_dt, certain_until = nothing
    )
    m <= 0 && return Inf
    ls = logccdf(kernel, dt)
    isfinite(ls) || return Inf
    # An opaque risk may block forever. Rejection sampling is supported only
    # when the remaining integrated hazard is finite; a finite time alone is
    # insufficient for a continuous kernel whose support ends in the window.
    #
    # A block known to be certain is the exception, `certain_until` holding when
    # it lapses. Reaching the end of the kernel's own survival, it answers every
    # proposal that could still happen, so the pair is finished rather than
    # resampled; lapsing before then, it leaves contacts it does not block, and
    # the redraws terminate on one of them.
    if !isfinite(logccdf(kernel, end_dt)) && certain_until !== nothing
        return isfinite(logccdf(kernel, certain_until)) ?
            _draw_next_contact(rng, kernel, m, ls, dt) : oftype(ls, Inf)
    end
    isfinite(logccdf(kernel, end_dt)) || throw(
        ArgumentError(
            "repeated contacts after a blocked proposal require finite remaining " *
                "integrated hazard. Close the infectious or introduction window before " *
                "the kernel survival reaches zero, or encode static protection in the " *
                "contact kernel or host traits. The likely cause is a case whose " *
                "infectious window never closes — either the progression has no " *
                "terminal transition reaching one of `until`'s states, or one is " *
                "reachable but gated so that some cases reach none of them (see " *
                "`exclusive_probabilities` for terminal transitions meant to " *
                "partition the population exactly). A window closed only by " *
                "`INTERVENTION_REMOVAL` reaches this too when the case's " *
                "isolation or quarantine is due to lapse, since only a removal " *
                "that stands for the rest of the infectious period closes the " *
                "window by itself. A component whose certain block is in force " *
                "also lands here until it declares `binding_release`, which is " *
                "what lets the race read the release it reports."
        )
    )
    return _draw_next_contact(rng, kernel, m, ls, dt)
end

function _draw_next_contact(rng::AbstractRNG, kernel, m::Real, ls, dt)
    nxt = _time_at_log_survival(kernel, ls + log(rand(rng)) / m)
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

"""
    standing_block(source) -> Bool

Whether a certain block `source` composes stands for every later proposal on
the same pair, so a continuous-time race can stop proposing for that pair
instead of redrawing towards an answer it already has. `false` by default.

A source is asked this because the `Risk` it returns cannot answer it.
`competing_risk` reads the state, so a block that is certain at one proposal
may have lifted by the next — a ward that reopens, a campaign that ends, a
quarantine that expires — and a `Risk` holding plain numbers looks identical in
both cases. Declare `true` only for a source whose certain block, once in
force for a pair, is in force for good. A race over an unbounded window needs
that declaration to terminate; without it a certain block raises
`ArgumentError` rather than silently dropping transmission that could still
happen.
"""
standing_block(source) = false
standing_block(w::InterventionWrapper) = standing_block(w.intervention)
# An abort is recorded on the infector and never withdrawn, so the block it
# composes lasts as long as the infector does.
standing_block(::AbortedInfection) = true
# A mode that disallows `waning` draws its block once at vaccination and it
# never fades, so once immunity has developed the block it composes is certain
# for good — which is what `supports_waning` reports. A mode that allows
# waning is not declared here even at `efficacy = 1.0`, because a later waning
# value could still give a smaller block to a later exposure.
standing_block(v::AbstractVaccination) = !supports_waning(effect_mode(v))

"""
    binding_release(component) -> Bool

Whether a `release_time` this component reports on a [`Risk`](@ref) binds its
later answers: a block it says lapses at `t` is not still in force after `t`,
and a block it reports as never releasing has not lifted by the next proposal.
The default is the conservative `false`.

The continuous-time race reads this where a pair's kernel has unboundedly many
contacts left in the window, to tell a block that ends the pair from one it
must go on proposing against. Without the declaration such a block raises
rather than silently dropping transmission that could still happen, for the
reason [`standing_block`](@ref EpiBranch.standing_block) gives: `competing_risk`
reads the state, so a block that looks certain at one proposal may have lifted
by the next.

The built-in removals declare it, their stretches being append-only state that
[`record_removal!`](@ref EpiBranch.record_removal!) only ever adds to. A
[`Scheduled`](@ref) that can close declares it away again, its window closing
being exactly a block withdrawn before the release it reported.
"""
binding_release(component) = false

# Whether a resolved risk is certain and already in force at this proposal: its
# `event_time`, a plain number rather than one resampled on each ask, has
# passed, its `block_probability`, also a plain number rather than a waning
# closure that could give a smaller value to a later exposure, is 1, and its
# `release_time` is infinite, a plain number, so the block is not due to lapse.
# Necessary for a standing block but not sufficient, which is what
# `standing_block` adds.
function _standing_risk(risk::Risk, transmission_time)
    return risk.event_time isa Real && risk.event_time <= transmission_time &&
        risk.block_probability isa Real && risk.block_probability >= 1.0 &&
        risk.release_time isa Real && isinf(risk.release_time)
end

# Whether `source` contributes a standing risk against this pair: it declares
# its certain blocks permanent, and the risk it composes here is such a block.
# Building its `Risk`(s) again reads only stored state and draws nothing from
# the rng, so asking costs nothing beyond the one already-blocked proposal it is
# asked for.
function _any_standing_risk(source, parent, contact, state, transmission_time)
    standing_block(source) || return false
    for risk in _iter_risks(competing_risk(source, parent, contact, state))
        _standing_risk(risk, transmission_time) && return true
    end
    return false
end

# Whether a resolved risk is certain and already in force at this proposal,
# whatever it does later: the weaker half of `_standing_risk`, which adds that
# the block never lapses.
function _in_force_certainly(risk::Risk, transmission_time)
    return risk.event_time isa Real && risk.event_time <= transmission_time &&
        risk.block_probability isa Real && risk.block_probability >= 1.0 &&
        risk.release_time isa Real
end

# The time every certain block now in force against this pair has lapsed, or
# `Inf` where one never does; `nothing` where no certain block is in force, or
# where one of them comes from a source whose reported releases do not bind. A
# pair certainly blocked to that time cannot transmit before it, which is what
# tells the race whether redrawing can terminate.
function _certain_block_release(
        state, parent, contact, transmission_time, model_risks, interventions
    )
    release, ok = _certain_release(
        nothing, AbortedInfection(), parent, contact, state, transmission_time
    )
    ok || return nothing
    for source in model_risks
        release, ok = _certain_release(
            release, source, parent, contact, state, transmission_time
        )
        ok || return nothing
    end
    for iv in interventions
        release, ok = _certain_release(
            release, iv, parent, contact, state, transmission_time
        )
        ok || return nothing
    end
    return release
end

# The release of every certain block `source` has in force, folded into
# `release`, and whether its releases bind at all (`binding_release`). One
# whose do not could be blocking at the next proposal whatever this risk says,
# so nothing it reports can end the pair, and no other source's release can
# speak for it either.
function _certain_release(release, source, parent, contact, state, transmission_time)
    for risk in _iter_risks(competing_risk(source, parent, contact, state))
        _in_force_certainly(risk, transmission_time) || continue
        binding_release(source) || return release, false
        release = release === nothing ? risk.release_time :
            max(release, risk.release_time)
    end
    return release, true
end

# Whether the block just resolved for `parent` → `contact` at `transmission_time`
# will stand for every later proposal on the same edge, so the race can stop
# proposing for the pair instead of redrawing towards a block it already knows
# is certain: true when any risk composed for it — the continuous-time
# built-ins, the model's own, or the interventions' — is a standing risk.
function _permanently_blocked(
        state, parent, contact, transmission_time,
        model_risks, interventions
    )
    _any_standing_risk(AbortedInfection(), parent, contact, state, transmission_time) &&
        return true
    for source in model_risks
        _any_standing_risk(source, parent, contact, state, transmission_time) && return true
    end
    for iv in interventions
        # A `Scheduled` wrapper can still turn its block off later unless the
        # wrapped intervention's protection persists outside the active
        # window (`_may_lapse`), so such a risk is never read as standing here,
        # however certain it looks at this one proposal.
        _may_lapse(iv) && continue
        _any_standing_risk(iv, parent, contact, state, transmission_time) && return true
    end
    return false
end

# Whether the composed risks block `parent` infecting `contact` at the proposed
# `transmission_time` on a continuous-time model, and whether that block is
# permanent (see `_permanently_blocked`). On the generation engine a contact's
# `infection_time` already holds its transmission time when the risks are
# resolved, and a risk may read the exposure from there. A contact these
# models propose is still susceptible, so its `infection_time` holds nothing
# yet: set it to the proposed time for the resolution, and put it back if the
# contact is blocked, leaving it as it was.
function _proposal_blocked(
        state::SimulationState, parent, contact, transmission_time,
        model_risks, interventions
    )
    previous = contact.infection_time
    previous_clock = state.max_infection_time
    contact.infection_time = transmission_time
    # Scheduled risks use the proposed contact time even when earlier contacts
    # were blocked. Outside this evaluation the clock records accepted infections.
    state.max_infection_time = transmission_time
    blocked = true
    try
        blocked = _composed_risks_block(
            state, parent, contact, transmission_time,
            model_risks, interventions, _sellke_builtin_risk_blocks
        )
        permanent = blocked && _permanently_blocked(
            state, parent, contact, transmission_time, model_risks, interventions
        )
        release = blocked && !permanent ?
            _certain_block_release(
                state, parent, contact, transmission_time, model_risks, interventions
            ) : nothing
        return blocked, permanent, release
    finally
        state.max_infection_time = previous_clock
        blocked && (contact.infection_time = previous)
    end
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
protection, which applies on every route; see [`risk_applies`](@ref
EpiBranch.risk_applies).
"""
const INTERVENTION_REMOVAL = :intervention_removal

# Close a window: the earliest of its `until` states' times, the intervention
# removal when the window opted into it, and a post-exposure abort, which ends
# the infection outright and so closes every route, opted in or not — the same
# reach as the `AbortedInfection` risk that blocks each route's transmission
# from that time on. Without this, a route whose only removal state is one an
# abort undoes (see `resolve_transitions!`) never closes, and the rejection
# sampler that redraws a blocked pair's next contact has no bound to redraw
# within.
function _route_close(ind, w::RouteWindow, interventions)
    t = _window_close(ind, w.until)
    if INTERVENTION_REMOVAL in w.until
        t = min(t, _intervention_removal_time(ind, interventions))
    end
    return min(t, infection_aborted_time(ind))
end

# The one window of `_sellke_race!`'s `from`/`until`/`targets` shorthand. Reading
# a single-route model's infectious windows back out of a simulation for the
# likelihood uses the same window, and the simulator and the likelihood then
# close each case's window at the same time.
function _shorthand_window(from, until)
    return RouteWindow(
        :transmission; from = something(from, :infection),
        until = (something(until, ())..., INTERVENTION_REMOVAL), kernel = nothing
    )
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
# `apply_post_transmission!` acts on a batch of freshly created contact objects,
# and a race that settles one pre-existing node at a time never builds those.
# `keep_active` is read, but only from inside a tracing walk (`_trace_from!`
# asks it which contacts the ring grows from), so an intervention that answers
# it and does not trace has nothing to call it. So an intervention with a method
# of its own for either hook is taken to reach its targets that way, and
# reported as unhonoured, unless it has a continuous-time counterpart: a method
# of its own for `on_infection_settled!`, called once a case's infection time is
# fixed, stands in for `apply_post_transmission!`, which an intervention
# written for both engines defines alongside it to treat the same case the
# moment each engine can; `trace_contacts!` stands in for `keep_active` the same
# way, but needs a model that can name a case's contacts as well. The check
# reads the methods themselves, so an intervention written outside the package
# is reported without declaring anything. `MassVaccination`'s rollout, for one,
# doses each new contact as the generation engine creates it, so on the
# continuous-time path nobody is ever dosed and the efficacy risk it contributes
# never blocks; `GroupVaccination` doses whole groups as their members are
# created, and goes the same way.
#
# The settled-hook shortcut yields to a method of the intervention's own for
# `continuous_actions`: that already answers, for its own configuration,
# whether `apply_post_transmission!`'s mechanism has a continuous-time
# counterpart, and a settled hook written for an unrelated feature must not
# override it. `RingVaccination`'s settled hook, for one, only ever
# reconsiders a post-exposure dose, never the ring dosing
# `apply_post_transmission!` performs, so a finite `eligibility_window` —
# which makes its own `continuous_actions` false — must still leave the ring
# dosing unhonoured.
#
# The settled hook stands in only for `apply_post_transmission!`; `keep_active`
# still needs its own counterpart, `trace_contacts!`. An intervention that doses
# through a settled hook and *also* grows a ring through `keep_active` is
# therefore unhonoured wherever that ring cannot trace, even though its dosing
# alone would pass.
#
# Tracing needs one thing more: the model has to be able to name the contacts a
# case reached, which is what `supplies_contacts` reports. A graph names a node's
# neighbours and a household its members, but the mass-action pool has no
# pairwise contact structure, so tracing has nothing to act along there and stays
# unhonoured. Supported candidate actions run after tracing through the shared
# admission protocol; legacy batch-only delivery remains unsupported.
function _sellke_honours(model, iv::AbstractIntervention)
    continuous_actions(iv) && return supplies_contacts(model)
    hooks = _generation_hooks(iv)
    isempty(hooks) && return true
    T = typeof(iv)
    if !_has_own_method(continuous_actions, T, AbstractIntervention) &&
            _has_own_method(on_infection_settled!, T, AbstractIntervention)
        :keep_active in hooks || return true
    end
    return traces_contacts(iv) && supplies_contacts(model)
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
#
# A ring (`ContactTracing(depth = 2)` and beyond) grows through uninfected
# contacts too. On the generation engine an intervention asks for that through
# `keep_active`, which keeps such a contact exposing for one more generation so
# `apply_post_transmission!` reaches its own contacts in turn. The race has no
# generations to keep a node alive for, so it walks the model's own contact
# structure breadth-first instead, and asks the same hook which contacts the
# next hop starts from. The depth semantics and the `:_ring_remaining` budget
# behind them stay with `ContactTracing`, so a ring of another shape takes part
# by answering `keep_active` rather than by writing a state key this loop would
# have to recognise.
#
# `visited` stops a node being traced twice by two branches of one walk
# reaching it at once. Across walks nothing is suppressed: a node reached as an
# uninfected ring member, which later becomes an infected and eligible case in
# its own right, seeds its own fresh full-radius ring when the race settles it,
# which is the contract `ContactTracing` documents. Whether that second attempt
# happens at all is the eligibility policy's call, not this loop's.
function _trace_from!(state, infector, interventions, contacts, pos, processed)
    contacts === nothing && return nothing
    any(traces_contacts, interventions) || return nothing
    return _walk_ring!(state, infector, interventions, contacts, pos, processed)
end

# The walk itself, which reports the members it offered to tracing. A ring
# wider than one hop reaches people the settled case does not neighbour, and
# the action layer has to be offered those too; which of them a tracing policy
# actually reached is its own business, and the layer reads that from them.
function _walk_ring!(state, infector, interventions, contacts, pos, processed)
    visited = Set{Int}((infector.id,))
    frontier = Individual[infector]
    while !isempty(frontier)
        src = popfirst!(frontier)
        pending = Individual[]
        not_before = typeof(infector.infection_time)[]
        timed = false
        for c in contacts(src.id, state)
            cid, t0 = c isa Tuple ? (c[1], c[2]) : (c, -Inf)
            timed |= c isa Tuple
            k = get(pos, cid, 0)
            (k == 0 || processed[k] || cid in visited) && continue
            push!(pending, state.individuals[cid])
            push!(not_before, t0)
        end
        isempty(pending) && continue
        for iv in interventions
            if timed
                trace_contacts!(iv, state, src, pending, not_before)
            else
                trace_contacts!(iv, state, src, pending)
            end
        end
        for ind in pending
            push!(visited, ind.id)
        end
        # Which of them the ring grows from is the intervention's call. Nothing
        # here is freshly created, so `is_new` is all false; the race settles
        # pre-existing members rather than creating contacts as it goes.
        is_new = falses(length(pending))
        for iv in interventions
            for id in keep_active(iv, state, pending, is_new)
                k = get(pos, id, 0)
                (k == 0 || processed[k]) && continue
                push!(frontier, state.individuals[id])
            end
        end
    end
    delete!(visited, infector.id)
    return visited
end

# Warn once (per `simulate` call) when a continuous-time model is handed
# interventions it cannot honour, so the limitation is loud rather than silent.
# Gated on `_honours_termination_controls`, which is `false` for exactly the
# structure-driven models that run their own Sellke loop. Names the hooks the
# race skips rather than the intervention itself, since an intervention
# reported here may still act through `competing_risk`, `resolve_individual!`,
# `infectious_removal_time` or `on_infection_settled!`.
function _warn_unhonoured_interventions(model, interventions)
    _honours_termination_controls(model) && return nothing
    unhonoured = unique(
        String[
            "$(nameof(typeof(iv))) ($(join(_generation_hooks(iv), ", ")))"
                for iv in interventions if !_sellke_honours(model, iv)
        ]
    )
    isempty(unhonoured) && return nothing
    @warn "$(nameof(typeof(model))) is a continuous-time model that settles one " *
        "pre-existing case at a time, so it never creates the batches of new " *
        "contacts the generation engine's post-transmission hooks act on; it " *
        "does not honour the named hooks, which will have no effect: " *
        "$(join(unhonoured, ", ")). Express such control as a removal " *
        "`Transition` in the progression, or through an intervention that acts " *
        "when each individual is initialised, resolved, settled, or traced."
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
model's own `risks` (what [`transmission_risks`](@ref) reports) and the
interventions — and declined if any of them blocks it. The pair goes on meeting:
a declined contact is followed by a further draw on the same edge, so blocking a
fraction of a pair's contacts thins that pair's hazard by the same fraction.
Per-individual susceptibility and infectiousness reach the same thinning through
the contact-interval draw, which turns a pair's survival `S(t)` into `S(t)^m`,
so they are not resolved here. A block that is certain and does not fade —
a constant `block_probability` of 1 past its `event_time`, neither given as
a `Distribution` or function that could read differently later — answers every
later proposal on the pair the same way, so the race stops proposing for it
instead of redrawing towards a foregone block, unless `recorder` ([`records_contacts`](@ref
EpiBranch.records_contacts)) says the pair's draws still matter; the pair
remains in each other's contacts for tracing and ring construction either
way, which read that relationship rather than the proposals.

A model with several transmission routes passes `routes`, a collection of
`(RouteWindow, targets)` pairs, in place of `from`/`until`/`targets`. Each route
opens and closes on its own window, and only a route listing
`INTERVENTION_REMOVAL` in its `until` is cut by the interventions' removals and
blocked by removal risks such as isolation. Other risks select their routes
through [`risk_applies`](@ref). A case infected along one of these routes has
the route's `name` in its `:infection_route`, which `linelist` reports. The
single-window shorthand has no named route and writes nothing.

`watches` names the host records each route's kernel reads, one tuple of
`individual.state` keys per route in route order, as
[`watched_records`](@ref EpiBranch.watched_records) reports them. A route that
watches nothing draws from hazards fixed for the run and its contacts are never
redrawn; `nothing`, the default, is a model with no such kernel at all.

`introduction`, when given, is the `(kernel, until)` of the community hazard the
model seeded its members from: the contact-interval distribution of an
introduction from outside the population, and the time the introduction window
closes. It says that a seeded time is a community introduction rather than an
index case, so the risks that act on the person are resolved against it — a
vaccinated person is protected from the community as from a neighbour — and a
blocked introduction is followed by the next one from the same hazard. The risks
of isolation and quarantine are not: they
stand in for removing an infector, and an introduction's source is outside the
population. Omit `introduction` for a model whose seeds are index cases, which
are put to no risk at all. An introduced case's `:infection_route` is
`:external`, including on a model that also names routes.

`max_time` ends the race at that time: individuals whose infection would fall
later are left uninfected, and the state is exactly the full run's state
restricted to infections up to `max_time`. Returns `true` when the race ran
until no candidate infection remained (the population reached extinction) and
`false` when it was cut off at `max_time` with candidates still pending.

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
function _sellke_race!(
        state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; seed!, targets = nothing,
        from::Union{Symbol, Nothing} = nothing, until::Union{Tuple, Nothing} = nothing,
        routes = nothing, interventions = (), contacts = nothing, risks = (),
        introduction = nothing, watches = nothing, max_time = Inf,
        recorder::ContactRecorder = NoContactRecorder()
    )
    # A model either passes `routes`, a collection of `(RouteWindow, targets)`
    # pairs, or the single-route shorthand `from`/`until`/`targets`. The
    # shorthand's one window opts into intervention removal, which is what a
    # model with no route structure of its own means by isolation. Passing both
    # is an error, because a routed model's windows would silently drop the
    # shorthand's censoring, including intervention removal.
    if routes === nothing
        targets === nothing && throw(
            ArgumentError(
                "_sellke_race! needs either `routes` or the `targets` shorthand"
            )
        )
        rts = ((_shorthand_window(from, until), targets),)
    else
        (targets === nothing && from === nothing && until === nothing) ||
            throw(
            ArgumentError(
                "_sellke_race! takes either `routes` or `from`/`until`/`targets`, " *
                    "not both; list the censoring states in each route's `until`"
            )
        )
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
    introduction_interventions = filter(iv -> risk_applies(iv, nothing), interventions)
    route_interventions = [
        filter(iv -> risk_applies(iv, w), interventions)
            for (w, _) in rts
    ]

    seed!(best, members, rng)
    # The host state each route's kernel reads, as the keys it declares through
    # `watched_records`. A route that declares nothing draws from hazards fixed
    # for the run, so its contacts are never redrawn; the race watches the union
    # of the keys the live routes declare, and a key belongs to the routes that
    # declared it, which is how a record that moves reaches only the pairs whose
    # own kernel reads it.
    route_keys = _route_watch_keys(watches, length(rts))
    watched_keys = _watched_union(route_keys)
    live = !isempty(watched_keys)
    live_route = Bool[!isempty(keys) for keys in route_keys]
    key_routes = [
        [ri for ri in eachindex(route_keys) if key in route_keys[ri]]
            for key in watched_keys
    ]
    # What each watched key held on each member when contacts were last drawn
    # from it. A key can change type as well as value (a date that was absent
    # until a dose), so the store takes anything.
    snapshot = live ?
        Any[
            _remember(_watched_value(state.individuals[id], key))
            for key in watched_keys, id in members
        ] : Matrix{Any}(undef, 0, 0)

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
        id -> (
            ind = state.individuals[id];
            ind.susceptibility != 1 || ind.infectiousness != 1
        ),
        members
    )
    may_block = live || !isempty(risks) ||
        any(
        iv -> _has_own_method(
            competing_risk, typeof(iv),
            AbstractIntervention
        ), interventions
    )
    openings = _RouteOpening{T}[] # one per case and route it transmits along
    proposals = _Pending{T}[]  # every proposal made, when something can block
    head = zeros(Int, may_block ? m : 0)  # first proposal to each member
    represents = zeros(Int, m) # what each member's heap entry names
    pending = Tuple{T, Int, Int}[]

    # The seeds' own opening: no infector, so nothing about it is ever read.
    push!(openings, _RouteOpening(0, 0, zero(T), T(Inf)))
    # Proposals a redraw has unlinked, which stay in `proposals` and the heap
    # until they are compacted away.
    orphans = 0
    # The clock of the last pop, each pop at it as how many openings existed
    # then and the member position popped, and the largest such position. The
    # heap orders ties by position, so a pop at `clock` resolves every contact
    # at `clock` to that position or below from the openings existing then.
    clock = T(-Inf)
    pops = Tuple{Int, Int}[]
    passed = 0
    # For a live kernel, which hosts' records a pending or future draw reads.
    # Empty and unused otherwise.
    watch = _LiveWatch(live ? m : 0, length(rts))
    for k in 1:m
        best[k] < Inf || continue
        seeded = best[k]
        best[k] = T(Inf)
        _propose!(pending, proposals, head, best, represents, k, 1, seeded, may_block)
    end

    while !isempty(pending)
        bt, j, p = _heap_pop!(pending)
        # Pops come in increasing time, so every later infection also falls
        # after `max_time`: those individuals stay uninfected, and the race
        # was cut off rather than reaching extinction on its own.
        bt > max_time && return false
        if bt != clock
            clock = bt
            empty!(pops)
            passed = 0
        end
        push!(pops, (length(openings), j))
        passed = max(passed, j)
        may_block && (proposals[p] = _dequeue(proposals[p]))
        (processed[j] || represents[j] != p) && continue
        opening = openings[may_block ? proposals[p].opening : p]
        infector_id = opening.infector

        ind = state.individuals[members[j]]
        # A seeded time is a community introduction when the model gave the race
        # an `introduction`, and an index case otherwise. An introduction arrives
        # from outside the population, so it is put to the risks that act on the
        # person being introduced: their susceptibility, a vaccine's protection,
        # a risk of the model's own. Not to the risks that stand in for removing
        # an infector — isolation and quarantine —
        # since the source is outside the population and no measure taken here
        # removes it: being isolated is not protection from acquiring an
        # infection. The person stands in for the infector those risks read, so a
        # risk of your own that reads the infector should return nothing when the
        # two are the same individual. An index case is where an outbreak is
        # defined to start, and is put to no risk at all.
        source = infector_id == 0 ? ind : state.individuals[infector_id]
        if may_block && (infector_id != 0 || introduction !== nothing)
            blocked, permanent, certain_release = _proposal_blocked(
                state, source, ind, bt, risks,
                opening.route == 0 ? introduction_interventions :
                    route_interventions[opening.route]
            )
            if blocked
                # The contact did not transmit. Unless the block is certain to
                # recur (`permanent`), the source goes on meeting the person: the
                # next contact is a draw from the same hazard conditioned on
                # falling later, kept while the window is still open for it. A
                # permanent block already answers every later proposal the same
                # way, so the pair is dropped without asking the model or the
                # kernel again, unless `recorder` says this pair's draws still
                # matter, in which case it is asked afresh at every proposal a
                # standing block would otherwise end — not just this one — so a
                # recorder that wants the stream can log each one. Tracing and
                # ring construction read the standing contacts, not these
                # proposals, so the pair itself is unaffected either way.
                if permanent && !records_contacts(recorder, source, ind, state, bt)
                    proposals[p] = _at(proposals[p], T(Inf))
                else
                    if opening.route == 0
                        kernel, close_t = introduction
                        open_t = zero(T)
                        mult = ind.susceptibility
                    else
                        # Which route's targets to ask is known only at run time,
                        # so the routes are walked rather than indexed: indexing a
                        # tuple of routes with a running value would put the whole
                        # tuple on the heap, once per race.
                        kernel = _route_pair_kernel(
                            rts, opening.route, infector_id,
                            members[j], state
                        )
                        open_t = opening.open_t
                        close_t = opening.close_t
                        mult = source.infectiousness * ind.susceptibility
                    end
                    nxt = open_t + _next_contact(
                        rng, kernel, mult, bt - open_t, close_t - open_t,
                        certain_release === nothing ? nothing :
                            certain_release - open_t
                    )
                    proposals[p] = _at(proposals[p], nxt <= close_t ? nxt : T(Inf))
                end
                _requeue!(pending, proposals, head, best, represents, j)
                continue
            end
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
            # Only a model with named routes reports one. The single-window
            # shorthand has one window covering the whole model, so its name
            # identifies the model, where `:infection_route` is meant to
            # identify the setting a case was infected in.
            routes === nothing || (ind.state[:infection_route] = rts[opening.route][1].name)
        elseif introduction !== nothing
            ind.state[:infection_route] = :external
        end
        # The infection time is now fixed, so an intervention whose effect
        # depends on the exposure the race chose can settle it (see
        # `on_infection_settled!`), before the onset derived from it or any
        # transition reads it.
        for iv in interventions
            on_infection_settled!(iv, ind, state, rng)
        end
        # A pre-created node has no infection time, and so no onset, until now.
        # Derive the onset from the infection time before transitions and
        # interventions, such as onset-triggered isolation, read it.
        _set_onset_from_incubation!(ind)
        resolve_transitions!(state, ind)
        _resolve_interventions!(state, ind, interventions)
        traced = _trace_from!(state, ind, interventions, contacts, pos, processed)
        contacts === nothing ||
            _apply_continuous_actions!(
            state, ind, interventions, members, processed, contacts, pos, traced
        )
        traits |= ind.susceptibility != 1 || ind.infectiousness != 1

        # Only a live kernel whose host records actually moved needs its pending
        # contacts redrawn. Resolving a case usually leaves every record alone —
        # a policy applies to one case out of hundreds — and then the contacts
        # already drawn still come from the hazards in force, so the race takes
        # the ordinary path. Either way this case's own openings are drawn
        # inline below, from the records as they now stand.
        if live && _records_changed!(
                snapshot, watched_keys, key_routes, state, members,
                j, bt, watch, openings, processed
            )
            orphans += _redraw_moved!(
                pending, proposals, head, best, represents,
                watch, openings, processed, pos, rts, state, bt, pops, passed, rng
            )
            if orphans > max(m, length(proposals) ÷ 2)
                _compact_proposals!(pending, proposals, head, best, represents, processed)
                orphans = 0
            end
        end

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
            live && _watch_opening!(watch, j, live_route[ri])

            for (target_id, kernel) in route_targets(members[j], state)
                k = get(pos, target_id, 0)
                (k == 0 || processed[k]) && continue
                if live_route[ri]
                    # This opening's draws come from the target's record as it
                    # stands now, which is what later comparisons start from.
                    _watch_target!(watch, opening_id, k)
                    _refresh_host!(
                        snapshot, watched_keys, state.individuals[target_id], k
                    )
                end
                # Both per-individual traits are rate multipliers on this
                # pair's contact interval, folded into the draw rather than
                # resolved contact by contact. A pair at the default 1 draws
                # exactly as it did before they were honoured here.
                dt = traits ?
                    _traits_scaled_draw(
                        rng, kernel,
                        ind.infectiousness *
                        state.individuals[target_id].susceptibility
                    ) :
                    rand(rng, kernel)
                cand = open_t + dt
                cand <= close_t || continue
                _propose!(
                    pending, proposals, head, best, represents, k, opening_id,
                    cand, may_block
                )
            end
        end
    end
    return true
end

"""
    race_groups(model, kernel)

The races `model` runs its `_sellke_race!` construction over for `kernel`:
disjoint groups of population ids, each raced independently in its own call
with its own RNG stream. A model with more than one natural grouping —
`HouseholdProcess`, over its households — defines this to say how
many races it needs and which members fall in each, so the choice is the
model's own rather than inlined in whichever loop calls `_sellke_race!`
repeatedly. A new kernel type can override the method for a given model to
pick a different partition outright.
"""
function race_groups(model::TransmissionModel, kernel)
    throw(
        ArgumentError(
            "$(nameof(typeof(model))) needs a method for `EpiBranch.race_groups` " *
                "naming how it partitions its races for a kernel of this type"
        )
    )
end

# Keep a record to compare against later. A projection may hand back a mutable
# history that an intervention appends to in place, which would then compare
# equal to itself and hide the change, so anything that is not plain bits is
# copied. The usual named tuple of numbers is bits and is kept as it stands.
_remember(record) = isbits(record) ? record : deepcopy(record)

# A member's value of one watched key. An absent key reads as `_ABSENT`, which
# no state value can equal, so a key an intervention writes for the first time
# counts as a move.
_watched_value(ind::Individual, key::Symbol) = get(ind.state, key, _ABSENT)

function _refresh_host!(snapshot, watched_keys, ind::Individual, k)
    for (ki, key) in enumerate(watched_keys)
        snapshot[ki, k] = _remember(_watched_value(ind, key))
    end
    return nothing
end

# The keys each route's kernel declares. A model gives one tuple per route, in
# route order; `nothing` is a model with no live kernel at all.
function _route_watch_keys(watches, nroutes::Int)
    watches === nothing && return [() for _ in 1:nroutes]
    keys = [Tuple(w) for w in watches]
    length(keys) == nroutes || throw(
        ArgumentError(
            "`watches` must name the watched records of each of the $nroutes " *
                "routes (got $(length(keys)))"
        )
    )
    for ks in keys
        all(k -> k isa Symbol, ks) || throw(
            ArgumentError("`watches` must name `individual.state` keys as `Symbol`s")
        )
    end
    return keys
end

function _watched_union(route_keys)
    declared = Symbol[]
    for ks in route_keys, key in ks
        key in declared || push!(declared, key)
    end
    return declared
end

# The hosts whose records a pending or future draw of a live kernel reads: the
# infectors of openings still open and the unsettled members those openings
# reach. Only these are compared after a case settles, each once however many
# openings read it, so the check costs what is in play rather than the whole
# population.
struct _LiveWatch
    open::Vector{Int}              # openings still open
    source::Vector{Int}            # each opening's infector, by position
    reach::Vector{Vector{Int}}     # the members each opening reached when drawn
    as_infector::Vector{Int}       # open openings each member is the infector of
    as_target::Vector{Int}         # open openings that reached each member
    tracked::Vector{Int}           # members any open opening reads
    slot::Vector{Int}              # each member's place in `tracked`, or 0
    opened_by::Vector{Vector{Int}} # the openings each member made
    reached_by::Vector{Vector{Int}} # the openings that reached each member
    # Members whose records moved at this case, per route: a route hears only
    # about the keys its own kernel declared.
    moved::Vector{Vector{Int}}
end
# The seeds' opening has no infector and a fixed kernel, so it is never watched.
function _LiveWatch(m::Int, nroutes::Int)
    return _LiveWatch(
        Int[], [0], [Int[]], zeros(Int, m), zeros(Int, m), Int[], zeros(Int, m),
        [Int[] for _ in 1:m], [Int[] for _ in 1:m], [Int[] for _ in 1:nroutes]
    )
end

function _track!(w::_LiveWatch, k)
    w.slot[k] == 0 || return nothing
    push!(w.tracked, k)
    w.slot[k] = length(w.tracked)
    return nothing
end

function _untrack_at!(w::_LiveWatch, idx)
    k = w.tracked[idx]
    moved = pop!(w.tracked)
    if idx <= length(w.tracked)
        w.tracked[idx] = moved
        w.slot[moved] = idx
    end
    w.slot[k] = 0
    return nothing
end

# Every opening takes a slot, so an opening's position in `openings` is its
# position here. A route that watches nothing takes its slot and no more: its
# contacts are never redrawn, so nothing reads its hosts.
function _watch_opening!(w::_LiveWatch, infector, watched::Bool)
    push!(w.source, infector)
    push!(w.reach, Int[])
    watched || return nothing
    push!(w.open, length(w.reach))
    push!(w.opened_by[infector], length(w.reach))
    w.as_infector[infector] += 1
    return _track!(w, infector)
end

function _watch_target!(w::_LiveWatch, opening, k)
    push!(w.reach[opening], k)
    push!(w.reached_by[k], opening)
    w.as_target[k] += 1
    return _track!(w, k)
end

# Whether a host record that a pending or future draw reads has moved since
# contacts were last drawn from it, updating the remembered records as it goes
# and listing the members that moved in `w.moved`.
# The settled case's own record is brought up to date without counting as a
# move: contacts to it are settled, and its own contacts are drawn after this
# check.
function _records_changed!(
        snapshot, watched_keys, key_routes, state, members, case, now,
        w::_LiveWatch, openings, processed
    )
    _refresh_host!(snapshot, watched_keys, state.individuals[members[case]], case)
    kept = 0
    for oi in w.open
        if openings[oi].close_t >= now
            kept += 1
            w.open[kept] = oi
        else
            w.as_infector[w.source[oi]] -= 1
            for k in w.reach[oi]
                w.as_target[k] -= 1
            end
            empty!(w.reach[oi])
        end
    end
    resize!(w.open, kept)
    for list in w.moved
        empty!(list)
    end
    idx = 1
    any_moved = false
    while idx <= length(w.tracked)
        k = w.tracked[idx]
        if w.as_infector[k] == 0 && (processed[k] || w.as_target[k] == 0)
            _untrack_at!(w, idx)
            continue
        end
        ind = state.individuals[members[k]]
        for (ki, key) in enumerate(watched_keys)
            current = _watched_value(ind, key)
            isequal(snapshot[ki, k], current) && continue
            snapshot[ki, k] = _remember(current)
            any_moved = true
            # The member's keys are compared together, so a route it is already
            # listed for is listed once however many of its keys moved.
            for ri in key_routes[ki]
                list = w.moved[ri]
                (isempty(list) || last(list) != k) && push!(list, k)
            end
        end
        idx += 1
    end
    return any_moved
end

_link(p::_Pending, chain) = _Pending(p.opening, chain, p.time, p.queued)

# The largest member position popped at the current clock since opening `oi`
# was made. The number of openings only grows, so the pops since then are the
# last ones in `pops`.
function _passed_since(pops, passed, oi)
    isempty(pops) && return 0
    first(pops[1]) >= oi && return passed
    since = 0
    for k in length(pops):-1:1
        made, position = pops[k]
        made < oi && break
        since = max(since, position)
    end
    return since
end

# A pair's contact interval under its kernel scaled by the multiplier `m`, given
# that it exceeds `after`; infinite when no mass lies beyond.
function _draw_beyond(rng::AbstractRNG, kernel, m::Real, after)
    ls = logccdf(kernel, after)
    isfinite(ls) || return oftype(float(after), Inf)
    return _time_at_log_survival(kernel, ls + log(rand(rng)) / m)
end

# A record that moved changes only the pairs it enters: those of an open opening
# whose infector moved, and those reaching a moved member that has not settled.
# Their contacts are drawn again from the hazards now in force, conditioned on
# the exposure already elapsed, and every other proposal stands, external
# introductions included. A pair with no pending proposal is drawn again too:
# its earlier draw may have fallen past the window, which a new record can
# change. Fixed kernels never take this path. Returns how many proposals it
# unlinked.
function _redraw_moved!(
        pending, proposals, head, best, represents, w::_LiveWatch,
        openings, processed, pos, rts, state, now, pops, passed, rng
    )
    # Opening => the members whose pairs with it are drawn again, or `nothing`
    # for all of them.
    redo = Dict{Int, Union{Nothing, Set{Int}}}()
    for (ri, moved) in enumerate(w.moved)
        for k in moved
            for oi in w.opened_by[k]
                openings[oi].route == ri || continue
                openings[oi].close_t >= now && (redo[oi] = nothing)
            end
            processed[k] && continue
            for oi in w.reached_by[k]
                openings[oi].route == ri || continue
                openings[oi].close_t >= now || continue
                members_hit = get!(Set{Int}, redo, oi)
                members_hit === nothing || push!(members_hit, k)
            end
        end
    end
    redoes(oi, j) = haskey(redo, oi) && (redo[oi] === nothing || j in redo[oi])
    hit = Set{Int}()
    for (oi, members_hit) in redo
        for j in (members_hit === nothing ? w.reach[oi] : members_hit)
            processed[j] || push!(hit, j)
        end
    end
    # Unlink the proposals drawn again, keeping the rest in order.
    unlinked = 0
    for j in hit
        q = head[j]
        head[j] = 0
        last = 0
        while q != 0
            proposal = proposals[q]
            next_q = proposal.chain
            if !redoes(proposal.opening, j)
                last == 0 ? (head[j] = q) : (proposals[last] = _link(proposals[last], q))
                last = q
            else
                unlinked += 1
            end
            q = next_q
        end
        last == 0 || (proposals[last] = _link(proposals[last], 0))
        best[j] = oftype(best[j], Inf)
        represents[j] = 0
    end
    # A record change at this clock governs contacts at this clock too, so a
    # pair is drawn again given no contact strictly before `now`, and a contact
    # due at `now` under the new hazard, an atom there, stays due. That holds
    # only for pairs the race has not yet resolved at this clock: a pair whose
    # member sits at or below a position popped at `now` since its opening was
    # made has had its contact at `now` resolved, and is drawn given no contact
    # up to and including `now`. Contact times are stored as `open_t + dt`, and the exposure recomputed as `now - open_t`
    # can be off by the clock's resolution, so the draw starts `slack` early and
    # is then carried past any contact whose stored time falls before `now`.
    slack = 2 * eps(float(now))
    # In opening order, so that what the race draws depends on the records that
    # moved and not on the order the bookkeeping happened to visit them in.
    for oi in sort!(collect(keys(redo)))
        members_hit = redo[oi]
        opening = openings[oi]
        source = state.individuals[opening.infector]
        passed_since = _passed_since(pops, passed, oi)
        for (ri, (_, targets)) in enumerate(rts)
            ri == opening.route || continue
            for (id, kernel) in targets(opening.infector, state)
                j = get(pos, id, 0)
                (j == 0 || processed[j]) && continue
                members_hit === nothing || j in members_hit || continue
                m = source.infectiousness * state.individuals[id].susceptibility
                m <= 0 && continue
                lower = now - opening.open_t - slack
                dt = lower <= 0 ? _traits_scaled_draw(rng, kernel, m) :
                    _draw_beyond(rng, kernel, m, lower)
                # The next contact after one in the past is the same hazard
                # conditioned on falling later, so this draws exactly given no
                # contact before `now`, or none up to it once `now` is resolved.
                resolved = j <= passed_since
                while opening.open_t + dt < now || (resolved && opening.open_t + dt == now)
                    dt = _draw_beyond(rng, kernel, m, dt)
                end
                candidate = opening.open_t + dt
                candidate <= opening.close_t || continue
                _propose!(
                    pending, proposals, head, best, represents, j, oi, candidate, true
                )
            end
        end
    end
    # Each member's earliest proposal, kept or new, takes its place in the heap.
    for j in hit
        _requeue!(pending, proposals, head, best, represents, j)
    end
    return unlinked
end

# Rebuild `proposals` and the heap from the proposals still linked to members
# that have not settled, dropping those a redraw unlinked and those to members
# already settled. Each member keeps its proposals in order, and only its
# earliest goes back in the heap: an entry for any other is skipped when popped,
# and a blocked contact requeues the next earliest.
function _compact_proposals!(pending, proposals, head, best, represents, processed)
    kept = empty(proposals)
    empty!(pending)
    for j in eachindex(head)
        q = head[j]
        head[j] = 0
        representative = 0
        last = 0
        while !processed[j] && q != 0
            proposal = proposals[q]
            push!(kept, _Pending(proposal.opening, 0, proposal.time, false))
            id = length(kept)
            last == 0 ? (head[j] = id) : (kept[last] = _link(kept[last], id))
            q == represents[j] && (representative = id)
            last = id
            q = proposal.chain
        end
        represents[j] = representative
        if representative != 0
            kept[representative] = _queue(kept[representative])
            _heap_push!(pending, (best[j], j, representative))
        end
    end
    copy!(proposals, kept)
    return nothing
end
