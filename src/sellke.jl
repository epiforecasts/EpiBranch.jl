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
        _state_real(ind.state, Symbol(from, :_time), T(Inf))
end

# Earliest of the `until` removal states' times (Inf if none reached).
_window_close(ind::Individual, until::Tuple) = _close_at(ind, _time_keys(until))

# The state keys holding the times of `states`. Building a key concatenates two
# names, which costs more than reading it, so a loop that closes many windows on
# the same states builds the keys once.
_time_keys(states::Tuple) = map(s -> Symbol(s, :_time), states)

# Earliest of the times held under `keys` (Inf if none is set).
function _close_at(ind::Individual{T}, keys) where {T}
    t = T(Inf)
    for key in keys
        t = min(t, _state_real(ind.state, key, T(Inf)))
    end
    return t
end

# ── Interventions on the race ────────────────────────────────────────
# An intervention's effect on transmission runs through two seams. The first
# is the infectious window: an intervention that removes a case from onward
# transmission (isolation) shortens it. Each case is settled as the race pops
# it (`_settle_case!`): its natural history first, then each intervention's
# `resolve_individual!`, which is where `Isolation` writes its isolation time.
# Its window is then closed at the earliest removal across interventions as
# well as the `until` states. The second is the per-contact competing risk,
# resolved against each candidate infection time the race proposes (see the
# next block).

# ── Per-contact competing risks on the continuous-time path ──────────
#
# A competing risk blocks one infector → contact transmission with some
# probability, conditional on the risk having arrived before that transmission's
# time. On the race, a potential transmission *is* a drawn time: the contact
# interval from the infector's window opening to the moment it would infect that
# neighbour. So the risks are resolved against that time, when the candidate is
# popped rather than when it was proposed, so that a dose or an isolation that
# arrived in between is in force (see the loop in `_sellke_race!`).
#
# A blocked contact does not transmit, and the pair goes on meeting: its next
# contact is a draw from the same kernel conditioned on falling later, offered if
# the infector's window is still open for it. The points of a sequence drawn that
# way are the points of the kernel's own hazard `h(t)`, so blocking each of them
# with probability `p` thins that hazard to `(1-p)·h(t)` and the first contact
# that gets through arrives with survival `S(t)^(1-p)`. That is the per-exposure
# reading of a leaky vaccine and the rate-multiplier reading of a relative
# susceptibility. A degenerate kernel (`Dirac`) has one contact and no more, so a
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
# asked about every blocked proposal, permanent or not
# ([`records_contacts`](@ref EpiBranch.records_contacts)), so it can log each
# one; only its answer for a proposal whose block is permanent decides whether
# a standing block ends the pair's draws. Answering yes there keeps the race
# drawing exactly as it does for a block that is not certain, which puts the
# pair back under the rejection-continuation guard rather than needing one of
# its own. The default `NoContactRecorder` answers no to every pair, so a run
# with no recorder attached drops the pair exactly as above.
#
# It is not what a block means for a single exposure, a contact a branching
# process drew or one a model names through `contacts_of`, which is a
# transmission lost with nothing to follow it: an efficacy of 0.5 halves those
# transmissions, and for a kernel pair it leaves `1 - exp(-∫h/2)` of them.
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
        p isa Type || return false
        p !== base && p <: base
    end
end

# As above, but also requiring the method to be callable with exactly `n`
# arguments (the hook itself counted among them): a hook written for `T` with
# the wrong arity turns up under `methods` here, same as one with the right
# arity, but is never the method an exact-arity call resolves to, so it does
# not count. A trailing `args...` is handled separately, since it is reachable
# at any arity its minimum allows, rather than the one `methods` lists it
# under.
function _has_own_method(f, T::Type, base::Type, n::Int)
    return any(methods(f, Tuple{T, Vararg{Any}})) do mm
        params = Base.unwrap_unionall(mm.sig).parameters
        if Base.isvarargtype(params[end])
            # A trailing `Vararg` in a method signature is always open-ended:
            # Julia expands a counted one, so `f(x::Int, ::Vararg{Any, 2})`
            # has three fixed parameters and no vararg at all. Such a method
            # is reachable at any arity its fixed arguments allow.
            n + 1 >= length(params) - 1 || return false
        else
            length(params) == n + 1 || return false
        end
        p = params[2]
        p isa TypeVar && (p = p.ub)
        # A hook written with every argument variadic leaves a `Vararg` here
        # rather than a type, and says nothing about any one intervention.
        p isa Type || return false
        p !== base && p <: base
    end
end

# The intervention whose own method for `f` would actually run: an
# `InterventionWrapper` that does not override `f` only inherits its blanket
# forwarding to `w.intervention` (that method is declared on
# `InterventionWrapper` itself, not the concrete wrapper). Judging ownership
# against the wrapper's type would therefore credit it with a method it never
# wrote; unwrapping reaches the wrapped intervention's own method instead.
function _hook_owner(f, iv::AbstractIntervention)
    while iv isa InterventionWrapper && !_has_own_method(f, typeof(iv), InterventionWrapper)
        iv = iv.intervention
    end
    return iv
end

_innermost(iv::InterventionWrapper) = _innermost(iv.intervention)
_innermost(iv::AbstractIntervention) = iv

# Whether an intervention discovers actions of its own, which the race offers
# per settled case.
function _discovers_actions(iv::AbstractIntervention)
    return _has_own_method(intervention_actions, typeof(iv), AbstractIntervention)
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
function _propose!(queue, proposals, head, best, represents, k, w, t, may_block)
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
        _enqueue!(queue, _Event(t, k, pid))
    end
    return nothing
end

# After a member's earliest contact was blocked, or its proposals were redrawn,
# hand its place to the earliest remaining. Contacts at the same time go to the
# opening made first, as `_propose!` orders them, so the infector does not
# depend on the order the chain was built in. The chosen proposal may still have
# an entry in the heap, from before a later proposal overtook it, in which case
# there is nothing to push.
function _requeue!(queue, proposals, head, best, represents, k)
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
        _enqueue!(queue, _Event(t, k, pid))
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
function _intervention_removal_time(ind::Individual{T}, interventions) where {T}
    t = T(Inf)
    for iv in interventions
        t = min(t, convert(T, infectious_removal_time(iv, ind))::T)
    end
    return t
end

"""
    standing_block(source) -> Bool

For intervention authors: whether a block that is certain for an infector and
contact (probability 1) stays in force for good once it applies, for example
permanent immunity. When it does, a continuous-time simulation (households,
networks, `HomogeneousProcess`, `MixingProcess`) stops proposing further
contacts between the
pair. `false` by default.

The [`Risk`](@ref) a source returns cannot say this itself. `competing_risk`
reads the current state, so a block that is certain at one contact may have
lifted by the next (a ward that reopens, a campaign that ends, a quarantine
that expires), and a `Risk` holding plain numbers looks the same in both cases.
Declare `true` only for a source whose certain block, once in force for a pair,
never lifts. A simulation in which a case's infectious period may never end
(no `until` state that every case reaches) needs that declaration to finish;
without it, a certain block raises an `ArgumentError` instead of silently
dropping transmission that could still happen.
"""
standing_block(source) = false
standing_block(w::InterventionWrapper) = standing_block(w.intervention)
# An abort is recorded on the infector and never withdrawn, so the block it
# composes lasts as long as the infector does.
standing_block(::AbortedInfection) = true
# Whether this particular pair's block is standing is decided by
# `_standing_risk`, from the resolved `Risk`'s own `release_time`: infinite
# when the contact's progression never reaches `:susceptible_again`, and so
# genuinely permanent, finite and not standing otherwise. Declaring it here
# only opts `HostImmunity` into that per-pair check.
standing_block(::HostImmunity) = true
# A mode that disallows `waning` draws its block once at vaccination and it
# never fades, so once immunity has developed the block it composes is certain
# for good — which is what `supports_waning` reports. A mode that allows
# waning is not declared here even at `efficacy = 1.0`, because a later waning
# value could still give a smaller block to a later exposure.
standing_block(v::AbstractVaccination) = !supports_waning(effect_mode(v))

"""
    binding_release(component) -> Bool

For intervention authors: whether the `release_time` this intervention reports
on a [`Risk`](@ref) can be relied on. A block it says ends on day `t` is not
still in force after `t`, and a block it reports as never ending has not lifted
by the next contact. The default is `false`, the cautious answer.

A continuous-time simulation (households, networks, `HomogeneousProcess`,
`MixingProcess`) asks
this where a pair could go on having contacts indefinitely, to tell a block
that ends the pair's
transmission from one it must keep testing contacts against. Without the
declaration such a block raises an error instead of silently dropping
transmission that could still happen, for the reason
[`standing_block`](@ref EpiBranch.standing_block) gives.

The built-in isolation and quarantine declare it, because the periods
[`record_removal!`](@ref EpiBranch.record_removal!) records are only ever added
to. A [`Scheduled`](@ref) intervention that can end withdraws the declaration,
since ending it lifts a block before the release it reported.
"""
binding_release(component) = false
# `susceptible_again_time` is set once, when the contact's natural history
# resolves, and stands until their next infection; nothing moves it in
# between.
binding_release(::HostImmunity) = true

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
    release, ok = _certain_release(
        release, HostImmunity(), parent, contact, state, transmission_time
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
    _any_standing_risk(HostImmunity(), parent, contact, state, transmission_time) &&
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
# `transmission_time`, and whether that block is permanent (see
# `_permanently_blocked`). A risk may read the exposure from the contact's
# `infection_time`, but a proposed contact is still susceptible, so it holds
# nothing yet: set it to the proposed time for the resolution, and put it back
# if the contact is blocked, leaving it as it was.
function _proposal_blocked(
        state::SimulationState, parent, contact, transmission_time,
        model_risks, interventions, builtin_blocks = _sellke_builtin_risk_blocks
    )
    previous = contact.infection_time
    previous_clock = state.max_infection_time
    contact.infection_time = transmission_time
    # Scheduled risks use the proposed contact time even when earlier contacts
    # were blocked. Outside this evaluation the clock records accepted infections.
    state.max_infection_time = transmission_time
    blocked = _composed_risks_block(
        state, parent, contact, transmission_time,
        model_risks, interventions, builtin_blocks
    )
    permanent = blocked && _permanently_blocked(
        state, parent, contact, transmission_time, model_risks, interventions
    )
    release = blocked && !permanent ?
        _certain_block_release(
            state, parent, contact, transmission_time, model_risks, interventions
        ) : nothing
    state.max_infection_time = previous_clock
    blocked && (contact.infection_time = previous)
    return blocked, permanent, release
end

"""
    INTERVENTION_REMOVAL

List this in a [`RouteWindow`](@ref)'s `until` to let control measures end
transmission along that route: the route then stops when the model's
interventions (isolation, quarantine after tracing) remove the case.

Other states in `until` come from the natural history, so a route ends when
the case recovers, dies or is buried. Removal by an intervention is different
because whether it removes the case at all depends on the intervention:
perfect isolation takes the case out of transmission entirely, whereas leaky
isolation only reduces it. Listing `INTERVENTION_REMOVAL` asks each
intervention when it removes the case (through `infectious_removal_time`).

This is why routes are worth separating. A community route lists it, so
isolating a case ends their community transmission; a household route does
not, so the case goes on infecting the people they live with. The same applies
to the reduction from leaky isolation, but not to a vaccine's protection, which
applies on every route; see [`risk_applies`](@ref EpiBranch.risk_applies).
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
    return _route_close(ind, _time_keys(w.until), INTERVENTION_REMOVAL in w.until, interventions)
end

# The same from what the race reads off each route once: the keys of its
# `until` states and whether it is cut by intervention removal. A window's
# `until` is a tuple of any length, so reading it per case would leave the
# close time's type to be found at run time.
function _route_close(ind, keys, removable::Bool, interventions)
    t = _close_at(ind, keys)
    removable && (t = min(t, _intervention_removal_time(ind, interventions)))
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
# `Scheduled` delegates to its wrapped intervention. The loop exposes the running
# clock and count (see `_settle_case!`), which its time and count gates read
# wherever the wrapped intervention takes effect.

"""
    supplies_contacts(model) -> Bool

Whether a model knows each case's contacts (its offspring, household members
or network neighbours), so that contact tracing, and actions that need a
case's contacts, have people to act on (see
[`trace_contacts!`](@ref EpiBranch.trace_contacts!)). True for the branching,
household and network processes and for a model written against
[`generate_offspring`](@ref) or [`contacts_of`](@ref); false for the
mass-action models (`HomogeneousProcess`, `MixingProcess`), whose random
mixing has no list of contacts. A model with a simulation loop of its own
answers `false` unless it declares otherwise, and one returning `true` must
pass a `contacts` function to `_sellke_race!`.
"""
supplies_contacts(model::TransmissionModel) =
    !_has_own_method(simulate_once, typeof(model), TransmissionModel)

# Warn once per run about an intervention that cannot act on a model naming no
# contacts: one that traces, or proposes actions that need a case's contacts.
# What it does to each case itself, such as isolating it, still applies.
function _warn_contactless(model, interventions)
    supplies_contacts(model) && return nothing
    names = unique(
        String[
            string(nameof(typeof(_unwrap_scheduled(iv))))
                for iv in interventions
                if traces_contacts(iv) || (_proposes_actions(iv) && needs_contacts(iv))
        ]
    )
    isempty(names) && return nothing
    @warn "$(nameof(typeof(model))) names no contacts, so $(join(names, ", ")) " *
        "has no contacts to trace or act on. What it does to each case itself " *
        "still applies."
    return nothing
end

# Tracing on the continuous-time path. A case's trace time follows from its own
# timeline, so it can only be stamped once the race has finalised that case —
# which is also the point at which its contacts are known. A neighbour already
# settled, whether infected earlier or finally missed by every proposal that
# reached it, is offered to tracing all the same: its own infection (if any) is
# already final, but its pending outgoing proposals are not, and are checked
# against every intervention's `competing_risk` when they are popped, exactly
# as `_trace_source!` below relies on for a case's own infector. Quarantining a
# settled neighbour this way still blocks what it has not yet gone on to
# transmit, and lets the ring grow through it to its own neighbours in turn —
# the walk below does not stop at one just because it is already settled.
#
# Tracing *backwards*, to the case's own (already-final) infector, is
# `_trace_source!` (in simulation.jl), which `SourceTracing` answers
# `traces_source` for.
#
# A ring (`ContactTracing(depth = 2)` and beyond) grows through every traced
# contact's own neighbours, settled or not. The race walks the model's own
# contact structure breadth-first, and asks `keep_active` which contacts the
# next hop starts from. The depth semantics and the `:_ring_remaining` budget
# behind them stay with `ContactTracing`, so a ring of another shape takes part
# by answering `keep_active` rather than by writing a state key this loop would
# have to recognise.
#
# `visited` stops a node being traced twice by two branches of one walk
# reaching it at once. Across walks nothing is suppressed: a node reached as a
# ring member, which later becomes an infected and eligible case in its own
# right, seeds its own fresh full-radius ring when the race settles it, which
# is the contract `ContactTracing` documents. Whether that second attempt
# happens at all is the eligibility policy's call, not this loop's.
# `tracing` is whether any intervention traces, which the race works out once.
# `contacts` is typed so the call specialises on it: a closure passed through
# unspecialised is put on the heap at every call, which is every settled case.
function _trace_from!(
        state, infector, interventions, contacts::C, pos,
        tracing = any(traces_contacts, interventions)
    ) where {C}
    contacts === nothing && return nothing
    tracing || return nothing
    return _walk_ring!(state, infector, interventions, contacts, pos)
end

# The walk itself, which reports the members it offered to tracing. A ring
# wider than one hop reaches people the settled case does not neighbour, and
# the action layer has to be offered those too; which of them a tracing policy
# actually reached is its own business, and the layer reads that from them.
function _walk_ring!(
        state, infector, interventions, contacts::C, pos
    ) where {C}
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
            (k == 0 || cid in visited) && continue
            push!(pending, state.individuals[cid])
            push!(not_before, t0)
        end
        isempty(pending) && continue
        # `src` is the walk's own root only on the first pop; every later one
        # reached the frontier through `keep_active`, so an infected `src`
        # here is a ring member passed through, not a fresh case settling on
        # its own account. `_trace_pair!` reads this to trace from such a
        # `src` as a ring member only, rather than reseeding.
        hopped = src !== infector
        hopped && (src.state[:_ring_hop] = true)
        for iv in interventions
            if timed
                trace_contacts!(iv, state, src, pending, not_before)
            else
                trace_contacts!(iv, state, src, pending)
            end
        end
        hopped && delete!(src.state, :_ring_hop)
        for ind in pending
            push!(visited, ind.id)
        end
        # Which of them the ring grows from is the intervention's call. `is_new`
        # marks contacts a generation step created, and the race has no
        # generations, so it is all false.
        is_new = falses(length(pending))
        for iv in interventions
            for id in keep_active(iv, state, pending, is_new)
                k = get(pos, id, 0)
                k == 0 && continue
                push!(frontier, state.individuals[id])
            end
        end
    end
    delete!(visited, infector.id)
    return visited
end

"""
Whether `state.transitions` can ever set a finite `susceptible_again_time`,
the one way a continuous-time race is told a settled host becomes a target
again (see [`HostImmunity`](@ref EpiBranch.HostImmunity)). Read from
`state.transitions` rather than taken as a keyword, so every model built on
`_sellke_race!` gets reinfection for free the moment its `progression`
declares a `:susceptible_again` step, with no extra bookkeeping cost where it
does not.

A literal [`Transition`](@ref) is recognised directly from its `state`
field, which costs nothing and is exact regardless of its `probability`. Any
other `AbstractClinicalTransition` is probed instead: a disposable individual
is given `state.attributes`, to keep a `Transition` reading one of them from
failing on a missing key, then marked to make `_transition_selected` take
every `probability` gate it meets unless the probability is exactly zero; it
is run through every transition's `initialise_individual!` and
`resolve_individual!` on a copy of `state.rng`, and the result is read back
off `susceptible_again_time`. That marking keeps detection exact regardless
of probability on the custom-transition path too, instead of leaving it to
the single draw a real run would make.

The probe never runs a model's own per-individual `setup` (`HouseholdProcess`
sets `:household` there, for instance), so a custom transition that reads a
key only `setup` supplies raises a `KeyError` on the probe, even though a
real run always has it. Treating that as "reinfection cannot be ruled out",
rather than letting it abort the whole simulation, costs only the
bookkeeping that keeps a settled member's proposals pending: the real run
never hits this path, because the key is always set there.
"""
function _allows_reinfection(state)
    transitions = state.transitions
    any(t -> t isa Transition && t.state === :susceptible_again, transitions) && return true
    any(t -> !(t isa Transition), transitions) || return false
    probe = try
        _reinfection_probe(state)
    catch e
        e isa KeyError && return true
        rethrow()
    end
    return isfinite(susceptible_again_time(probe))
end

function _reinfection_probe(state)
    probe = Individual(; id = 0, infection_time = zero(_timetype(state)))
    probe.state[_REINFECTION_PROBE_KEY] = true
    saved_rng = state.rng
    state.rng = copy(saved_rng)
    try
        _apply_attributes!(state.attributes, state.rng, probe)
        for t in state.transitions
            initialise_individual!(t, probe, state)
        end
        for t in state.transitions
            resolve_individual!(t, probe, state)
        end
    finally
        state.rng = saved_rng
    end
    return probe
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

`projections` names each route's projection in the same order, as
[`kernel_projection`](@ref EpiBranch.kernel_projection) reports it, or
`nothing` for a route with no live kernel. A route with a projection is
tracked even when it watches nothing: after a case's interventions resolve,
the race re-projects every host a pending or future draw still reads and
compares the result against what it read last, throwing when it has changed
without any of the route's declared `watches` moving — the declaration is
checked rather than trusted. `nothing`, the default, is a model with no such
kernel at all.

`introduction`, when given, is the `(kernel, until)` of the community hazard the
model seeded its members from: the contact-interval distribution of an
introduction from outside the population, and the time the introduction window
closes. The kernel can also differ between people, as a vector indexed by
host id or a function of the host id, each giving a rate or a distribution,
as `external_hazard` takes them. It says that a seeded time is a community
introduction rather than an index case, so the risks that act on the person are resolved against it — a
vaccinated person is protected from the community as from a neighbour — and a
blocked introduction is followed by the next one from the same hazard. The risks
of isolation and quarantine are not: they
stand in for removing an infector, and an introduction's source is outside the
population. Omit `introduction` for a model whose seeds are index cases, which
are put to no risk at all. An introduced case's `:infection_route` is
`:external`, including on a model that also names routes.

A `progression` whose transitions can reach `:susceptible_again` (see
[`susceptible_again_time`](@ref)) lets a settled member become a target once
more: every proposal still pending for them, from a household-mate still
infectious or a repeated community introduction, is held back by
[`HostImmunity`](@ref EpiBranch.HostImmunity) exactly as a blocked contact is,
and goes through once their immunity has waned. The earlier infection is kept
on [`Individual.episodes`](@ref Individual) via [`close_episode!`](@ref). A
progression that never sets `susceptible_again_time` costs nothing extra: no
settled member is ever offered again.

`max_time` ends the race at that time: individuals whose infection would fall
later are left uninfected, and the state is exactly the full run's state
restricted to infections up to `max_time`. The exception is a mass-action route
(`_MassAction`) whose infectious window never closes, which samples its contacts
only up to `max_time`: there the cut run is a draw of the same process rather
than the same draw. Returns `true` when the race ran
until no candidate infection remained (the population reached extinction) and
`false` when it was cut off at `max_time` with candidates still pending.

`max_generation` drops a settled case's own proposals once its generation
reaches it, leaving every case beyond that depth undrawn; everything at or
within the bound still runs to its own extinction or `max_time`.
`stopping_rules` is polled with [`should_stop`](@ref) once a case settles, for
whatever a caller's `max_time`/`max_generation` do not already cover (see
[`honoured_without_should_stop`](@ref EpiBranch.honoured_without_should_stop));
the race ends there too, also returning `false`.

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
        introduction = nothing, watches = nothing, projections = nothing,
        max_time = Inf, max_generation = typemax(Int), stopping_rules = (),
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
    # Candidate times share the state's timing type (a dual under automatic
    # differentiation), so a gradient with respect to a transmission parameter
    # flows through the contact draws into the infection times.
    T = _timetype(state)
    buffers = _take_buffers(T)
    best = fill!(resize!(buffers.best, m), T(Inf))
    processed = fill!(resize!(buffers.processed, m), false)
    pos = _member_positions(members)
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
    close_keys = map(
        ((w, _),) -> Symbol[_time_keys(filter(!=(INTERVENTION_REMOVAL), w.until))...], rts
    )
    removable = map(((w, _),) -> INTERVENTION_REMOVAL in w.until, rts)

    # An intervention declaring `acts_on_population` needs no case settled to
    # propose its actions, so it is offered every member before the race
    # starts rather than case by case as the race goes.
    _apply_population_actions!(state, interventions, _population_members(members))
    seed!(best, members, rng)
    # The host state each route's kernel reads, as the keys it declares through
    # `watched_records`. A route that declares nothing draws from hazards fixed
    # for the run, so its contacts are never redrawn; the race watches the union
    # of the keys the live routes declare, and a key belongs to the routes that
    # declared it, which is how a record that moves reaches only the pairs whose
    # own kernel reads it.
    (;
        watch, snapshot, watched_keys, key_routes, route_projections,
        full_snapshot, live_route, live,
    ) = _race_watch(state, members, rts, watches, projections)
    route_names = Tuple(w.name for (w, _) in rts)

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
    # A settled member only ever becomes a target again by going through the
    # same blocked-contact redraw every other risk uses (see `reinfects`
    # below), which needs every proposal kept rather than just each member's
    # earliest. Only members that can be reached more than once can be
    # reinfected: a contact created as it is proposed has a single infector.
    reinfects = _proposal_fallback(members) && _allows_reinfection(state)
    # Lists retain alternatives after a block. A live kernel can redraw a
    # contact, and an introduction can recur even for single-exposure members.
    # A route that checks its own built-in risks (`_route_may_block`) can block
    # a member that another case may still reach.
    may_block = live || reinfects || (
        (introduction !== nothing || _proposal_fallback(members)) && (
            !isempty(risks) || any(
                iv -> _has_own_method(
                    competing_risk, typeof(iv),
                    AbstractIntervention
                ), interventions
            )
        )
    ) || (_proposal_fallback(members) && any(((_, rt),) -> _route_may_block(rt), rts))
    # A route whose contacts the race checks against built-in risks of its own
    # (`_route_may_block`) checks every proposal too. Where its members cannot
    # fall back (`_Offspring`), it proposes each member once, so a blocked
    # contact has nothing to fall back on and needs no lists.
    checks = may_block || any(((_, rt),) -> _route_may_block(rt), rts)
    tracing = any(traces_contacts, interventions)
    tracing_source = any(traces_source, interventions)
    # Whether `_apply_case_actions!` is worth calling on a model that
    # names no contacts: an intervention declaring `needs_contacts = false`
    # (`GroupVaccination`) still has candidates to offer it there.
    acting = filter(_proposes_actions, interventions)
    # Those a person a route creates as it goes is offered on creation.
    creating = filter(iv -> !acts_on_population(iv) && !needs_contacts(iv), acting)
    contactless_actions = any(iv -> !needs_contacts(iv), acting)
    openings = empty!(buffers.openings) # one per case and route it transmits along
    proposals = empty!(buffers.proposals) # proposals retained for fallback
    head = fill!(resize!(buffers.head, may_block ? m : 0), 0) # first proposal to each member
    represents = fill!(resize!(buffers.represents, m), 0) # what each member's heap entry names
    queue = empty!(buffers.queue)

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
    # Whether `max_generation` has suppressed a case's own transmission, so
    # the run finishing with no candidates left is reported as cut off rather
    # than as having reached extinction on its own.
    capped_generation = false
    # What a route needs to propose from an opening (see `_propose_along!`).
    # Every field is bound once and only mutated in place.
    race = (;
        queue, proposals, head, best, represents, processed, pos, state, rng, members,
        watch, snapshot, watched_keys, may_block, max_time, reinfects,
        route_projections, full_snapshot, interventions, creating, buffers,
    )
    # What tracing and the interventions' actions read as a case's contacts. A
    # route that creates contacts as it goes creates a ring member's when the
    # ring reaches it.
    case_contacts = _case_contacts(contacts, rts, race, max_generation)
    for k in 1:m
        best[k] < Inf || continue
        seeded = best[k]
        best[k] = T(Inf)
        _propose!(race, k, 1, seeded)
    end

    while !isempty(queue)
        event = _dequeue!(queue)
        bt, j, p = event.time, event.member, event.ref
        # Pops come in increasing time, so every later infection also falls
        # after `max_time`: those individuals stay uninfected, and the race
        # was cut off rather than reaching extinction on its own.
        bt > max_time && return _finish_race!(rts, race, false, max_time)
        # Only a redraw of a live kernel reads which pops happened at this clock.
        if live
            if bt != clock
                clock = bt
                empty!(pops)
                passed = 0
            end
            push!(pops, (length(openings), j))
            passed = max(passed, j)
        end
        may_block && (proposals[p] = _dequeue(proposals[p]))
        _stale_proposal(members, processed, represents, j, p) && continue
        opening = _event_opening(members, event, openings, proposals, may_block)
        infector_id = opening.infector

        ind = _member!(members, race, j)
        # Set once the progression can reach `:susceptible_again`: `ind` has
        # already settled an earlier episode, so this candidate is a
        # reinfection if `HostImmunity` lets it through below, rather than a
        # first infection. `_proposal_blocked` resolves a risk against `ind`
        # as the candidate it is testing, leaving `ind.infection_time` at `bt`
        # once it decides not to block; the prior episode's own time is saved
        # here so it survives that to reach `InfectionEpisode` below.
        reinfection = processed[j]
        prior_infection_time = reinfection ? ind.infection_time : zero(T)
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
        if checks && (infector_id != 0 || introduction !== nothing)
            blocked, permanent, certain_release = _proposal_blocked(
                state, source, ind, bt, risks,
                opening.route == 0 ? introduction_interventions :
                    route_interventions[opening.route],
                opening.route == 0 ? _sellke_builtin_risk_blocks :
                    _on_route((_, rt) -> _route_builtin_blocks(rt), rts, opening.route)
            )
            if blocked
                # The contact did not transmit. `recorder` is asked about every
                # blocked proposal, permanent or not, so it can log the
                # exposure even when the pair goes on meeting; only its answer
                # for a permanent block decides whether the pair's draws
                # continue. Unless the block is certain to recur (`permanent`),
                # the source goes on meeting the person: the next contact is a
                # draw from the same hazard conditioned on falling later, kept
                # while the window is still open for it. A permanent block
                # already answers every later proposal the same way, so the
                # pair is dropped without asking the model or the kernel again,
                # unless `recorder` says this pair's draws still matter, in
                # which case it is asked afresh at every later proposal too, so
                # a recorder that wants the stream can log each one. Tracing
                # and ring construction read the standing contacts, not these
                # proposals, so the pair itself is unaffected either way.
                keeps_drawing = _logs_blocked(members, source, ind) &&
                    records_contacts(recorder, source, ind, state, bt)
                if !may_block
                    _discard_proposal!(members, best, j)
                    continue
                end
                if permanent && !keeps_drawing
                    proposals[p] = _at(proposals[p], T(Inf))
                else
                    if opening.route == 0
                        source_ext, close_t = introduction
                        kernel = _ext_survival_of(source_ext, members[j])
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
                        stream(rng, (infector_id, members[j]), (:contact, opening.route)),
                        kernel, mult, bt - open_t, close_t - open_t,
                        certain_release === nothing ? nothing :
                            certain_release - open_t
                    )
                    proposals[p] = _at(proposals[p], nxt <= close_t ? nxt : T(Inf))
                end
                _requeue!(queue, proposals, head, best, represents, j)
                continue
            end
        end
        processed[j] = true
        _settled_proposal!(members, j)
        if reinfects
            # Anyone else still meeting `ind` — a household-mate whose own
            # proposal to them lost this race — gets a chance to redraw
            # towards `ind`'s next `susceptible_again_time` rather than being
            # abandoned: the entry this candidate used is retired, and the
            # chain's next earliest, if any, takes its place in the heap.
            proposals[p] = _at(proposals[p], T(Inf))
            _requeue!(queue, proposals, head, best, represents, j)
            # The community hazard made its one introduction attempt at the
            # seed, before anyone was settled; a settled member needs a fresh
            # one armed for them, on the same memoryless schedule, to be
            # reachable again after this episode.
            if introduction !== nothing
                source_ext, close_t = introduction
                kernel = _ext_survival_of(source_ext, members[j])
                nxt = _next_contact(rng, kernel, ind.susceptibility, bt, close_t)
                nxt <= close_t &&
                    _propose!(queue, proposals, head, best, represents, j, 1, nxt, true)
            end
        end
        if reinfection
            ind.infection_time = prior_infection_time
            close_episode!(ind, InfectionEpisode(ind))
            # `ind` is the case being processed again, so `is_settled` must
            # go back to false until this new episode's own call to
            # `_apply_case_actions!` below marks it settled afresh.
            ind.state[:_settled] = false
        end

        ind.infection_time = bt
        ind.state[:infected] = true
        ind.state[:index] = infector_id == 0
        if infector_id != 0
            infector = state.individuals[infector_id]
            ind.parent_id = infector.id
            ind.generation = infector.generation + 1
            ind.chain_id = infector.chain_id
            # This pair may have been logged earlier from a non-permanent
            # block that `recorder` was asked to keep drawing; it infected
            # after all, so that logged exposure is retracted.
            _retract_contact!(recorder, infector, ind, state)
            # Only a model with named routes reports one. The single-window
            # shorthand has one window covering the whole model, so its name
            # identifies the model, where `:infection_route` is meant to
            # identify the setting a case was infected in.
            _on_route(
                (w, rt) -> _record_infection!(rt, infector, ind, w, routes !== nothing),
                rts, opening.route
            )
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
        # Cases are popped in increasing infection time: this case's infection
        # time is the clock, and it is the next case in sequence. The post-loop
        # reconcile (`_reconcile_sellke_bookkeeping!`) sets the final values.
        _settle_case!(state, ind, interventions, state.cumulative_cases + 1)
        # `_settle_case!` has just brought `state.cumulative_cases`,
        # `max_infection_time` and `current_generation` up to this case, which
        # is what a rule not already captured by `max_time`/`max_generation`
        # reads; ending here rather than after this case transmits matches
        # `MaxCases` stopping at the first `n` infections in time.
        _any_stop(stopping_rules, state) && return _finish_race!(rts, race, false, bt)
        # A route that creates its contacts as it goes creates this case's now,
        # so that tracing and the interventions' actions can reach them.
        # Unrolled over `rts` rather than looped over (see `_mint_all!`), for
        # the reason given there.
        _mint_all!(rts, race, ind, max_generation)
        traced = _trace_from!(
            state, ind, interventions, case_contacts, pos, tracing
        )
        _trace_source!(state, ind, interventions, tracing_source)
        (contacts !== nothing || contactless_actions) &&
            _apply_case_actions!(
            state, ind, acting, members, processed, case_contacts, pos, traced
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
                j, bt, watch, openings, processed,
                full_snapshot, route_projections, route_names
            )
            orphans += _redraw_moved!(
                queue, proposals, head, best, represents,
                watch, openings, processed, pos, rts, state, bt, pops, passed, rng,
                reinfects
            )
            if orphans > max(m, length(proposals) ÷ 2)
                _compact_proposals!(queue, proposals, head, best, represents, processed, reinfects)
                orphans = 0
            end
        end

        # Each route opens and closes on its own states, so a case can still be
        # transmitting on one while another has been cut. A route whose `from`
        # state was never reached contributes nothing, which is how a survivor
        # never materialises funeral contacts. Unrolled over `rts` rather than
        # looped over (see `_open_routes!`), for the reason given there.
        capped_generation = _open_routes!(
            rts,
            (;
                race, ind, members, openings, j, max_generation, close_keys,
                removable, watch, live, live_route, route_projections,
                full_snapshot, traits, interventions, T,
            ),
            capped_generation
        )
    end
    return _finish_race!(
        rts, race,
        !capped_generation &&
            !any(((_, route_targets),) -> _pending_beyond(route_targets, race), rts),
        race.max_time
    )
end

# Let each route tidy up what the race left undecided when it ended at
# `stop_t`, then report whether the race reached extinction.
function _finish_race!(rts, race, extinct, stop_t = Inf)
    for (_, route_targets) in rts
        _finish!(route_targets, race, stop_t)
    end
    _return_buffers!(race.buffers)
    return extinct
end
_finish!(route_targets, race, stop_t) = nothing

# Whether any of the stopping rules ends the run now.
function _any_stop(rules, state)
    for rule in rules
        _should_stop(rule, state) && return true
    end
    return false
end

# What a race grows as it runs, kept by each task for its next race. A large
# outbreak grows these to hundreds of thousands of entries, and growing them
# afresh on every run allocates enough to set off a garbage collection that
# walks every person of the outbreak, several times a run. A race that throws
# leaves its buffers to be collected, and the next race starts new ones.
struct _RaceBuffers{T <: Real}
    queue::_EventQueue{T}
    best::Vector{T}
    processed::BitVector
    represents::Vector{Int}
    head::Vector{Int}
    openings::Vector{_RouteOpening{T}}
    proposals::Vector{_Pending{T}}
end
function _RaceBuffers{T}() where {T}
    return _RaceBuffers{T}(
        _EventQueue{T}(), T[], falses(0), Int[], Int[], _RouteOpening{T}[], _Pending{T}[]
    )
end

_take_buffers(::Type{T}) where {T} = _take_pooled!(_RaceBuffers{T})
_return_buffers!(b::_RaceBuffers) = _return_pooled!(b)

# One of this task's spare objects of type `B`, or a new one. Each race takes
# its own, so two races never share one.
function _take_pooled!(::Type{B}) where {B}
    pool = get!(() -> B[], task_local_storage(), B)::Vector{B}
    return isempty(pool) ? B() : pop!(pool)
end
function _return_pooled!(b::B) where {B}
    pool = get(task_local_storage(), B, nothing)
    pool === nothing || push!(pool::Vector{B}, b)
    return nothing
end

# Fixed-population members can have competing proposals and retain route openings.
_discard_proposal!(members, best, j) = (best[j] = oftype(best[j], Inf))
_settled_proposal!(members, j) = nothing
_stale_proposal(members, processed, represents, j, p) = represents[j] != p
_event_opening(members, event, openings, proposals, may_block) =
    openings[may_block ? proposals[event.ref].opening : event.ref]
function _store_opening!(members, openings, infector, route, open_t, close_t)
    push!(openings, _RouteOpening(infector, route, open_t, close_t))
    return length(openings)
end

# Whether a blocked contact can have another proposal to try. Members in a
# fixed population can be reached along several edges or repeated contacts.
_proposal_fallback(members) = true
# The members offered actions for the whole population before the race starts.
_population_members(members) = members
# Whether a blocked proposal from `source` to `ind` is offered to the recorder.
# Members created as contacts are already listed as their infector's contacts,
# blocked or not.
_logs_blocked(members, source, ind) = true

# Each member's position in `members`, by id.
_member_positions(members) = Dict{Int, Int}(id => k for (k, id) in enumerate(members))

# Call `f(window, route_targets)` on the route numbered `route`. A tuple of routes
# is walked rather than indexed, since indexing it with a running value would
# put the whole tuple on the heap.
_on_route(f::F, rts::AbstractVector, route) where {F} = f(rts[route]...)
_on_route(f::F, ::Tuple{}, route, ri = 1) where {F} = throw(BoundsError())
function _on_route(f::F, rts::Tuple, route, ri = 1) where {F}
    ri == route && return f(first(rts)...)
    return _on_route(f, Base.tail(rts), route, ri + 1)
end

# Call `_mint!`, which draws a route's own contacts for a case as it settles,
# on every route rather than one at a time. A generated function unrolls over
# `rts`'s literal length instead of looping over it or recursing through
# `Base.tail`: a `for` loop over routes of more than one type (a different
# offspring law, kernel or `until`, say) reads every visit through their
# common supertype and boxes it, and recursion deep enough to cover a few
# dozen routes defeats the compiler's inlining budget and falls back to that
# same boxing. Code indexing each route by its literal position avoids both,
# specialising every visit on its own route's concrete type regardless of how
# many routes there are or how many share a type.
@generated function _mint_all!(rts::Tuple, race, ind, max_generation)
    body = Expr(:block)
    for i in 1:length(rts.parameters)
        push!(body.args, :(_mint!(rts[$i][2], race, ind, max_generation)))
    end
    push!(body.args, nothing)
    return body
end

# Open every route a settling case reaches and offer its targets a proposal,
# unrolled over `rts`'s literal length for the same reason `_mint_all!` is.
# `ctx` bundles what every route reads alike, and `capped` threads the "cut
# off by `max_generation`" flag through the unrolled calls.
@generated function _open_routes!(rts::Tuple, ctx, capped)
    body = Expr(:block)
    for i in 1:length(rts.parameters)
        push!(body.args, :(capped = _open_route!(rts[$i][1], rts[$i][2], $i, ctx, capped)))
    end
    push!(body.args, :(return capped))
    return body
end

function _open_route!(w, route_targets, ri, ctx, capped)
    (;
        race, ind, members, openings, j, max_generation, close_keys, removable,
        watch, live, live_route, route_projections, full_snapshot, traits,
        interventions, T,
    ) = ctx
    open_t = _route_open(route_targets, ind, w)
    isfinite(open_t) || return capped
    # A case at the generation bound opened this route and would otherwise
    # have transmitted along it, so the run was cut off rather than reaching
    # extinction on its own — the same distinction `max_time` draws for a
    # route it censors.
    ind.generation >= max_generation && return true
    close_t = convert(T, _route_close(ind, close_keys[ri], removable[ri], interventions))
    opening_id = _store_opening!(members, openings, members[j], ri, open_t, close_t)
    live && _watch_opening!(watch, j, live_route[ri])
    if live_route[ri]
        proj = route_projections[ri]
        proj === nothing || (full_snapshot[ri][j] = _remember(proj(ind)))
    end
    _propose_along!(
        route_targets, race, members[j], opening_id, open_t, close_t,
        traits, live_route[ri], ri
    )
    return capped
end

# What a route needs beyond proposing along its pairs. A route over existing
# members has nothing to create, opens on its window's `from` state, records
# the case with its infector and the route's name, and needs no built-in risk
# but an aborted infection, since the per-individual traits are already in its
# draws. A route that creates contacts as it goes (`_Offspring`) answers each
# of these differently.
_mint!(route_targets, race, ind, max_generation) = nothing
_member!(members, race, k) = race.state.individuals[members[k]]
_case_contacts(contacts, rts, race, max_generation) = contacts
_route_open(route_targets, ind, w::RouteWindow) = window_open(ind, w)
_route_may_block(route_targets) = false
_route_builtin_blocks(route_targets) = _sellke_builtin_risk_blocks
function _record_infection!(route_targets, infector, ind, w, named)
    push!(infector.secondary_case_ids, ind.id)
    named && (ind.state[:infection_route] = w.name)
    return nothing
end

# Whether a route left contacts past the time limit that it never proposed, so
# that a race which ran out of candidates was cut off rather than extinct. A
# route that proposes every contact it draws leaves none.
_pending_beyond(route_targets, race) = false

# Propose the infections one case's window on one route leads to. The default
# asks the route for every susceptible target and its pair kernel, and draws one
# contact per pair, keeping those inside the window. A route that can find the
# pairs with a contact inside the window without drawing every pair, such as
# mass action (`_MassAction`), has a method of its own.
function _propose_along!(
        route_targets, race, infector_id, opening_id, open_t, close_t, traits, live, ri
    )
    (;
        state, rng, pos, processed, watch, snapshot, watched_keys,
        route_projections, full_snapshot, reinfects,
    ) = race
    infector = state.individuals[infector_id]
    proj = route_projections[ri]
    for (target_id, kernel) in route_targets(infector_id, state)
        k = get(pos, target_id, 0)
        # A settled target still draws a proposal when the progression allows
        # reinfection: `HostImmunity`, on the pop, blocks it until the target's
        # immunity has waned.
        (k == 0 || (processed[k] && !reinfects)) && continue
        if live
            # This opening's draws come from the target's record as it
            # stands now, which is what later comparisons start from.
            _watch_target!(watch, opening_id, k)
            _refresh_host!(
                snapshot, watched_keys, state.individuals[target_id], k
            )
            proj === nothing || (
                full_snapshot[ri][k] = _remember(proj(state.individuals[target_id]))
            )
        end
        # Both per-individual traits are rate multipliers on this
        # pair's contact interval, folded into the draw rather than
        # resolved contact by contact. A pair at the default 1 draws
        # exactly as it did before they were honoured here.
        pair_rng = stream(rng, (infector_id, target_id), (:contact, ri))
        dt = traits ?
            _traits_scaled_draw(
                pair_rng, kernel,
                infector.infectiousness *
                state.individuals[target_id].susceptibility
            ) :
            rand(pair_rng, kernel)
        cand = open_t + dt
        cand <= close_t || continue
        _propose!(race, k, opening_id, cand)
    end
    return nothing
end

# Record a proposal through the race's own queues.
_propose!(race, k, opening_id, t) = _propose_member!(race.members, race, k, opening_id, t)

function _propose_member!(members, race, k, opening_id, t)
    return _propose!(
        race.queue, race.proposals, race.head, race.best, race.represents,
        k, opening_id, t, race.may_block
    )
end

"""
    sellke_race!(state, members, rng, sim_opts; kwargs...) -> state

Public wrapper around `_sellke_race!`: resolves the race's
`max_time`, `max_generation` and `stopping_rules` from `sim_opts`'s stopping
rules, then reconciles `state`'s aggregate bookkeeping (`cumulative_cases`,
`max_infection_time`, `extinct`) from the per-individual state the race wrote
directly, before returning `state`. A model with its own simulation loop
calls this rather than `_sellke_race!` itself, so it never has to resolve
these or reconcile the bookkeeping by hand. `kwargs` are `_sellke_race!`'s own
(`seed!`, `targets` or `routes`, `from`, `until`, `interventions`, `contacts`,
`risks`, `introduction`, `watches`, `recorder`), everything but the three this
wrapper sets.

A model with more than one natural race partition (a household process, over
its households, via [`race_groups`](@ref)) calls `_sellke_race!` directly in
its own loop instead, with `max_time`, `max_generation` and `stopping_rules`
resolved once outside it, and reconciles `state` once after the loop: this
wrapper's per-call reconciliation scans every individual, which would cost
O(races × population) run inside such a loop rather than the O(population) a
single-race model gets from calling this once. See `HouseholdProcess` in the
`EpiHouseholds` package.
"""
function sellke_race!(
        state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG, sim_opts::SimOpts; kwargs...
    )
    extinct = _sellke_race!(
        state, members, rng; max_time = _max_time(sim_opts),
        max_generation = _max_generation(sim_opts),
        stopping_rules = _poll_rules(sim_opts), kwargs...
    )
    return _reconcile_sellke_bookkeeping!(state, extinct)
end

"""
    race_groups(model, kernel)

For model authors: the groups of people a continuous-time model simulates
separately, as disjoint groups of population ids, each simulated in its own
call to `_sellke_race!` with its own random number stream. `HouseholdProcess`
defines this to simulate each household on its own. A new kernel type can
define a method for a given model to choose a different grouping.
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

# A race with fixed kernels has no host records to watch. Dispatching on
# absent declarations keeps the unused watch buffers and snapshots out of setup.
function _race_watch(state, members, rts, ::Nothing, ::Nothing)
    return (;
        watch = nothing, snapshot = nothing, watched_keys = (), key_routes = (),
        route_projections = map(_ -> nothing, rts), full_snapshot = nothing,
        live_route = map(_ -> false, rts), live = false,
    )
end

function _race_watch(state, members, rts, watches, projections)
    m = length(members)
    route_keys = _route_watch_keys(watches, length(rts))
    watched_keys = _watched_union(route_keys)
    # A route with a live projection is tracked even when it watches nothing:
    # declaring no key is a claim that nothing can move its hazard, and the
    # race holds the kernel to that claim (`_check_undeclared_read!`) rather
    # than drawing from it unchecked.
    route_projections = _route_projections(projections, length(rts))
    live_route = Bool[
        !isempty(route_keys[ri]) || route_projections[ri] !== nothing
            for ri in eachindex(route_keys)
    ]
    live = any(live_route)
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
    # The full record each route's projection read last from a host with a
    # pending or future draw, kept only for a route that has one.
    full_snapshot = [
        proj === nothing ? Any[] : Any[nothing for _ in 1:m] for proj in route_projections
    ]

    watch = _LiveWatch(live ? m : 0, length(rts))
    return (;
        watch, snapshot, watched_keys, key_routes, route_projections,
        full_snapshot, live_route, live,
    )
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

# The projection each route's kernel reads through, one per route in route
# order, or `nothing` for a route with no live kernel. Mirrors
# `_route_watch_keys`, since the race checks a kernel's `watches` against what
# its own projection actually reads.
function _route_projections(projections, nroutes::Int)
    projections === nothing && return fill(nothing, nroutes)
    projs = collect(projections)
    length(projs) == nroutes || throw(
        ArgumentError(
            "`projections` must name the projection of each of the $nroutes " *
                "routes (got $(length(projs)))"
        )
    )
    return projs
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
# position here. A route neither watching a key nor holding a projection
# takes its slot and no more: its contacts are never redrawn and nothing of
# its hosts is read.
# Absent watch declarations have no opening records to retain.
_watch_opening!(::Nothing, infector, watched::Bool) = nothing
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
        w::_LiveWatch, openings, processed,
        full_snapshot = nothing, route_projections = nothing, route_names = nothing
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
        # The routes this host's own moved keys already explain, so the
        # undeclared-read check below compares only a route none of them
        # cover.
        host_moved = Int[]
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
                ri in host_moved || push!(host_moved, ri)
            end
        end
        _check_undeclared_read!(
            full_snapshot, route_projections, route_names, w, openings,
            ind, k, now, processed, host_moved
        )
        idx += 1
    end
    return any_moved
end

# The live routes currently reading `k`'s record: those of an open opening it
# is the infector of, and, while it has not settled, those of an open opening
# that reached it. A route not in this list has nothing of its projection to
# compare here.
function _live_routes(w::_LiveWatch, openings, k, now, processed)
    routes = Int[]
    for oi in w.opened_by[k]
        openings[oi].close_t >= now || continue
        openings[oi].route in routes || push!(routes, openings[oi].route)
    end
    if !processed[k]
        for oi in w.reached_by[k]
            openings[oi].close_t >= now || continue
            openings[oi].route in routes || push!(routes, openings[oi].route)
        end
    end
    return routes
end

# A kernel's `watches` is a claim that nothing else its projection reads can
# move. Re-project every live route still reading `k` and compare against
# what it read last; a route whose own declared keys (`host_moved`) already
# cover `k` is skipped, since it redraws regardless, but a mismatch on any
# other route means the claim was false, and the race throws naming it rather
# than drawing from a hazard that went stale unnoticed.
function _check_undeclared_read!(
        full_snapshot, route_projections, route_names, w::_LiveWatch, openings,
        ind, k, now, processed, host_moved
    )
    route_projections === nothing && return nothing
    for ri in _live_routes(w, openings, k, now, processed)
        proj = route_projections[ri]
        proj === nothing && continue
        # Resynced on every pass, whether this route redraws `k` or not, so a
        # later pass compares against what the route last read rather than
        # what it read when it first drew from this host.
        current = _remember(proj(ind))
        previous = full_snapshot[ri][k]
        full_snapshot[ri][k] = current
        (ri in host_moved || isequal(previous, current)) && continue
        on_route = route_names === nothing ? "" : " on route :$(route_names[ri])"
        throw(
            ArgumentError(
                "a PairKernel$on_route reads a record that changed although " *
                    "none of its declared `watches` did; declare the key it reads"
            )
        )
    end
    return nothing
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
# whose infector moved, and those reaching a moved member that has not settled,
# or that has but can still be reinfected (see `reinfects` in `_propose_along!`).
# Their contacts are drawn again from the hazards now in force, conditioned on
# the exposure already elapsed, and every other proposal stands, external
# introductions included. A pair with no pending proposal is drawn again too:
# its earlier draw may have fallen past the window, which a new record can
# change. Fixed kernels never take this path. Returns how many proposals it
# unlinked.
function _redraw_moved!(
        queue, proposals, head, best, represents, w::_LiveWatch,
        openings, processed, pos, rts, state, now, pops, passed, rng, reinfects
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
            processed[k] && !reinfects && continue
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
            (processed[j] && !reinfects) || push!(hit, j)
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
                (j == 0 || (processed[j] && !reinfects)) && continue
                members_hit === nothing || j in members_hit || continue
                m = source.infectiousness * state.individuals[id].susceptibility
                m <= 0 && continue
                pair_rng = stream(rng, (opening.infector, id), (:contact, opening.route))
                lower = now - opening.open_t - slack
                dt = lower <= 0 ? _traits_scaled_draw(pair_rng, kernel, m) :
                    _draw_beyond(pair_rng, kernel, m, lower)
                # The next contact after one in the past is the same hazard
                # conditioned on falling later, so this draws exactly given no
                # contact before `now`, or none up to it once `now` is resolved.
                resolved = j <= passed_since
                while opening.open_t + dt < now || (resolved && opening.open_t + dt == now)
                    dt = _draw_beyond(pair_rng, kernel, m, dt)
                end
                candidate = opening.open_t + dt
                candidate <= opening.close_t || continue
                _propose!(
                    queue, proposals, head, best, represents, j, oi, candidate, true
                )
            end
        end
    end
    # Each member's earliest proposal, kept or new, takes its place in the heap.
    for j in hit
        _requeue!(queue, proposals, head, best, represents, j)
    end
    return unlinked
end

# Rebuild `proposals` and the heap from the proposals still linked to members
# that have not settled, dropping those a redraw unlinked and those to members
# already settled. Each member keeps its proposals in order, and only its
# earliest goes back in the heap: an entry for any other is skipped when popped,
# and a blocked contact requeues the next earliest. A settled member's chain is
# kept too when `reinfects`, since a proposal to them held back by
# `HostImmunity` is still due once they wane, not abandoned.
function _compact_proposals!(queue, proposals, head, best, represents, processed, reinfects = false)
    kept = empty(proposals)
    empty!(queue)
    for j in eachindex(head)
        q = head[j]
        head[j] = 0
        representative = 0
        last = 0
        while (reinfects || !processed[j]) && q != 0
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
            _enqueue!(queue, _Event(best[j], j, representative))
        end
    end
    copy!(proposals, kept)
    return nothing
end
