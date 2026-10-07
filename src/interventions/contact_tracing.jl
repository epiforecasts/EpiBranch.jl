# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. The default built-ins reproduce the original
# `ContactTracing(probability, isolation_to_trace_delay, quarantine_on_trace)`
# behaviour; user-defined subtypes slot in via a single method.

"""
    TraceEligibility

Rule for which cases have their contacts traced, and from when. Built in:
[`OnSymptomOnset`](@ref), [`OnLabConfirmation`](@ref), [`OnIsolation`](@ref),
[`SymptomaticParent`](@ref) (the default), [`PreviouslyTraced`](@ref),
[`TraceEveryone`](@ref) and [`TraceNobody`](@ref). Combine them with `&`
(and), `|` (or) and `!` (not), for example
`OnSymptomOnset() | OnLabConfirmation()` to trace suspected or confirmed
cases.

To write a new rule, define a subtype and a method of
[`is_eligible`](@ref EpiBranch.is_eligible).
"""
abstract type TraceEligibility end

"""
    is_eligible(eligibility, infector, contact, state) -> Bool

Whether `contact` of the case `infector` is traced under this rule (`true` by
default).
"""
is_eligible(::TraceEligibility, infector, contact, state) = true

# ── Built-in eligibility policies ─────────────────────────────────
#
# Each built-in is an *atomic* predicate — it tests one thing about the
# infector. Compose them with the boolean operators `&`, `|`, `!` (see
# below): e.g. `OnSymptomOnset() & !OnIsolation()` for "symptomatic but
# not yet isolated". Keeping predicates atomic is what makes that
# composition read correctly.

"""Trace contacts of symptomatic cases (clinical suspicion), starting from
symptom onset, so tracing can begin before lab confirmation or without it.
See [`trigger_time`](@ref EpiBranch.trigger_time)."""
struct OnSymptomOnset <: TraceEligibility end
is_eligible(::OnSymptomOnset, infector, contact, state) = _develops_symptoms(infector)

"""Trace contacts of cases that tested positive (lab confirmation), starting
from the case's isolation."""
struct OnLabConfirmation <: TraceEligibility end
function is_eligible(::OnLabConfirmation, infector, contact, state)
    return get(infector.state, :test_positive, false)
end

"""Trace contacts of cases that have been isolated, starting from isolation."""
struct OnIsolation <: TraceEligibility end
is_eligible(::OnIsolation, infector, contact, state) = is_isolated(infector)

"""Trace the contacts of every case, whatever its status, starting from the
case's isolation."""
struct TraceEveryone <: TraceEligibility end
is_eligible(::TraceEveryone, infector, contact, state) = true

"""Trace no contacts."""
struct TraceNobody <: TraceEligibility end
is_eligible(::TraceNobody, infector, contact, state) = false

"""Trace contacts only of cases that were themselves traced as someone
else's contact. Negate it to stop a traced contact who later becomes a case
from being interviewed again: `SymptomaticParent() & !PreviouslyTraced()`.
By default such a case is interviewed like any other, on every transmission
model."""
struct PreviouslyTraced <: TraceEligibility end
is_eligible(::PreviouslyTraced, infector, contact, state) = is_traced(infector)

"""Trace contacts of cases that are symptomatic and isolated, starting from
isolation. This is the default `eligibility` of [`ContactTracing`](@ref), and
the same as `OnSymptomOnset() & OnIsolation()`."""
struct SymptomaticParent <: TraceEligibility end
function is_eligible(::SymptomaticParent, infector, contact, state)
    return _develops_symptoms(infector) && is_isolated(infector)
end

# `AlwaysEligible` and `NoTracing` were the previous names for tracing
# everyone / no-one; their semantics are identical to the new policies,
# so they are aliases.
"""Another name for [`TraceEveryone`](@ref): every contact is traced."""
const AlwaysEligible = TraceEveryone

"""Another name for [`TraceNobody`](@ref): no contact is traced."""
const NoTracing = TraceNobody

# ── Composition ────────────────────────────────────────────────────
#
# Policies form a boolean algebra over the infector predicate. The
# wrapper types below are normally built through the operators `&`, `|`,
# `!` rather than by name, so user code reads as ordinary boolean logic:
#
#     OnSymptomOnset() | OnLabConfirmation()    # suspected or confirmed
#     OnSymptomOnset() & !OnIsolation()         # symptomatic, not isolated

"""Trace if any of the listed conditions holds. Usually written `a | b`."""
struct AnyOf{T <: Tuple} <: TraceEligibility
    conditions::T
    AnyOf(conditions...) = new{typeof(conditions)}(conditions)
end

function is_eligible(e::AnyOf, infector, contact, state)
    for condition in e.conditions
        is_eligible(condition, infector, contact, state) && return true
    end
    return false
end

"""Trace only if all of the listed conditions hold. Usually written `a & b`."""
struct AllOf{T <: Tuple} <: TraceEligibility
    conditions::T
    AllOf(conditions...) = new{typeof(conditions)}(conditions)
end

function is_eligible(e::AllOf, infector, contact, state)
    for condition in e.conditions
        is_eligible(condition, infector, contact, state) || return false
    end
    return true
end

"""Trace only if none of the listed conditions holds. `!a` is the
shorthand for a single condition."""
struct NoneOf{T <: Tuple} <: TraceEligibility
    conditions::T
    NoneOf(conditions...) = new{typeof(conditions)}(conditions)
end

function is_eligible(e::NoneOf, infector, contact, state)
    for condition in e.conditions
        is_eligible(condition, infector, contact, state) && return false
    end
    return true
end

# Boolean operators on policies. These only *construct* the wrappers
# above; evaluation happens later in `is_eligible`.
Base.:|(a::TraceEligibility, b::TraceEligibility) = AnyOf(a, b)
Base.:&(a::TraceEligibility, b::TraceEligibility) = AllOf(a, b)
Base.:!(a::TraceEligibility) = NoneOf(a)

# ── Trigger time ────────────────────────────────────────────────────
#
# The same eligibility policy that decides *whether* to trace also sets
# *when*: the trace is timed from when the infector first meets the
# condition. `OnSymptomOnset` times from symptom onset, so tracing can
# start before lab confirmation (or without it), while the historical
# default and the isolation/confirmation policies time from isolation.
# `ContactTracing` adds its delay on top. Combinators reduce over the
# conditions that are met for the infector and contact: an `AnyOf`
# triggers when its first met condition is met, an `AllOf` when its last
# one is. A condition that is not met has no trigger time of its own (an
# asymptomatic case has no onset). Including it in the reduction could
# make the time earlier than any real event, or `NaN`. A negation that
# holds has no event to time it either, so it sets no time inside an
# `AllOf`.

# How combined rules are timed, in full. Each wrapped condition is either
# not met, met at a time, or met with no time of its own (a negation that
# holds). A negation that does not hold is not met. A negated combinator is
# timed as its De Morgan form, so `!(a & b)` is timed as `!a | !b`, and `!!a`
# as `a`. `AllOf` is not met if any condition is not met; otherwise it is met
# at the latest time among its timed conditions, or with no time if none is
# timed. `AnyOf` is not met if no condition is met; if one of its met
# conditions has no time, so does the `AnyOf` (and inside an `AllOf` it sets
# no time); otherwise it is met at the earliest time among its met
# conditions. A policy met with no time starts at the default trigger time
# (isolation, as for `TraceEveryone`) or earlier if a timed branch is met
# earlier, so `OnSymptomOnset() | !OnIsolation()` traces a never-isolated case
# from onset. A `NaN` time from a wrapped condition counts as never met.
#
# Two met policies get the same time if one is rewritten into the other by De
# Morgan's laws, double negation, commutativity, associativity or
# distributing `&` over `|` outside a negation, and `TraceNobody() | p` is
# timed as `p`; the exception is a policy whose own time is `NaN`, which
# gives `Inf` once wrapped. Other logically equal policies can differ, since
# a rewrite that adds or removes an untimed negation, or `TraceEveryone()`
# (timed at isolation), changes which times count. With `S = OnSymptomOnset()`,
# `L = OnLabConfirmation()`, `I = OnIsolation()` and a case with onset at 4,
# isolated at 9 and never lab-confirmed:
#
# - distributing inside a negation: `!(L & (!I | !S))` triggers at 9 and
#   `!((L & !I) | (L & !S))` at 4;
# - absorption by an untimed branch: `!L | (!L & S)` triggers at 4, `!L` at 9;
# - a condition joined with its negation: `S & (I | !I)` triggers at 9, `S`
#   at 4;
# - joining `TraceEveryone()`: `S & TraceEveryone()` at 9 and `S` at 4;
#   `S | TraceEveryone()` at 4 and `TraceEveryone()` at 9;
# - `!TraceNobody()` behaves as `TraceEveryone()` only at the top level:
#   `S & !TraceNobody()` at 4 and `S & TraceEveryone()` at 9.
#
# The four-argument form checks each wrapped condition against the contact
# and times it with its four-argument method. The three-argument form times
# each with its three-argument method and checks it with `nothing` for the
# contact, so `trigger_time(p, ...)`, `trigger_time(!!p, ...)` and
# `trigger_time(AllOf(p), ...)` agree for any met `p` whose own time is not
# `NaN`.
"""
    trigger_time(eligibility, infector, contact, state) -> Float64
    trigger_time(eligibility, infector, state) -> Float64

The time (in days) from which contacts of the case `infector` are traced
under this eligibility rule; [`ContactTracing`](@ref) adds its
`isolation_to_trace_delay` on top. By default this is the case's isolation
time, or `Inf` (never) if the isolation does not count as a detection (see
[`is_isolated`](@ref)). [`OnSymptomOnset`](@ref) uses the onset time instead,
so suspicion-based tracing starts at symptom onset rather than waiting for
isolation or confirmation.

Combined rules take their time from the conditions they combine. `a & b`
starts at the latest time among its conditions, so
`OnSymptomOnset() & !OnIsolation()` traces from onset; `a | b` starts at the
earliest time among the conditions that hold. A negation that holds, such as
`!OnIsolation()` for a case not yet isolated, has no time of its own: it does
not move the time of an `&`, and a rule with no timed condition starts at the
case's isolation time (or earlier if a timed branch of an `|` holds). A
combined rule that does not hold gives `Inf`.

Rules that are logically equal usually give the same time, but not always,
because untimed negations and `TraceEveryone()` (timed at isolation) can add
or remove times. For example, `S & TraceEveryone()` with `S = OnSymptomOnset()`
starts at isolation while `S` alone starts at onset.

For extension authors: a rule timed from another event of the case defines a
three-argument method; one timed from the contact defines a four-argument
method (the four-argument form falls back to the three-argument one).
`ContactTracing` calls the four-argument form, and only once
[`is_eligible`](@ref EpiBranch.is_eligible) holds. A single rule's
`trigger_time` does not itself check eligibility. A combined rule containing a
condition that reads the contact must be timed with the four-argument form.
"""
trigger_time(::TraceEligibility, infector, state) = _recorded_isolation_time(infector)
trigger_time(::OnSymptomOnset, infector, state) = onset_time(infector)
function trigger_time(e::Union{AnyOf, AllOf, NoneOf}, infector, state)
    return _combined_time(_WithoutContact(), e, infector, nothing, state)
end

function trigger_time(e::TraceEligibility, infector, contact, state)
    return trigger_time(e, infector, state)
end

function trigger_time(e::Union{AnyOf, AllOf, NoneOf}, infector, contact, state)
    return _combined_time(_WithContact(), e, infector, contact, state)
end

# Which `trigger_time` method times the policies inside a combinator: the
# form the combinator was called with, so that wrapping a policy does not
# change which of its methods is used.
struct _WithContact end
struct _WithoutContact end
function _single_time(::_WithContact, e, infector, contact, state)
    return trigger_time(e, infector, contact, state)
end
function _single_time(::_WithoutContact, e, infector, contact, state)
    return trigger_time(e, infector, state)
end

# What every node of a combinator is evaluated against. `never` is `Inf` in
# the infector's time type, so AD dual numbers pass through, and is computed
# once per evaluation.
struct _TimingContext{F, T, I, C, S}
    form::F
    never::T
    infector::I
    contact::C
    state::S
end

_never(::Individual{T}) where {T} = T(Inf)
_never(infector) = oftype(isolation_time(infector), Inf)

function _combined_time(form, e, infector, contact, state)
    cx = _TimingContext(form, _never(infector), infector, contact, state)
    t, untimed = _met_time(cx, e, false)
    untimed || return t
    default = _single_time(form, TraceEveryone(), infector, contact, state)
    return min(t, isnan(default) ? cx.never : default)
end

# How `condition`, or its negation if `negated`, is met, as
# `(time, untimed)`. `time` is the earliest time at which it is met through
# a timed branch, `Inf` if there is none. `untimed` is whether it also
# holds with no time of its own, as a negation that holds does. A
# condition that is not met gives `(Inf, false)`. Negations are pushed
# inwards by De Morgan's laws, so equal policies get equal times.
function _met_time(cx::_TimingContext, condition::TraceEligibility, negated::Bool)
    met = is_eligible(condition, cx.infector, cx.contact, cx.state)::Bool
    negated && return (cx.never, !met)
    met || return (cx.never, false)
    t = _single_time(cx.form, condition, cx.infector, cx.contact, cx.state)
    return (isnan(t) ? cx.never : t, false)
end

function _met_time(cx::_TimingContext, e::AnyOf, negated::Bool)
    return negated ? _all_met(cx, e.conditions, true) : _any_met(cx, e.conditions, false)
end

function _met_time(cx::_TimingContext, e::AllOf, negated::Bool)
    return negated ? _any_met(cx, e.conditions, true) : _all_met(cx, e.conditions, false)
end

# `NoneOf(a, b)` is `!a & !b`, and its negation is `a | b`.
function _met_time(cx::_TimingContext, e::NoneOf, negated::Bool)
    return negated ? _any_met(cx, e.conditions, false) : _all_met(cx, e.conditions, true)
end

# The reductions below walk the conditions tuple one element at a time, so
# each step is compiled for that condition's type and the times stay
# unboxed.

# Met through any condition: the earliest of their times, and untimed if
# any condition is.
_any_met(cx, conditions, negated) = _any_met(cx, conditions, negated, cx.never, false)
_any_met(cx, ::Tuple{}, negated, t, untimed) = (t, untimed)
function _any_met(cx, conditions::Tuple, negated, t, untimed)
    tc, uc = _met_time(cx, first(conditions), negated)
    return _any_met(cx, Base.tail(conditions), negated, min(t, tc), untimed | uc)
end

# Met when every condition is. A condition that holds with no time of its
# own sets no time, so the latest time among the others decides. If no
# condition is timed, the whole holds with no time of its own and keeps
# the earliest of their timed branches: `(a | !b) & !c` is met at `a`'s
# time through `a & !c`.
function _all_met(cx, conditions, negated)
    return _all_met(cx, conditions, negated, oftype(cx.never, -Inf), cx.never, false)
end
function _all_met(cx, ::Tuple{}, negated, latest, earliest, timed)
    return timed ? (latest, false) : (earliest, true)
end
function _all_met(cx, conditions::Tuple, negated, latest, earliest, timed)
    tc, uc = _met_time(cx, first(conditions), negated)
    rest = Base.tail(conditions)
    uc && return _all_met(cx, rest, negated, latest, min(earliest, tc), timed)
    return _all_met(cx, rest, negated, max(latest, tc), earliest, true)
end

"""
    TraceRate

Rule for whether an eligible contact is actually found by tracing. Built in:
[`ConstantRate`](@ref), a fixed probability per contact. To write a new rule,
define a subtype and a method of [`traces`](@ref EpiBranch.traces).
"""
abstract type TraceRate end

"""
    traces(rate, infector, contact, state, rng) -> Bool

Whether this eligible contact is successfully traced (randomly, using `rng`).
"""
traces(::TraceRate, infector, contact, state, rng) = false

"""Each eligible contact is traced with the same probability `p`."""
struct ConstantRate <: TraceRate
    p::Float64
end
traces(r::ConstantRate, infector, contact, state, rng) = rand(rng) < r.p

"""
    TraceDelay

Rule for the delay, in days, from the case's tracing start (its isolation by
default; see [`trigger_time`](@ref EpiBranch.trigger_time)) to its contact
being reached. Built in: [`ConstantDelay`](@ref), one distribution for every
contact. To write a new rule, define a subtype and a method of
[`draw_trace_delay`](@ref EpiBranch.draw_trace_delay).
"""
abstract type TraceDelay end

"""
    draw_trace_delay(delay, infector, contact, state, rng) -> Float64

Draw the delay in days before this contact is reached.
"""
draw_trace_delay(::TraceDelay, infector, contact, state, rng) = 0.0

"""Every contact's tracing delay (days) is drawn from the same distribution
`dist`."""
struct ConstantDelay{D <: Distribution} <: TraceDelay
    dist::D
end
draw_trace_delay(d::ConstantDelay, infector, contact, state, rng) = float(rand(rng, d.dist))

"""
    TraceAction

What happens to a contact once traced: [`Quarantine`](@ref) (the default) or
[`FlagOnly`](@ref). To write a new action, define a subtype and a method of
[`apply_trace!`](@ref EpiBranch.apply_trace!).
"""
abstract type TraceAction end

"""
    apply_trace!(action, contact, state, trace_time, rng)

Act on `contact`, traced at `trace_time` (days); changes `contact` in place.
"""
apply_trace!(::TraceAction, contact, state, trace_time, rng) = nothing

# The key a quarantine records its own removals under, apart from the shared
# history `set_isolated!` keeps, so that its block covers its own days only.
const QUARANTINE_STRETCHES_KEY = :_quarantine_stretches

"""
    Quarantine(; duration = Inf)

Quarantine a traced contact from the time they are reached (or from their own
isolation, if that is earlier), so they cannot infect others while
quarantined. A trace that never arrives (trace time `Inf`) marks the contact
as traced and quarantined without changing any isolation already in place.

`duration` is how long the quarantine lasts, in days: a number, a distribution
or a function of the random number generator and the individual,
`(rng, ind) -> ...`, drawn once per trace. The default `Inf` never releases
the contact. A finite duration matters for a contact who escapes the traced
exposure and is infected later through another route: otherwise the
quarantine keeps blocking that person's own onward transmission long after
its reason has passed.
"""
struct Quarantine{D} <: TraceAction
    duration::D
end
Quarantine(; duration = Inf) = Quarantine(duration)

function apply_trace!(q::Quarantine, contact, state, trace_time, rng)
    contact.state[:traced] = true
    contact.state[:quarantined] = true
    # A trace with no arrival time quarantines nobody: an isolation at `Inf`
    # removes the contact from nothing while reporting it as isolated and
    # detected, and `min` would carry a `NaN` into a standing isolation and
    # from there into the case's infectious window. `!OnIsolation()` reaches a
    # contact this way, no earlier than an isolation its infector has not had.
    # Returning before the draw also leaves the stream where it was for a
    # trace that changes nothing.
    isfinite(trace_time) || return nothing
    release_time = trace_time + _removal_duration(
        q.duration, rng, contact, "`Quarantine`'s `duration`"
    )
    record_removal!(contact, trace_time, release_time; key = QUARANTINE_STRETCHES_KEY)
    if _isolation_in_force(contact)
        standing = isolation_time(contact)
        was_unrecorded = _isolation_unrecorded(contact)
        final_time, final_release = _combine_removal(
            standing, isolation_release_time(contact), trace_time, release_time
        )
        # An isolation that was not recorded keeps that status only while its own
        # start is the one in force. Where the trace's start wins, or replaces a
        # spent removal, the removal is the trace's and is recorded as such.
        unrecorded = was_unrecorded && final_time == standing
        set_isolated!(contact, final_time; release_time = final_release)
        unrecorded && (contact.state[:_isolation_unrecorded] = true)
    else
        set_isolated!(contact, trace_time; release_time = release_time)
    end
    return nothing
end

"""Record the contact as traced without quarantining them: they keep
transmitting until they would be isolated as a case. If they develop
symptoms, [`Isolation`](@ref) then isolates them at whichever comes first,
their own self-reported isolation or the later of the trace time and their
symptom onset. An asymptomatic contact is never isolated this way."""
struct FlagOnly <: TraceAction end
function apply_trace!(::FlagOnly, contact, state, trace_time, rng)
    contact.state[:traced] = true
    contact.state[:quarantined] = false
    is_asymptomatic(contact) && return nothing
    # A continuous-time model traces a contact before the race has settled its
    # infection, so its onset is still unknown. Isolation holds the recorded
    # time back to the onset once it is known, and never isolates a contact
    # that has none, which makes the trace time alone safe to record here.
    # A contact reached by several infectors keeps its earliest trace.
    ind_onset = onset_time(contact)
    traced_iso = isnan(ind_onset) ? trace_time : max(ind_onset, trace_time)
    contact.state[:_traced_isolation_time] = min(
        get(contact.state, :_traced_isolation_time, Inf), traced_iso
    )
    return nothing
end

# ── ContactTracing intervention ──────────────────────────────────────

"""
    ContactTracing(eligibility, probability, isolation_to_trace_delay,
                   action = Quarantine(); depth = 1)
    ContactTracing(; probability, isolation_to_trace_delay,
                   quarantine_on_trace = true,
                   eligibility = SymptomaticParent(), depth = 1)

Trace the contacts of cases, and quarantine (or just record) the contacts
found. Isolation applies to cases ([`Isolation`](@ref)); quarantine applies to
traced contacts who are not yet known cases.

# Arguments
- `eligibility`: which cases have their contacts traced and from when, for
  example [`OnSymptomOnset`](@ref) (clinical suspicion, from onset),
  [`OnLabConfirmation`](@ref), [`OnIsolation`](@ref), [`TraceEveryone`](@ref)
  or [`TraceNobody`](@ref). Combine with `&`, `|` and `!`. The keyword form
  defaults to [`SymptomaticParent`](@ref) (symptomatic and isolated).
- `probability`: share of an eligible case's contacts that tracing finds.
- `isolation_to_trace_delay`: distribution of days from when tracing of the
  case starts (its isolation by default, onset for `OnSymptomOnset`) to each
  contact being reached.
- `action`: [`Quarantine`](@ref) (default) or [`FlagOnly`](@ref). In the
  keyword form, `quarantine_on_trace = false` chooses `FlagOnly()`.
- `depth`: how far tracing reaches: 1 (default) traces contacts, 2 also
  contacts of contacts, and so on. Must be at least 1.

# Examples
```julia
# Trace on symptoms, without waiting for confirmation
ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0))

# Wait for lab confirmation
ContactTracing(OnLabConfirmation(), 0.6, Exponential(2.0))

# Trace suspected or confirmed cases
ContactTracing(OnSymptomOnset() | OnLabConfirmation(), 0.7, Exponential(1.5))

# Keyword form with the default eligibility (symptomatic and isolated)
ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(1.0))
```

# Ring depth

With `depth = 2` the traced contacts of a case have their own contacts traced
in turn: the contacts of contacts that level-2 ring vaccination targets. Each
infected, eligible case starts a fresh ring of radius `depth`. Uninfected ring
members are followed for one more generation so the ring can extend past them,
but they do not infect anyone. A traced contact who later becomes an eligible
case starts its own ring like any other case; add `!PreviouslyTraced()` to the
eligibility to interview each case only once.

Pair with [`RingVaccination`](@ref) to vaccinate the whole ring:

```julia
[ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0); depth = 2),
 RingVaccination(efficacy = 0.9)]
```

# Requirements and output

Needs symptom onset and the asymptomatic flag from
[`clinical_presentation`](@ref), and, depending on the eligibility, isolation
or test results from [`Isolation`](@ref). Each contact records whether it was
traced and quarantined, and `trace_time`, the day it was reached, from which
interventions acting on traced contacts are timed.

# Custom eligibility

A new eligibility rule is a subtype of [`TraceEligibility`](@ref) with a
method of [`is_eligible`](@ref EpiBranch.is_eligible). Case characteristics
set by [`clinical_presentation`](@ref) or [`demographics`](@ref) are in
`infector.state`:

```julia
struct SymptomaticOver65 <: TraceEligibility end

function EpiBranch.is_eligible(::SymptomaticOver65, infector, contact, state)
    return !is_asymptomatic(infector) && get(infector.state, :age, 0) >= 65
end
```
"""
struct ContactTracing{
        E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction,
    } <:
    AbstractIntervention
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A
    depth::Int

    # Validate the ring radius once, here, so every construction path
    # (all the convenience constructors below funnel through this) rejects
    # depth < 1 rather than silently behaving as depth 1.
    function ContactTracing(
            eligibility::E, trace_rate::F,
            isolation_to_trace_delay::D, action::A,
            depth::Integer
        ) where {
            E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction,
        }
        depth >= 1 ||
            throw(ArgumentError("ContactTracing depth must be at least 1, got $depth"))
        return new{E, F, D, A}(
            eligibility, trace_rate, isolation_to_trace_delay, action, Int(depth)
        )
    end
end

# Fully-typed form (eligibility + trait objects) with a default depth, so
# callers that build the traits directly need not pass `depth`.
function ContactTracing(
        eligibility::TraceEligibility, trace_rate::TraceRate,
        isolation_to_trace_delay::TraceDelay, action::TraceAction;
        depth::Integer = 1
    )
    return ContactTracing(
        eligibility, trace_rate, isolation_to_trace_delay, action, Int(depth)
    )
end

function ContactTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        quarantine_on_trace::Bool = true,
        eligibility::TraceEligibility = SymptomaticParent(),
        depth::Integer = 1
    )
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(isolation_to_trace_delay),
        quarantine_on_trace ? Quarantine() : FlagOnly(),
        Int(depth)
    )
end

# Terse positional form: an eligibility policy with a constant trace
# probability and a constant delay distribution. The fully-typed inner
# constructor (taking `TraceRate`/`TraceDelay` objects) is unaffected —
# `probability::Real` and `delay::Distribution` do not match those.
function ContactTracing(
        eligibility::TraceEligibility, probability::Real,
        isolation_to_trace_delay::Distribution, action::TraceAction = Quarantine();
        depth::Integer = 1
    )
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(isolation_to_trace_delay),
        action,
        Int(depth)
    )
end

required_fields(ct::ContactTracing) = required_fields(ct.eligibility)
intervention_time(::ContactTracing, ind::Individual) = isolation_time(ind)

# Required-field validation dispatches on the eligibility trait. Each
# atomic predicate declares only the state key it reads.
required_fields(::OnSymptomOnset) = [:asymptomatic, :onset_time]
required_fields(::OnLabConfirmation) = [:test_positive]
required_fields(::OnIsolation) = [:isolated]
required_fields(::TraceEveryone) = Symbol[]
required_fields(::TraceNobody) = Symbol[]
required_fields(::SymptomaticParent) = [:asymptomatic, :isolated]
required_fields(::TraceEligibility) = Symbol[]  # Default for custom types

# Combinators inherit requirements from the policies they wrap.
function required_fields(e::AnyOf)
    return reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end
function required_fields(e::AllOf)
    return reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end
function required_fields(e::NoneOf)
    return reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end

function reset!(::ContactTracing, ind::Individual)
    ind.state[:traced] = false
    ind.state[:quarantined] = false
    haskey(ind.state, :traced_by) && delete!(ind.state, :traced_by)
    haskey(ind.state, :trace_level) && delete!(ind.state, :trace_level)
    haskey(ind.state, :trace_time) && delete!(ind.state, :trace_time)
    haskey(ind.state, :_ring_remaining) && delete!(ind.state, :_ring_remaining)
    haskey(ind.state, :_ring_propagated) && delete!(ind.state, :_ring_propagated)
    delete!(ind.state, QUARANTINE_STRETCHES_KEY)
    _isolation_in_force(ind) && clear_isolated!(ind)
    return nothing
end

function initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

# One trace attempt for a single infector → contact pair. Both engines funnel
# through this, so the generation-based and continuous-time paths apply exactly
# the same eligibility, rate, delay and action policy.
function _trace_pair!(ct::ContactTracing, state, infector, ind, rng; not_before = -Inf)
    # A contact enters the ring two ways. As a *seed*, when its
    # infector is an infected case meeting the eligibility condition
    # (a symptomatic case, say): this starts a fresh ring of radius
    # `depth` around that case. By *propagation*, when its infector is
    # itself a traced ring node with budget left: this is how the ring
    # reaches contacts-of-contacts. The seed path is the original
    # behaviour; propagation only happens when `depth > 1`.
    #
    # Requiring the seed's infector to be infected is a no-op at
    # `depth == 1` (only infected cases ever generate contacts there),
    # but it stops an uninfected ring member, which carries its own
    # clinical state, from re-seeding a fresh full-radius ring and
    # letting the fringe grow without bound.
    seed = is_infected(infector) && is_eligible(ct.eligibility, infector, ind, state)
    # A ring grows past a member once. The first walk that reaches it spends
    # its budget on its contacts; a later one would trace the same pairs again,
    # which is a second attempt at the same relationship rather than a wider
    # ring. Seeding is unaffected: a member that becomes an eligible case in
    # its own right starts a fresh full-radius ring, through the branch above.
    propagate = ct.depth > 1 && !seed && is_traced(infector) &&
        !get(infector.state, :_ring_propagated, false)::Bool &&
        get(infector.state, :_ring_remaining, 0)::Int > 0
    (seed || propagate) || return nothing
    traces(ct.trace_rate, infector, ind, state, rng) || return nothing

    trace_delay = draw_trace_delay(
        ct.isolation_to_trace_delay, infector, ind, state, rng
    )
    base = seed ? trigger_time(ct.eligibility, infector, ind, state) :
        get(infector.state, :trace_time, _recorded_isolation_time(infector))
    # A contact cannot be sought before it exists, such as a funeral contact
    # before the funeral, so the delay runs from whichever comes later.
    trace_time = max(base, not_before) + trace_delay
    apply_trace!(ct.action, ind, state, trace_time, rng)

    # Record the source this contact was traced from. The engine makes
    # one trace attempt per node, from the earliest-exposure infector, so
    # this is the *first* tracer: exact (the parent) on a tree,
    # first-reached on a cyclic network. `compute_trace_level!` walks it
    # back to the index case post-run; see issue #150.
    ind.state[:traced_by] = infector.id

    # When the contact was reached. Interventions that act on a traced
    # contact time themselves from this, which is the same whatever the
    # trace action and the contact's own clinical course.
    #
    # Keep the earliest across tracing systems, matching how `Quarantine`
    # keeps the earliest isolation time: with several `ContactTracing`
    # interventions in the stack, a contact is reached when the first of
    # them gets there.
    #
    # A `NaN` says nothing about when the contact was reached, and `min`
    # propagates it, so it would overwrite a good time another tracing system
    # had already written. A custom `trigger_time` can return one.
    if !isnan(trace_time)
        ind.state[:trace_time] = min(get(ind.state, :trace_time, Inf), trace_time)
    end

    # Record how far the ring can still grow from this contact, so a
    # contact-of-contact one hop further out can time its own trace from
    # here. Only `depth > 1` rings expand.
    if ct.depth > 1
        ind.state[:_ring_remaining] = seed ? ct.depth - 1 :
            get(infector.state, :_ring_remaining, 0)::Int - 1
    end
    return nothing
end

function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        ind.parent_id > length(state.individuals) && continue
        _trace_pair!(ct, state, state.individuals[ind.parent_id], ind, rng)
    end
    return nothing
end

traces_contacts(::ContactTracing) = true

"""Trace the contacts of `infector` in a continuous-time model, applying the
same eligibility, probability, delay and action as in a branching process.
Each contact is traced no earlier than its `not_before` time when one is
given."""
function trace_contacts!(
        ct::ContactTracing, state, infector, contacts, not_before = nothing
    )
    rng = state.rng
    # Whether this batch spends `infector`'s ring budget is decided per pair in
    # `_trace_pair!`, and the same answer holds for every pair in the batch, so
    # the budget is marked spent once the batch is done rather than part way
    # through it.
    propagating = ct.depth > 1 && !is_infected(infector) && is_traced(infector)
    for (i, ind) in enumerate(contacts)
        ind.id == infector.id && continue
        _trace_pair!(
            ct, state, infector, ind, rng;
            not_before = not_before === nothing ? -Inf : not_before[i]
        )
    end
    propagating && (infector.state[:_ring_propagated] = true)
    return nothing
end

"""A quarantined contact stops transmitting from its quarantine time. A
contact recorded with `FlagOnly` is removed only once [`Isolation`](@ref)
isolates it. A quarantine that is never released ends the person's
infectious period. With a finite `duration` (see [`Quarantine`](@ref)) each of
their contacts during quarantine is blocked and they transmit again once
released, as with [`Isolation`](@ref)."""
function infectious_removal_time(ct::ContactTracing, ind::Individual)
    get(ind.state, :quarantined, false) || return Inf
    t = Inf
    for key in _trace_removal_keys(ct.action)
        t = min(t, permanent_removal_time(ind, key))
    end
    return t
end

# As for `Isolation`: a quarantine's recorded stretches are append-only.
binding_release(::ContactTracing) = true

removal_gap_host_times(ct::ContactTracing) = removal_gap_host_times(ct.action)
removal_gap_host_times(::TraceAction) = ()
removal_gap_host_times(::Quarantine) = (QUARANTINE_STRETCHES_KEY,)

# Where the action recorded the removals this tracing should honour. An action
# written outside the package names its own key, as the built-in quarantine
# does, and both the window and the per-contact risk read it. One that names
# none and still removes the contact is read from the shared history
# `set_isolated!` keeps, which is where such an action will have recorded, so a
# quarantine it never releases goes on closing the window as it did before this
# seam existed. One it does release leaves the window open with nothing taken
# out of the exposure, which `infection_likelihood_compatible`'s `false`
# default keeps out of a likelihood.
# That fallback cannot tell the action's own stretches from another removal's,
# so an action composed with a leaky `Isolation` blocks the isolation's days
# fully as well; naming a key is what separates them.
function _trace_removal_keys(action::TraceAction)
    keys = removal_gap_host_times(action)
    return isempty(keys) ? (REMOVAL_STRETCHES_KEY,) : keys
end

# A quarantine's block is the quarantine's own, never the stretches some other
# removal put this host in: a leaky `Isolation` composed with tracing would
# otherwise become a perfect block over days the quarantine had nothing to do
# with. A quarantine that never releases is honoured by the infectious window
# instead, as a standing isolation is, which is what keeps it inside a
# fixed-size pool's single clock.
risk_depends_on_infector(ct::ContactTracing) = risk_depends_on_infector(ct.action)
risk_depends_on_infector(::TraceAction) = false
risk_depends_on_infector(q::Quarantine) = !(q.duration === Inf)

# A quarantine that lapses leaves the window open above, and the stretches it
# removed the case for are blocked per contact here instead. One with no
# release is blocked here too, from its own start and never released, which is
# what reduces onward transmission on the generation engine, where there is no
# window to close.
function competing_risk(ct::ContactTracing, parent, contact, state)
    get(parent.state, :quarantined, false) || return nothing
    risks = ()
    for key in _trace_removal_keys(ct.action)
        more = _removal_risks(parent, 1.0, key)
        more === nothing || (risks = (risks..., more...))
    end
    return isempty(risks) ? nothing : risks
end

# A quarantine is a removal, so it reaches only the routes a removal can cut.
function risk_applies(::ContactTracing, route)
    return route !== nothing && INTERVENTION_REMOVAL in route.until
end

# Tracing itself reads and writes only the infector's own contacts, but it
# hands each decision to a component that receives the whole state, and
# answers for those components too. The built-in ones read only the infector
# or the contact; one written outside the package declares its own read.
function reads_population_state(ct::ContactTracing)
    return reads_population_state(ct.eligibility) ||
        reads_population_state(ct.trace_rate) ||
        reads_population_state(ct.isolation_to_trace_delay) ||
        reads_population_state(ct.action)
end
reads_population_state(::TraceEligibility) = false
# A combinator is an eligibility too, and answers for what it wraps.
reads_population_state(e::AnyOf) = any(reads_population_state, e.conditions)
reads_population_state(e::AllOf) = any(reads_population_state, e.conditions)
reads_population_state(e::NoneOf) = any(reads_population_state, e.conditions)
reads_population_state(::TraceRate) = false
reads_population_state(::TraceDelay) = false
reads_population_state(::TraceAction) = false

"""With `depth > 1`, keep following uninfected traced contacts for one more
generation so tracing can reach their contacts. They do not infect those
contacts. Returns the ids of those uninfected ring members (none for
`depth == 1`)."""
function keep_active(ct::ContactTracing, state, targets, is_new)
    ct.depth > 1 || return ()
    ids = Int[]
    for t in targets
        is_infected(t) && continue
        is_traced(t) || continue
        get(t.state, :_ring_remaining, 0)::Int > 0 || continue
        push!(ids, t.id)
    end
    return ids
end
