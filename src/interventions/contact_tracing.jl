# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. A user-defined subtype slots in via a single method.

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
isolation. This is the default `eligibility` of [`ContactTracing`](@ref).
`OnSymptomOnset() & OnIsolation()` picks out the same cases but starts from
the later of onset and isolation, so for a case isolated before its own onset
the two trace from different times (see [`trigger_time`](@ref EpiBranch.trigger_time))."""
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
# Combined rules in full (see the `trigger_time` docstring). Each wrapped condition is either
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
function trigger_time(::TraceEligibility, infector, state)
    # A quarantine recorded while the person was still a susceptible contact
    # can precede their infection. The trace of their case starts no earlier
    # than that infection.
    t = _recorded_isolation_time(infector)
    return is_infected(infector) ? max(t, infector.infection_time) : t
end
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

What happens to a contact once traced: [`Quarantine`](@ref) or
[`FlagOnly`](@ref). To write a new action, define a subtype and a method of
[`apply_trace!`](@ref EpiBranch.apply_trace!). An action that needs to follow
a contact up later, such as a test some days after the trace, also defines
[`intervention_actions`](@ref EpiBranch.intervention_actions) and, to read
what the follow-up found once the contact's own infection is known,
[`on_infection_settled!`](@ref EpiBranch.on_infection_settled!);
[`ContactTracing`](@ref) forwards both to the action it holds.
"""
abstract type TraceAction end

"""
    apply_trace!(action, contact, state, trace_time, rng)

Act on `contact`, traced at `trace_time` (days); changes `contact` in place.
"""
apply_trace!(::TraceAction, contact, state, trace_time, rng) = nothing

"""
    intervention_actions(action::TraceAction, state, candidates)

Further actions a `TraceAction` proposes for its traced contacts, besides
what [`apply_trace!`](@ref EpiBranch.apply_trace!) already did immediately,
put through the same schedule and capacity machinery as any other
[`InterventionAction`](@ref EpiBranch.InterventionAction): a test some days
after the trace, say. Default: `nothing`, no further actions.
[`ContactTracing`](@ref) forwards to this once per case, exactly as
[`intervention_actions`](@ref EpiBranch.intervention_actions) does for a
plain [`AbstractIntervention`](@ref).
"""
intervention_actions(::TraceAction, state, candidates) = nothing

"""
    on_infection_settled!(action::TraceAction, individual, state, rng)

Continuous-time models only: called once a traced contact's own infection
time is fixed, before its onset or clinical transitions are worked out from
it, exactly as [`on_infection_settled!`](@ref EpiBranch.on_infection_settled!)
is for a plain [`AbstractIntervention`](@ref). By then everything a trace
recorded on the contact, such as a scheduled follow-up's day, already stands:
a contact traced from its own infector always settles after being traced (see
[`trace_contacts!`](@ref EpiBranch.trace_contacts!)). One reached only
through a wider ring (`ContactTracing`'s `depth > 1`) can already be settled,
in which case this has already run for it and does not run again. Use it for
an action whose outcome depends on the contact's own infection, such as a
test result that can shorten a quarantine already recorded with
[`revise_removal!`](@ref EpiBranch.revise_removal!). Default: does nothing.
[`ContactTracing`](@ref) forwards to this.
"""
on_infection_settled!(::TraceAction, individual, state, rng) = nothing

# Every tracer acts through this rather than `apply_trace!` directly: an
# action that draws on a capacity (one with a `capacity_key`) is filed in
# the running count as soon as it has set its flag, so a tracer that is not
# itself under `CapacityConstrained`, such as `SourceTracing`, still counts
# against the budget a wrapped one shares, whatever the action's type.
function _apply_trace!(action::TraceAction, contact, state, trace_time, rng, owner::Symbol)
    handles_before = get(contact.state, :_removal_handle_counter, 0)::Int
    apply_trace!(action, contact, state, trace_time, rng)
    _note_owned_handles!(contact, owner, handles_before)
    key = capacity_key(action)
    key === nothing && return nothing
    _capacity_used(get(contact.state, key, nothing)) && _note_capacity_use!(state, key, contact)
    return nothing
end

# The key a quarantine records its own removals under, apart from the shared
# history `set_isolated!` keeps, so that its block covers its own days only.
const QUARANTINE_STRETCHES_KEY = :_quarantine_stretches

"""
    Quarantine(; duration)

Quarantine a traced contact from the time they are reached (or from their own
isolation, if that is earlier), so they cannot infect others while
quarantined. A trace that never arrives (trace time `Inf`) marks the contact
as traced and quarantined without changing any isolation already in place.

`duration` (required) is how long the quarantine lasts, in days: a number, a
distribution or a function of the random number generator and the individual,
`(rng, ind) -> ...`, drawn once per trace. There is no default, because
policies differ on how long quarantine should last. `Inf` keeps the contact
quarantined until the end of their infectious period. A finite duration
matters for a contact who escapes the traced exposure and is infected later
through another route: that person transmits normally once the quarantine
ends.

`ind.state[:trace_time]` is already set by the time `duration` is called. A
rule phrased as an end day counted from exposure, such as "until 14 days after
exposure", can subtract the trace time from that day to get a duration. A rule
that ends quarantine at symptom onset can compare it against
[`onset_time`](@ref)`(ind)`. A contact traced more than once (ring propagation
keeping the widest budget any infector gave it, or more than one child
triggering `SourceTracing` on the same parent) keeps its *earliest* trace in
`:trace_time`. On a later trace, that stored time predates the one `duration`
is being called for, so a rule that subtracts it from a fixed end day will run
longer than intended.
"""
struct Quarantine{D} <: TraceAction
    duration::D
end
Quarantine(; duration) = Quarantine(duration)

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
    # This trace's own entry, read back merged with whatever isolation already
    # stands where the two overlap or touch, and apart from it otherwise.
    # Quarantine is never itself a detection; where it overlaps an unrecorded
    # isolation that started earlier, the merged view keeps that status (see
    # `_refresh_isolation_span!`), without this call having to ask.
    handle = _set_isolated!(contact, trace_time; release_time)
    # Every trace this contact has ever been quarantined by, so a full reset
    # (see `_reset_traced!`) withdraws each one: a contact traced
    # more than once (ring propagation, or more than one child triggering
    # `SourceTracing` on the same parent) extends its quarantine with a fresh
    # entry each time rather than revising the first.
    handles = get(contact.state, :_quarantine_removal_handles, nothing)
    contact.state[:_quarantine_removal_handles] =
        handles === nothing ? [handle] : push!(copy(handles), handle)
    # Filed here as well as by the built-in tracers (`_apply_trace!`), so a
    # tracer written outside the package, which calls this directly, still
    # counts the quarantine against the budget a wrapped tracer shares.
    _note_capacity_use!(state, capacity_key(q), contact)
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
    flag_for_isolation!(contact, traced_iso)
    return nothing
end

# ── ContactTracing intervention ──────────────────────────────────────

# What `ContactTracing` and `SourceTracing` share: both read an `eligibility`,
# a `trace_rate`, an `isolation_to_trace_delay` and an `action`, and write the
# same tracing state, so the methods that depend only on those are written once.
abstract type _Tracing <: AbstractIntervention end

_trace_rate(probability::Real) = ConstantRate(probability)
_trace_rate(rate::TraceRate) = rate
_trace_delay(delay::Distribution) = ConstantDelay(delay)
_trace_delay(delay::TraceDelay) = delay

"""
    ContactTracing(eligibility, probability, isolation_to_trace_delay, action;
                   depth = 1)
    ContactTracing(; probability, isolation_to_trace_delay, action,
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
- `action` (required): what happens to a contact who is found,
  [`Quarantine`](@ref) for a given `duration` or [`FlagOnly`](@ref) to record
  the trace without quarantining. There is no default, because policies differ
  in whether and for how long traced contacts are quarantined.
- `depth`: how far tracing reaches: 1 (default) traces contacts, 2 also
  contacts of contacts, and so on. Must be at least 1.

The keyword `quarantine_on_trace` is deprecated: `quarantine_on_trace = true`
means `action = Quarantine(duration = Inf)` and `false` means
`action = FlagOnly()`. It still works but gives a deprecation warning.

# Examples
```julia
# Trace on symptoms, without waiting for confirmation; quarantine for 7 days
ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 7.0))

# Wait for lab confirmation; quarantine for 14 days
ContactTracing(OnLabConfirmation(), 0.6, Exponential(2.0), Quarantine(duration = 14.0))

# Trace suspected or confirmed cases
ContactTracing(
    OnSymptomOnset() | OnLabConfirmation(), 0.7, Exponential(1.5), Quarantine(duration = 7.0)
)

# Keyword form with the default eligibility (symptomatic and isolated)
ContactTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = 7.0)
)
```

# Ring depth

With `depth = 2` the traced contacts of a case have their own contacts traced
in turn: the contacts of contacts that level-2 ring vaccination targets. Each
infected, eligible case starts a fresh ring of radius `depth`. The ring also
grows through members who are never infected, wherever the model can name a
member's contacts before it settles (see
[`EpiBranch.reads_settled_state`](@ref)); such a member infects nobody. On a
model whose contacts persist (a household or a network), tracing also reaches
a neighbour already infected earlier than the case doing the tracing, not only
the case's own, still-uninfected neighbours: quarantining that neighbour blocks
transmission it has not yet made, the way `SourceTracing` blocks an infector
found through its own case. A traced contact who later becomes an eligible
case starts its own ring like any other case; add `!PreviouslyTraced()` to the
eligibility to interview each case only once.

Pair with [`RingVaccination`](@ref) to vaccinate the whole ring:

```julia
[ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 7.0); depth = 2),
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
    } <: _Tracing
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A
    depth::Int

    # Every construction path ends here, so the ring radius is validated once
    # and `depth < 1` is rejected rather than silently behaving as depth 1. A
    # plain probability or delay distribution becomes a constant rate or delay.
    function ContactTracing(
            eligibility::TraceEligibility, probability::Union{Real, TraceRate},
            isolation_to_trace_delay::Union{Distribution, TraceDelay},
            action::TraceAction;
            depth::Integer = 1
        )
        depth >= 1 ||
            throw(ArgumentError("ContactTracing depth must be at least 1, got $depth"))
        rate = _trace_rate(probability)
        delay = _trace_delay(isolation_to_trace_delay)
        return new{typeof(eligibility), typeof(rate), typeof(delay), typeof(action)}(
            eligibility, rate, delay, action, Int(depth)
        )
    end
end

function ContactTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        action::Union{TraceAction, Nothing} = nothing,
        quarantine_on_trace::Union{Bool, Nothing} = nothing,
        eligibility::TraceEligibility = SymptomaticParent(),
        depth::Integer = 1
    )
    action = _resolve_action(action, quarantine_on_trace)
    return ContactTracing(
        eligibility, probability, isolation_to_trace_delay, action; depth
    )
end

# `quarantine_on_trace` hid the action behind a `Bool`; `action` replaces it,
# with no default, since a plausible policy might flag a contact without
# quarantining it, or quarantine it for any length of time. The deprecated
# keyword still resolves to an action for a release, mapping onto the same
# indefinite `Quarantine` it always built.
function _resolve_action(action, quarantine_on_trace)
    if quarantine_on_trace !== nothing
        action === nothing || throw(
            ArgumentError(
                "pass only one of `action` and the deprecated `quarantine_on_trace`"
            )
        )
        Base.depwarn(
            "`quarantine_on_trace` is deprecated, pass `action` instead " *
                "(`Quarantine(duration = ...)` or `FlagOnly()`)",
            :ContactTracing; force = true
        )
        return quarantine_on_trace ? Quarantine(duration = Inf) : FlagOnly()
    end
    action === nothing &&
        throw(ArgumentError("ContactTracing requires `action`"))
    return action
end

required_fields(t::_Tracing) = required_fields(t.eligibility)
# This is the time of the trace itself rather than the isolation time of the
# contact. A `FlagOnly` trace never writes an isolation, and a `Quarantine`
# trace can merge into an existing isolation that started at another time.
function intervention_time(t::_Tracing, ind::Individual)
    return _state_real(ind.state, _own_trace_time_key(_tracer_owner(t)), Inf)
end

# Required-field validation dispatches on the eligibility trait. Each
# atomic predicate declares only the state key it reads.
required_fields(::OnSymptomOnset) = [:asymptomatic, :onset_time]
required_fields(::OnLabConfirmation) = [:test_positive]
required_fields(::OnIsolation) = [:isolated]
required_fields(::SymptomaticParent) = [:asymptomatic, :isolated]
required_fields(::TraceEligibility) = Symbol[]  # Default for custom types

# Combinators inherit requirements from the policies they wrap.
function required_fields(e::Union{AnyOf, AllOf, NoneOf})
    return reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end

reset!(t::_Tracing, ind::Individual) = _reset_traced!(ind, _tracer_owner(t))

# `ContactTracing` (`:contact`) and `SourceTracing` (`:source`) write the same
# tracing state on whoever `apply_trace!` acts on. Each therefore keeps its own
# trace time and the removal handles of its own traces, as `Isolation` keeps
# `:_isolation_handle`. `reset!` undoes only the records of the calling tracer
# and leaves those of the other tracer, and any isolation recorded by another
# intervention, in place.
_own_trace_time_key(owner::Symbol) = owner === :contact ? :_contact_trace_time : :_source_trace_time
_own_handles_key(owner::Symbol) = owner === :contact ? :_contact_trace_handles : :_source_trace_handles
_other_tracer(owner::Symbol) = owner === :contact ? :source : :contact
_tracer_owner(::ContactTracing) = :contact

# Every removal entry `apply_trace!` just recorded on `contact`, whether a
# quarantine stretch or an isolation span, was numbered after `handles_before`.
# Those numbers belong to this tracer and it alone withdraws them.
function _note_owned_handles!(contact::Individual, owner::Symbol, handles_before::Int)
    handles_after = get(contact.state, :_removal_handle_counter, 0)::Int
    handles_after > handles_before || return nothing
    key = _own_handles_key(owner)
    new = collect((handles_before + 1):handles_after)
    owned = get(contact.state, key, nothing)
    contact.state[key] = owned === nothing ? new : vcat(owned, new)
    return nothing
end

function _reset_traced!(ind::Individual, owner::Symbol)
    other = _other_tracer(owner)
    other_traced = haskey(ind.state, _own_trace_time_key(other))
    owned = Int[]
    claimed = Int[]
    for tracer in (owner, other)
        handles = get(ind.state, _own_handles_key(tracer), nothing)
        handles === nothing && continue
        append!(claimed, handles)
        tracer === owner && append!(owned, handles)
    end
    # A quarantine applied outside the tracers, for example by a custom tracer
    # through `apply_trace!`, is claimed by neither. Either tracer's reset
    # still undoes it, as a full reset always has.
    standing = get(ind.state, :_quarantine_removal_handles, nothing)
    standing === nothing || append!(owned, filter(h -> !(h in claimed), standing))
    entries = get(ind.state, :_removal_entries, nothing)
    if entries !== nothing
        for e in entries
            e[2] === QUARANTINE_STRETCHES_KEY && !(e[1] in claimed) && push!(owned, e[1])
        end
    end
    for handle in owned
        withdraw_removal!(ind, handle)
    end
    delete!(ind.state, _own_handles_key(owner))
    # `CapacityConstrained` counts the quarantine handles still standing, so
    # the handles withdrawn here must leave that count.
    if standing !== nothing
        kept = filter(h -> !(h in owned), standing)
        if isempty(kept)
            delete!(ind.state, :_quarantine_removal_handles)
        else
            ind.state[:_quarantine_removal_handles] = kept
        end
    end
    delete!(ind.state, _own_trace_time_key(owner))
    # `:trace_time` is the earliest of the two tracers' own times.
    if other_traced
        ind.state[:trace_time] = _state_real(ind.state, _own_trace_time_key(other), Inf)
    else
        delete!(ind.state, :trace_time)
        # A `FlagOnly` trace writes this for both tracers alike. It can be
        # told apart only when the other tracer has not traced the individual.
        _clear_isolation_flag!(ind)
    end
    ind.state[:traced] = other_traced
    ind.state[:quarantined] =
        other_traced && !isempty(get(ind.state, _own_handles_key(other), ()))
    if owner === :contact
        for k in (:traced_by, :tracers, :trace_level, :_ring_remaining, :_ring_propagated)
            delete!(ind.state, k)
        end
    else
        delete!(ind.state, :source_traced_by)
    end
    return nothing
end

function initialise_individual!(::_Tracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

# Keep the earliest trace time across tracing systems, matching how
# `Quarantine` keeps the earliest isolation time: shared by `ContactTracing`
# and `SourceTracing`, whichever tracing system reaches a person first. A
# `NaN` says nothing about when they were reached, and `min` propagates it, so
# it would overwrite a good time another tracing system had already written; a
# custom `trigger_time` can return one.
function _record_trace_time!(ind::Individual, trace_time, owner::Union{Symbol, Nothing} = nothing)
    isnan(trace_time) && return nothing
    if owner !== nothing
        key = _own_trace_time_key(owner)
        ind.state[key] = min(_state_real(ind.state, key, Inf), trace_time)
    end
    ind.state[:trace_time] = min(_state_real(ind.state, :trace_time, Inf), trace_time)
    return nothing
end

# Record `tracer` among everyone whose trace reached `ind`, in `:tracers`,
# which `compute_trace_level!` searches for the nearest index case. `:traced_by`
# is the one of them that reached `ind` first, by the earliest `trace_time`
# and the first recorded on a tie, which makes reading a single tracer
# deterministic.
# Called before `_record_trace_time!`, which overwrites the time it compares to.
function _record_tracer!(ind::Individual, tracer::Int, trace_time)
    tracers = get!(() -> Int[], ind.state, :tracers)::Vector{Int}
    tracer in tracers || push!(tracers, tracer)
    if !haskey(ind.state, :traced_by) ||
            trace_time < _state_real(ind.state, :trace_time, Inf)
        ind.state[:traced_by] = tracer
    end
    return nothing
end

# The trace attempts for a single infector → contact pair, pushing an
# `InterventionAction` for each one that succeeds onto `actions` rather than
# acting immediately, so the batch collected over a whole `trace_contacts!`
# call can be admitted (and, under `CapacityConstrained`, rationed) together.
function _trace_pair!(
        ct::ContactTracing, state, infector, ind, rng, actions;
        ring::Bool = false, not_before = -Inf
    )
    # A contact enters the ring two ways. As a *seed*, when its infector is an
    # infected case meeting the eligibility condition (a symptomatic case,
    # say): this starts a fresh ring of radius `depth` around that case, timed
    # from the case's own trigger. By *propagation*, when its infector is
    # itself a traced ring member with budget left (`ring`): this is how the
    # ring reaches contacts-of-contacts, timed from when the ring reached that
    # member. A member that is both makes both attempts, as it does when its
    # contacts are known as soon as the ring reaches it, before it is
    # infected, and again once it is: whether a model can name a member's
    # contacts that early does not change how they are traced.
    #
    # Requiring the seed's infector to be infected stops an uninfected ring
    # member, which has its own clinical state, from re-seeding a fresh
    # full-radius ring and letting the fringe grow without bound. The ring
    # walk can also pass through an already-infected neighbour (see
    # `_walk_ring!` in sellke.jl); that walk marks the infector with
    # `_ring_hop`, so it is traced from as a ring member only, rather than
    # reseeding a second time.
    if ring
        base = _state_real(infector.state, :trace_time, _recorded_isolation_time(infector))
        remaining = get(infector.state, :_ring_remaining, 0)::Int - 1
        _trace_attempt!(ct, state, infector, ind, rng, base, remaining, not_before, actions)
    end
    if is_infected(infector) && is_eligible(ct.eligibility, infector, ind, state) &&
            !is_ring_hop(infector)
        base = trigger_time(ct.eligibility, infector, ind, state)
        _trace_attempt!(ct, state, infector, ind, rng, base, ct.depth - 1, not_before, actions)
    end
    return nothing
end

# One attempt to trace `ind` from `infector`, the trace delay running from
# `base`, leaving the ring `remaining` hops to grow from `ind`. Who was
# reached, and from when, is recorded immediately — ring growth and later
# eligibility checks read it before any action is admitted — and only the
# action that follows up the contact (`apply_trace!`, a quarantine's removal
# or a flag) is queued, so `CapacityConstrained` can still turn it down.
function _trace_attempt!(
        ct::ContactTracing, state, infector, ind, rng, base, remaining, not_before, actions
    )
    traces(ct.trace_rate, infector, ind, state, rng) || return nothing
    trace_delay = draw_trace_delay(
        ct.isolation_to_trace_delay, infector, ind, state, rng
    )
    # A contact cannot be sought before it exists, such as a funeral contact
    # before the funeral, so the delay runs from whichever comes later.
    trace_time = max(base, not_before) + trace_delay

    # When the contact was reached. Recorded before the action runs, so an
    # action reading the contact's own state, such as a `Quarantine`
    # `duration` function, can see it. Interventions that act on a traced
    # contact time themselves from this, which is the same whatever the
    # trace action and the contact's own clinical course.
    _record_tracer!(ind, infector.id, trace_time)
    _record_trace_time!(ind, trace_time, :contact)
    ind.state[:traced] = true
    if isfinite(trace_time)
        push!(actions, _trace_action(ct, ind, trace_time))
    else
        # A trace with no arrival time (`Inf`) follows nobody up, so it is
        # applied directly rather than queued: `InterventionAction` admission
        # always skips a non-finite time (see `_admit_actions!`), and this
        # contact is traced, not scheduled or rationed, as `Quarantine`'s own
        # docstring says.
        _apply_trace!(ct.action, ind, state, trace_time, rng, :contact)
    end

    # Record how far the ring can still grow from this contact, so a
    # contact-of-contact one hop further out can time its own trace from
    # here. Only `depth > 1` rings expand. A node several cases reach keeps
    # the widest budget any of them gave it.
    ct.depth > 1 && (
        ind.state[:_ring_remaining] = max(
            get(ind.state, :_ring_remaining, 0)::Int, remaining
        )
    )
    return nothing
end

# The `InterventionAction` that follows up `ind` at `trace_time`, exactly as
# `apply_trace!` always has, so `Scheduled` and `CapacityConstrained` can
# admit it the same way they admit a dose.
function _trace_action(ct::ContactTracing, ind, trace_time)
    effect! = (person, at, state) -> _apply_trace!(ct.action, person, state, at, state.rng, :contact)
    return InterventionAction(ind, trace_time, effect!)
end

traces_contacts(::ContactTracing) = true

"""Trace the contacts of `infector`, applying the same eligibility,
probability, delay and action to each pair. Each contact is traced no earlier
than its `not_before` time when one is given. Called once per case."""
function trace_contacts!(
        ct::ContactTracing, state, infector, contacts, not_before = nothing
    )
    rng = state.rng
    # A ring grows past a member once. The first batch that reaches it spends
    # its budget on its contacts; a later one would trace the same pairs again,
    # which is a second attempt at the same relationship rather than a wider
    # ring. A batch with no contacts spends nothing, so a member whose contacts
    # are known only once it settles still grows the ring then.
    ring = ct.depth > 1 && is_traced(infector) &&
        !get(infector.state, :_ring_propagated, false)::Bool &&
        get(infector.state, :_ring_remaining, 0)::Int > 0
    actions = InterventionAction[]
    for (i, ind) in enumerate(contacts)
        ind.id == infector.id && continue
        _trace_pair!(
            ct, state, infector, ind, rng, actions;
            ring, not_before = not_before === nothing ? -Inf : not_before[i]
        )
    end
    ring && !isempty(contacts) && (infector.state[:_ring_propagated] = true)
    admit_trace_actions!(ct, state, actions)
    return nothing
end

"""
    admit_trace_actions!(tracer, state, actions)

For extension authors: admit the follow-up `actions` (a vector of
[`EpiBranch.InterventionAction`](@ref)) that a tracer proposed in its own
[`trace_contacts!`](@ref EpiBranch.trace_contacts!), so that whatever wraps it
can ration them. A tracer written outside the package calls this at the end
of its `trace_contacts!`, in place of [`apply_actions!`](@ref
EpiBranch.apply_actions!), which would ignore the wrapper.

The actions are admitted by `tracer` itself, unconditionally, unless a
[`CapacityConstrained`](@ref) or [`HeldCapacity`](@ref) is rationing it, which
is recorded in `state.scratch` for the span of the wrapped tracer's call
(so a [`Scheduled`](@ref) in between still finds it); they are then admitted
through that wrapper, against its budget. `tracer` is the innermost
intervention, the one the wrapper's `trace_contacts!` hands the call to.
"""
function admit_trace_actions!(tracer::AbstractIntervention, state, actions)
    isempty(actions) && return nothing
    admitter = get(state.scratch, (:trace_admitter, tracer), tracer)
    _admit_actions!(admitter, state, actions)
    return nothing
end

# Forwarded to the action so a `TraceAction` that needs to follow a contact
# up later, such as a test some days after the trace, can propose its own
# further actions through the ordinary schedule and capacity funnel, apart
# from the trace's own immediate follow-up `admit_trace_actions!` admits.
function intervention_actions(ct::ContactTracing, state, candidates)
    return intervention_actions(ct.action, state, candidates)
end

# `ContactTracing` always defines its own `intervention_actions` method,
# forwarding to `ct.action` above, so checking `ct` directly would credit
# every instance with discovering actions regardless of whether its action
# does: a plain `Quarantine` or `FlagOnly`, neither of which overrides
# `intervention_actions`, would still pay the race's per-settled-case cost
# for nothing. Checking the action's own method instead, against the
# fallback `TraceAction` declares, avoids that for actions with nothing to
# propose.
function _discovers_actions(ct::ContactTracing)
    return _has_own_method(intervention_actions, typeof(ct.action), TraceAction)
end

# Forwarded so a `TraceAction` can read what a contact's own settlement
# fixes, such as a test result that depends on whether the contact turned out
# to be infected. A contact traced from its own infector always settles after
# being traced, so whatever the trace recorded on it already stands; one
# reached only through a wider ring can already be settled, in which case
# this has already run for it (see the docstring).
function on_infection_settled!(ct::ContactTracing, individual, state, rng)
    return on_infection_settled!(ct.action, individual, state, rng)
end

"""A quarantined contact stops transmitting from its quarantine time. A
contact recorded with `FlagOnly` is removed only once [`Isolation`](@ref)
isolates it. A quarantine that is never released ends the person's
infectious period. With a finite `duration` (see [`Quarantine`](@ref)) each of
their contacts during quarantine is blocked and they transmit again once
released, as with [`Isolation`](@ref)."""
function infectious_removal_time(t::_Tracing, ind::Individual)
    get(ind.state, :quarantined, false) || return Inf
    t_removal = Inf
    for key in _trace_removal_keys(t.action)
        t_removal = min(t_removal, permanent_removal_time(ind, key))
    end
    return t_removal
end

# As for `Isolation`: a quarantine's recorded stretches are append-only.
binding_release(::_Tracing) = true

removal_gap_host_times(t::_Tracing) = removal_gap_host_times(t.action)
removal_gap_host_times(::TraceAction) = ()
removal_gap_host_times(::Quarantine) = (QUARANTINE_STRETCHES_KEY,)
removal_block_probability(ct::ContactTracing) = removal_block_probability(ct.action)

removal_gap_host_times_withdrawn(t::_Tracing) = removal_gap_host_times_withdrawn(t.action)
removal_gap_host_times_withdrawn(::TraceAction) = ()
# `apply_trace!` keeps the shared history `set_isolated!` writes as well as
# its own key, so a releasing quarantine leaves a stretch on
# `REMOVAL_STRETCHES_KEY` too. Since `QUARANTINE_STRETCHES_KEY` already names
# the same days, the shared one is withdrawn rather than read twice.
removal_gap_host_times_withdrawn(::Quarantine) = (REMOVAL_STRETCHES_KEY,)

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

# A quarantine that lapses leaves the window open above, and the stretches it
# removed the case for are blocked per contact here instead. One with no
# release is blocked here too, from its own start and never released.
function competing_risk(t::_Tracing, parent, contact, state)
    get(parent.state, :quarantined, false) || return nothing
    keys = _trace_removal_keys(t.action)
    length(keys) == 1 && return _removal_risks(parent, 1.0, only(keys))
    risks = Risk[]
    for key in keys
        more = _removal_risks(parent, 1.0, key)
        more === nothing || append!(risks, more)
    end
    return isempty(risks) ? nothing : risks
end

# A quarantine is a removal, so it reaches only the routes a removal can cut.
function risk_applies(::_Tracing, route)
    return route !== nothing && INTERVENTION_REMOVAL in route.until
end

"""With `depth > 1`, keep following traced contacts with ring budget left for
one more hop, whether or not they are themselves infected, so tracing can
reach their own contacts: an infected one may already be settled, with no
further turn at which it could be offered as a fresh ring root (see
`_walk_ring!` in sellke.jl). An uninfected one does not infect those contacts
either way. Returns the ids of those ring members (none for `depth == 1`)."""
function keep_active(ct::ContactTracing, state, targets, is_new)
    ct.depth > 1 || return ()
    ids = Int[]
    for t in targets
        is_traced(t) || continue
        get(t.state, :_ring_remaining, 0)::Int > 0 || continue
        push!(ids, t.id)
    end
    return ids
end

# ── SourceTracing intervention ─────────────────────────────────────────

"""
    SourceTracing(eligibility, probability, isolation_to_trace_delay, action)
    SourceTracing(; probability, isolation_to_trace_delay, action,
                  eligibility = SymptomaticParent())

Trace a case back to its own infector, once the case itself meets
`eligibility`, and act on the infector found this way (quarantine or just
record it). Where [`ContactTracing`](@ref) reaches forward, to the people a
case infects, `SourceTracing` reaches backward, to the person who infected it,
through the same [`TraceEligibility`](@ref), [`TraceRate`](@ref),
[`TraceDelay`](@ref) and [`TraceAction`](@ref) protocol: `eligibility` and its
timing are read off the case found eligible, as they are for
[`ContactTracing`](@ref), and the `action` is applied to its infector instead
of to a forward contact.

# Arguments
Same as [`ContactTracing`](@ref): `eligibility` (which cases are traced back,
default [`SymptomaticParent`](@ref)), `probability` (share of eligible cases
whose infector is found), `isolation_to_trace_delay` and `action` (required).

# Engine differences

By the time a case is itself confirmed and traced back, a branching process has
usually already generated its infector's secondary cases in one step, together
rather than over time, so there is nothing left for `SourceTracing` to shorten
there; it still records the link. On a continuous-time model — the household
and network processes, and the mass-action pools, where [`ContactTracing`](@ref)
has no pairwise structure to act along at all — an infector's contacts already
proposed but not yet resolved are checked against every intervention's
[`competing_risk`](@ref) as each is reached, so quarantining the infector here
blocks its remaining onward transmission without undoing any infection already
settled.

# Examples
```julia
# Quarantine a case's own infector once the case is isolated
SourceTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = 14.0)
)
```

# Requirements and output

Same requirements as [`ContactTracing`](@ref) for the same `eligibility`. The
infector found this way is marked traced (and quarantined, under
[`Quarantine`](@ref)), with its `trace_time` and, under the key
`:source_traced_by`, the id of the case whose tracing reached it — the reverse
of [`ContactTracing`](@ref)'s `:traced_by`, which always names an earlier-
infected node. `:source_traced_by` is kept separate because it names a later
one instead.
"""
struct SourceTracing{
        E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction,
    } <: _Tracing
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A

    # A plain probability or delay distribution becomes a constant rate or
    # delay, as for `ContactTracing`.
    function SourceTracing(
            eligibility::TraceEligibility, probability::Union{Real, TraceRate},
            isolation_to_trace_delay::Union{Distribution, TraceDelay},
            action::TraceAction
        )
        rate = _trace_rate(probability)
        delay = _trace_delay(isolation_to_trace_delay)
        return new{typeof(eligibility), typeof(rate), typeof(delay), typeof(action)}(
            eligibility, rate, delay, action
        )
    end
end

_tracer_owner(::SourceTracing) = :source

function SourceTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        action::TraceAction,
        eligibility::TraceEligibility = SymptomaticParent()
    )
    return SourceTracing(eligibility, probability, isolation_to_trace_delay, action)
end

# `trace_contacts!` below traces backward; the forward ring walk the race
# builds for `traces_contacts` has nothing for `SourceTracing` to do.
# `traces_source` asks the race to call `trace_contacts!` through a second
# route instead (see `trace_contacts!`'s docstring).
traces_contacts(::SourceTracing) = false
traces_source(::SourceTracing) = true

"""Trace `case` back to its own infector, named in `contacts` as the race
offers it (see [`traces_source`](@ref EpiBranch.traces_source)): the same
eligibility, probability, delay and action [`ContactTracing`](@ref) would use
for a forward trace, applied instead to the single contact offered here."""
function trace_contacts!(st::SourceTracing, state, case, contacts, not_before = nothing)
    rng = state.rng
    for (i, source) in enumerate(contacts)
        # The race also calls every intervention's `trace_contacts!`
        # unconditionally in the ordinary *forward* role, with `case`'s real
        # contacts — the one `traces_contacts(::SourceTracing) = false`
        # cannot prevent, since that trait only gates whether the race
        # bothers to gather contacts at all. Acting only on the single
        # element that actually is `case`'s own infector makes a forward call
        # a no-op here: no real contact of `case` ever has that id, since its
        # infector is, on every built-in model, already settled by the time
        # `case` reaches anyone new.
        source.id == case.parent_id || continue
        is_eligible(st.eligibility, case, source, state) || continue
        traces(st.trace_rate, case, source, state, rng) || continue
        trace_delay = draw_trace_delay(st.isolation_to_trace_delay, case, source, state, rng)
        base = trigger_time(st.eligibility, case, source, state)
        nb = not_before === nothing ? -Inf : not_before[i]
        trace_time = max(base, nb) + trace_delay
        # Recorded before the action runs, as `ContactTracing` does, so that
        # an action reading the contact's own state, such as a `Quarantine`
        # `duration` function, can see it.
        _record_trace_time!(source, trace_time, :source)
        _apply_trace!(st.action, source, state, trace_time, rng, :source)
        # The reverse of `ContactTracing`'s `:traced_by`: `source` was found
        # through `case`, a later-infected node, so it is kept under its own
        # key rather than break `compute_trace_level!`'s invariant that
        # `:traced_by` always points to an earlier one.
        source.state[:source_traced_by] = case.id
    end
    return nothing
end
