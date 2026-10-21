# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. The default built-ins reproduce the original
# `ContactTracing(probability, isolation_to_trace_delay, action)`
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

What happens to a contact once traced: [`Quarantine`](@ref) or
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
    # Stash what isolation, if any, stood before this quarantine, so a
    # `Scheduled` reset can restore exactly that rather than wiping out an
    # isolation another intervention recorded (the mirror image of what
    # `Isolation`'s own `reset!` does for a quarantine it is layered over). A
    # contact already quarantined by an earlier trace keeps the baseline
    # recorded before that first trace, instead of taking this trace's own
    # combined outcome as the new baseline.
    if !get(contact.state, :_quarantined_by_tracing, false)::Bool
        contact.state[:_isolation_time_before_quarantine] = isolation_time(contact)
        contact.state[:_isolation_release_time_before_quarantine] = isolation_release_time(contact)
        if _isolation_unrecorded(contact)
            contact.state[:_isolation_unrecorded_before_quarantine] = true
        else
            delete!(contact.state, :_isolation_unrecorded_before_quarantine)
        end
    end
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
    contact.state[:_quarantined_by_tracing] = true
    return nothing
end

# Where `TestAndRelease` keeps the day a contact is due to be tested, the
# exact `(start, release)` its own quarantine recorded for a negative test to
# shorten, whether the test has been granted (by a wrapping
# `CapacityConstrained`, or unconditionally with none), and whether the
# outcome has been decided.
const TEST_RELEASE_DAY_KEY = :_test_release_day
const TEST_RELEASE_BACKSTOP_KEY = :_test_release_backstop
const TEST_RELEASE_GRANTED_KEY = :_test_release_granted
const TEST_RELEASE_RESOLVED_KEY = :_test_release_resolved

"""
    TestAndRelease(; duration, test_day, sensitivity = 1.0)

Quarantine a traced contact as [`Quarantine`](@ref) does, for up to `duration`
days, but release them early if a test taken `test_day` days after the trace
comes back negative.

A test cannot be read against an infection the race has not yet settled, so
the outcome is decided once the contact's own infection time is fixed (see
[`on_infection_settled!`](@ref EpiBranch.on_infection_settled!)): a contact
not yet infected by `test_day` tests negative, and one already infected by
then tests positive with probability `sensitivity`, the test's chance of
detecting a true infection (perfect specificity is assumed). A positive test,
or one never reached, leaves the full `duration` standing; a negative one
shortens the quarantine to end at `test_day` instead.

`duration` must stay finite: an indefinite quarantine closes the person's
infectious window outright (see [`Quarantine`](@ref)), and a closed window
cannot reopen for a later test to shorten.

`test_day` and `sensitivity` take the same forms as `duration`: a number, a
distribution, or a function `(rng, ind) -> ...`, drawn once per trace.

Wrap the [`ContactTracing`](@ref) in [`CapacityConstrained`](@ref) to ration
how many tests are available per day; a contact turned down keeps its full
`duration`.

# Examples
```julia
# Quarantine for up to 14 days, released on a negative test taken on day 3
ContactTracing(
    OnSymptomOnset(), 0.8, Exponential(1.0),
    TestAndRelease(duration = 14.0, test_day = 3.0, sensitivity = 0.9)
)
```
"""
struct TestAndRelease{D, TD, S} <: TraceAction
    duration::D
    test_day::TD
    sensitivity::S

    function TestAndRelease(duration::D, test_day::TD, sensitivity::S) where {D, TD, S}
        duration isa Real && !isfinite(duration) && throw(
            ArgumentError(
                "`TestAndRelease`'s `duration` must be finite, got $duration. " *
                    "Use `Quarantine(duration = Inf)` for a quarantine with no " *
                    "test to shorten it."
            )
        )
        return new{D, TD, S}(duration, test_day, sensitivity)
    end
end
function TestAndRelease(; duration, test_day, sensitivity = 1.0)
    return TestAndRelease(duration, test_day, sensitivity)
end

# Only a contact's first trace through tracing schedules a test: a second one
# (ring propagation, or another infector reaching the same contact) extends
# the same quarantine rather than starting a fresh one (see `Quarantine`), and
# the backstop `apply_trace!` records below would no longer name an exact
# stretch to shorten once the two have merged. `:_quarantined_by_tracing` is
# what `Quarantine`'s own `apply_trace!` sets, shared by every `TraceAction`,
# so it is read before delegating to it.
function apply_trace!(t::TestAndRelease, contact, state, trace_time, rng)
    first_trace = !get(contact.state, :_quarantined_by_tracing, false)::Bool
    apply_trace!(Quarantine(t.duration), contact, state, trace_time, rng)
    (isfinite(trace_time) && first_trace) || return nothing
    # No quarantine stood on this contact before this call (a first trace,
    # and nothing else records under this key), so the stretch it just
    # recorded is this one alone, with nothing to merge it with.
    release = only(contact.state[QUARANTINE_STRETCHES_KEY])[2]
    contact.state[TEST_RELEASE_BACKSTOP_KEY] = (trace_time, release)
    contact.state[TEST_RELEASE_DAY_KEY] = trace_time + _sample_value(t.test_day, rng, contact)
    return nothing
end

# Read `ind.infection_time` to decide the test, trusting the caller to have
# checked it is final: `on_infection_settled!` calls this directly, under its
# own guarantee that the time is now fixed; the test's own grant
# (`intervention_actions`, below) calls it only once `is_settled` confirms the
# same for a contact already settled by the time the test is granted. Neither
# site may use `isfinite(ind.infection_time)` as that check instead — a
# proposal still being raced against others holds a provisional time there
# too (see `_proposal_blocked`), not yet the one the race will keep.
# Idempotent, so it is harmless if both sites end up calling it.
function _resolve_test_release!(t::TestAndRelease, ind, rng)
    get(ind.state, TEST_RELEASE_RESOLVED_KEY, false) && return nothing
    ind.state[TEST_RELEASE_RESOLVED_KEY] = true
    day = ind.state[TEST_RELEASE_DAY_KEY]
    positive = ind.infection_time <= day && action_draw!(ind, (t, :result)) do
        rand(rng) < _sample_value(t.sensitivity, rng, ind)
    end
    positive && return nothing
    start, backstop = ind.state[TEST_RELEASE_BACKSTOP_KEY]
    _shorten_quarantine!(ind, start, backstop, day)
    return nothing
end

# Shorten the quarantine `apply_trace!` recorded, from `(start, backstop)` to
# `(start, release)` instead, keeping both its own key and the shared removal
# history append-only: each is revised only by replacing the one stretch
# `apply_trace!` itself wrote there, found by its exact start and release,
# and only once (`_resolve_test_release!` calls this at most once per
# contact, before anything has read either key for this contact as a parent
# — see `on_infection_settled!`). A stretch that can no longer be found this
# way — merged with a later trace's, or combined with a standing isolation
# when it was first recorded — is left as it stands rather than guessed at.
function _shorten_quarantine!(ind, start, backstop, release)
    (release < backstop && release > start) || return nothing
    _retime_stretch!(ind, QUARANTINE_STRETCHES_KEY, start, backstop, release) || return nothing
    _retime_stretch!(ind, REMOVAL_STRETCHES_KEY, start, backstop, release)
    if ind.state[:isolation_time] == start && ind.state[:isolation_release_time] == backstop
        ind.state[:isolation_release_time] = release
    end
    return nothing
end

function _retime_stretch!(ind, key, start, old_release, new_release)
    stretches = get(ind.state, key, nothing)
    stretches === nothing && return false
    i = findfirst(==((start, old_release)), stretches)
    i === nothing && return false
    revised = copy(stretches)
    revised[i] = (start, new_release)
    ind.state[key] = revised
    return true
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
        _state_real(contact.state, :_traced_isolation_time, Inf), traced_iso
    )
    return nothing
end

# ── ContactTracing intervention ──────────────────────────────────────

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
  [`Quarantine`](@ref) for a given `duration`, [`TestAndRelease`](@ref) to
  release them early on a negative test, or [`FlagOnly`](@ref) to record the
  trace without quarantining. There is no default, because policies differ in
  whether and for how long traced contacts are quarantined.
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
        action::Union{TraceAction, Nothing} = nothing,
        quarantine_on_trace::Union{Bool, Nothing} = nothing,
        eligibility::TraceEligibility = SymptomaticParent(),
        depth::Integer = 1
    )
    action = _resolve_action(action, quarantine_on_trace)
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(isolation_to_trace_delay),
        action,
        Int(depth)
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

# Terse positional form: an eligibility policy with a constant trace
# probability and a constant delay distribution. The fully-typed inner
# constructor (taking `TraceRate`/`TraceDelay` objects) is unaffected —
# `probability::Real` and `delay::Distribution` do not match those.
function ContactTracing(
        eligibility::TraceEligibility, probability::Real,
        isolation_to_trace_delay::Distribution, action::TraceAction;
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

reset!(::ContactTracing, ind::Individual) = _reset_traced!(ind)

# Shared by `ContactTracing` and `SourceTracing`: both write the same tracing
# state on whoever `apply_trace!` acts on, forward or backward, through the
# same `TraceAction` protocol.
function _reset_traced!(ind::Individual)
    ind.state[:traced] = false
    ind.state[:quarantined] = false
    haskey(ind.state, :traced_by) && delete!(ind.state, :traced_by)
    haskey(ind.state, :source_traced_by) && delete!(ind.state, :source_traced_by)
    haskey(ind.state, :trace_level) && delete!(ind.state, :trace_level)
    haskey(ind.state, :trace_time) && delete!(ind.state, :trace_time)
    haskey(ind.state, :_ring_remaining) && delete!(ind.state, :_ring_remaining)
    haskey(ind.state, :_ring_propagated) && delete!(ind.state, :_ring_propagated)
    delete!(ind.state, QUARANTINE_STRETCHES_KEY)
    delete!(ind.state, TEST_RELEASE_DAY_KEY)
    delete!(ind.state, TEST_RELEASE_BACKSTOP_KEY)
    delete!(ind.state, TEST_RELEASE_GRANTED_KEY)
    delete!(ind.state, TEST_RELEASE_RESOLVED_KEY)
    _undo_quarantine_isolation!(ind)
    return nothing
end

# Undo only the isolation `Quarantine` itself recorded, restoring whatever
# stood before it (another intervention's isolation, or none), rather than
# clearing the field outright: a standing isolation from a separate
# intervention such as `Isolation` is not tracing's to undo. Mirrors
# `Isolation`'s own `reset!`, which protects a quarantine it was layered over
# the same way.
function _undo_quarantine_isolation!(ind::Individual)
    get(ind.state, :_quarantined_by_tracing, false)::Bool || return nothing
    previous = _state_real(ind.state, :_isolation_time_before_quarantine, Inf)
    if isfinite(previous)
        previous_release = _state_real(
            ind.state, :_isolation_release_time_before_quarantine, Inf
        )
        clear_isolated!(ind)
        set_isolated!(ind, previous; release_time = previous_release)
        get(ind.state, :_isolation_unrecorded_before_quarantine, false) &&
            (ind.state[:_isolation_unrecorded] = true)
    else
        clear_isolated!(ind)
    end
    delete!(ind.state, :_isolation_time_before_quarantine)
    delete!(ind.state, :_isolation_release_time_before_quarantine)
    delete!(ind.state, :_isolation_unrecorded_before_quarantine)
    ind.state[:_quarantined_by_tracing] = false
    return nothing
end

function initialise_individual!(::ContactTracing, individual, state)
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
function _record_trace_time!(ind::Individual, trace_time)
    isnan(trace_time) && return nothing
    ind.state[:trace_time] = min(_state_real(ind.state, :trace_time, Inf), trace_time)
    return nothing
end

# The trace attempts for a single infector → contact pair.
function _trace_pair!(
        ct::ContactTracing, state, infector, ind, rng; ring::Bool = false, not_before = -Inf
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
        _trace_attempt!(ct, state, infector, ind, rng, base, remaining, not_before)
    end
    if is_infected(infector) && is_eligible(ct.eligibility, infector, ind, state) &&
            !get(infector.state, :_ring_hop, false)::Bool
        base = trigger_time(ct.eligibility, infector, ind, state)
        _trace_attempt!(ct, state, infector, ind, rng, base, ct.depth - 1, not_before)
    end
    return nothing
end

# One attempt to trace `ind` from `infector`, the trace delay running from
# `base`, leaving the ring `remaining` hops to grow from `ind`.
function _trace_attempt!(ct::ContactTracing, state, infector, ind, rng, base, remaining, not_before)
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
    _record_trace_time!(ind, trace_time)
    apply_trace!(ct.action, ind, state, trace_time, rng)

    # Record the source this contact was traced from: the latest case whose
    # trace reached it, which on a tree is its parent. `compute_trace_level!`
    # walks it back to the index case post-run.
    ind.state[:traced_by] = infector.id

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
    for (i, ind) in enumerate(contacts)
        ind.id == infector.id && continue
        _trace_pair!(
            ct, state, infector, ind, rng;
            ring, not_before = not_before === nothing ? -Inf : not_before[i]
        )
    end
    ring && !isempty(contacts) && (infector.state[:_ring_propagated] = true)
    return nothing
end

"""A quarantined contact stops transmitting from its quarantine time. A
contact recorded with `FlagOnly` is removed only once [`Isolation`](@ref)
isolates it. A quarantine that is never released ends the person's
infectious period. With a finite `duration` (see [`Quarantine`](@ref)) each of
their contacts during quarantine is blocked and they transmit again once
released, as with [`Isolation`](@ref)."""
infectious_removal_time(ct::ContactTracing, ind::Individual) = _trace_removal_time(ct.action, ind)

# Shared by `ContactTracing` and `SourceTracing`, which both remove whoever
# `apply_trace!` acted on through the same `TraceAction` protocol; only which
# person ends up quarantined differs between forward and backward tracing.
function _trace_removal_time(action::TraceAction, ind::Individual)
    get(ind.state, :quarantined, false) || return Inf
    t = Inf
    for key in _trace_removal_keys(action)
        t = min(t, permanent_removal_time(ind, key))
    end
    return t
end

# As for `Isolation`: a quarantine's recorded stretches are append-only.
binding_release(::ContactTracing) = true

removal_gap_host_times(ct::ContactTracing) = removal_gap_host_times(ct.action)
removal_gap_host_times(::TraceAction) = ()
removal_gap_host_times(::Quarantine) = (QUARANTINE_STRETCHES_KEY,)
# `TestAndRelease` records under the same key, through the `Quarantine` it
# delegates to in `apply_trace!`.
removal_gap_host_times(::TestAndRelease) = (QUARANTINE_STRETCHES_KEY,)
removal_block_probability(ct::ContactTracing) = removal_block_probability(ct.action)

removal_gap_host_times_withdrawn(ct::ContactTracing) = removal_gap_host_times_withdrawn(ct.action)
removal_gap_host_times_withdrawn(::TraceAction) = ()
# `apply_trace!` keeps the shared history `set_isolated!` writes as well as
# its own key, so a releasing quarantine leaves a stretch on
# `REMOVAL_STRETCHES_KEY` too. Since `QUARANTINE_STRETCHES_KEY` already names
# the same days, the shared one is withdrawn rather than read twice.
removal_gap_host_times_withdrawn(::Quarantine) = (REMOVAL_STRETCHES_KEY,)
removal_gap_host_times_withdrawn(::TestAndRelease) = (REMOVAL_STRETCHES_KEY,)

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
function competing_risk(ct::ContactTracing, parent, contact, state)
    return _trace_competing_risk(ct.action, parent)
end

# Shared by `ContactTracing` and `SourceTracing`: see `_trace_removal_time`.
function _trace_competing_risk(action::TraceAction, parent)
    get(parent.state, :quarantined, false) || return nothing
    keys = _trace_removal_keys(action)
    length(keys) == 1 && return _removal_risks(parent, 1.0, only(keys))
    risks = Risk[]
    for key in keys
        more = _removal_risks(parent, 1.0, key)
        more === nothing || append!(risks, more)
    end
    return isempty(risks) ? nothing : risks
end

# A quarantine is a removal, so it reaches only the routes a removal can cut.
function risk_applies(::ContactTracing, route)
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

# A contact's `TestAndRelease` test cannot be read until its own infection
# time is fixed (see `_resolve_test_release!`); for a contact not yet settled
# when its test is granted, this is where that happens. A granted test
# already resolved (because the contact was already settled when it was
# granted) is a no-op here, as is a test never granted at all.
function on_infection_settled!(ct::ContactTracing{E, F, D, <:TestAndRelease}, ind, state, rng) where {E, F, D}
    get(ind.state, TEST_RELEASE_GRANTED_KEY, false) || return nothing
    _resolve_test_release!(ct.action, ind, rng)
    return nothing
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
    } <: AbstractIntervention
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A
end

function SourceTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        action::TraceAction,
        eligibility::TraceEligibility = SymptomaticParent()
    )
    return SourceTracing(
        eligibility, ConstantRate(probability), ConstantDelay(isolation_to_trace_delay), action
    )
end

# Terse positional form, as `ContactTracing`'s equivalent constructor is: an
# eligibility policy with a constant trace probability and a constant delay
# distribution. The fully-typed inner constructor (taking `TraceRate`/
# `TraceDelay` objects directly) is unaffected.
function SourceTracing(
        eligibility::TraceEligibility, probability::Real,
        isolation_to_trace_delay::Distribution, action::TraceAction
    )
    return SourceTracing(
        eligibility, ConstantRate(probability), ConstantDelay(isolation_to_trace_delay), action
    )
end

required_fields(st::SourceTracing) = required_fields(st.eligibility)
intervention_time(::SourceTracing, ind::Individual) = isolation_time(ind)
reset!(::SourceTracing, ind::Individual) = _reset_traced!(ind)

function initialise_individual!(::SourceTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

# `trace_contacts!` below traces backward; the forward ring walk the race
# builds for `traces_contacts` has nothing for `SourceTracing` to do.
# `traces_source` asks the race to call it through a second route instead
# (see `trace_contacts!`'s docstring).
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
        _record_trace_time!(source, trace_time)
        apply_trace!(st.action, source, state, trace_time, rng)
        # The reverse of `ContactTracing`'s `:traced_by`: `source` was found
        # through `case`, a later-infected node, so it is kept under its own
        # key rather than break `compute_trace_level!`'s invariant that
        # `:traced_by` always points to an earlier one.
        source.state[:source_traced_by] = case.id
    end
    return nothing
end

infectious_removal_time(st::SourceTracing, ind::Individual) = _trace_removal_time(st.action, ind)

# As for `ContactTracing`: a quarantine's recorded stretches are append-only.
binding_release(::SourceTracing) = true

removal_gap_host_times(st::SourceTracing) = removal_gap_host_times(st.action)
removal_gap_host_times_withdrawn(st::SourceTracing) = removal_gap_host_times_withdrawn(st.action)

function competing_risk(st::SourceTracing, parent, contact, state)
    return _trace_competing_risk(st.action, parent)
end

# A quarantine is a removal, so it reaches only the routes a removal can cut.
risk_applies(::SourceTracing, route) = route !== nothing && INTERVENTION_REMOVAL in route.until
