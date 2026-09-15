# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. The default built-ins reproduce the original
# `ContactTracing(probability, isolation_to_trace_delay, quarantine_on_trace)`
# behaviour; user-defined subtypes slot in via a single method.

"""
    TraceEligibility

Trait deciding whether a infector → contact pair is eligible to be
traced. Implementations override
[`is_eligible(eligibility, infector, contact, state)`](@ref).
"""
abstract type TraceEligibility end

"""
    is_eligible(eligibility, infector, contact, state) -> Bool
"""
is_eligible(::TraceEligibility, infector, contact, state) = true

# ── Built-in eligibility policies ─────────────────────────────────
#
# Each built-in is an *atomic* predicate — it tests one thing about the
# infector. Compose them with the boolean operators `&`, `|`, `!` (see
# below): e.g. `OnSymptomOnset() & !OnIsolation()` for "symptomatic but
# not yet isolated". Keeping predicates atomic is what makes that
# composition read correctly.

"""Trace when the infector is symptomatic (clinical suspicion), timed
from symptom onset, so tracing can start before lab confirmation, or
without it. See [`trigger_time`](@ref EpiBranch.trigger_time)."""
struct OnSymptomOnset <: TraceEligibility end
is_eligible(::OnSymptomOnset, infector, contact, state) = !is_asymptomatic(infector)

"""Trace when the infector has tested positive (lab confirmation)."""
struct OnLabConfirmation <: TraceEligibility end
function is_eligible(::OnLabConfirmation, infector, contact, state)
    get(infector.state, :test_positive, false)
end

"""Trace when the infector has been isolated."""
struct OnIsolation <: TraceEligibility end
is_eligible(::OnIsolation, infector, contact, state) = is_isolated(infector)

"""Trace every contact, regardless of infector status."""
struct TraceEveryone <: TraceEligibility end
is_eligible(::TraceEveryone, infector, contact, state) = true

"""Never trace any contacts."""
struct TraceNobody <: TraceEligibility end
is_eligible(::TraceNobody, infector, contact, state) = false

"""Original default gate: infector symptomatic *and* isolated. Equivalent
to `OnSymptomOnset() & OnIsolation()`; kept as a named type for
backwards compatibility (it is the default `eligibility`)."""
struct SymptomaticParent <: TraceEligibility end
function is_eligible(::SymptomaticParent, infector, contact, state)
    !is_asymptomatic(infector) && is_isolated(infector)
end

# `AlwaysEligible` and `NoTracing` were the previous names for tracing
# everyone / no-one; their semantics are identical to the new policies,
# so they are aliases.
"""Alias for [`TraceEveryone`](@ref): every contact is eligible."""
const AlwaysEligible = TraceEveryone

"""Alias for [`TraceNobody`](@ref): no contact is eligible."""
const NoTracing = TraceNobody

# ── Composition ────────────────────────────────────────────────────
#
# Policies form a boolean algebra over the infector predicate. The
# wrapper types below are normally built through the operators `&`, `|`,
# `!` rather than by name, so user code reads as ordinary boolean logic:
#
#     OnSymptomOnset() | OnLabConfirmation()    # suspected or confirmed
#     OnSymptomOnset() & !OnIsolation()         # symptomatic, not isolated

"""Eligible if **any** wrapped policy is. Build with `|`."""
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

"""Eligible only if **all** wrapped policies are. Build with `&`."""
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

"""Eligible only if **none** of the wrapped policies are. `!policy` is
the single-policy shorthand. (Named `NoneOf` to parallel `AnyOf`/`AllOf`
and to avoid colliding with `DataFrames.Not`.)"""
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

"""
    trigger_time(eligibility, infector, contact, state) -> Float64
    trigger_time(eligibility, infector, state) -> Float64

The time tracing from `infector` starts under this eligibility policy,
for `contact` in the four-argument form; [`ContactTracing`](@ref) adds
its delay to it.
Defaults to the infector's isolation time (the historical default).
[`OnSymptomOnset`](@ref) overrides this to onset time, so suspicion-based
tracing starts at symptom onset instead of waiting for isolation or
confirmation.

`ContactTracing` calls the four-argument form with the contact being
traced, as it does for [`is_eligible`](@ref EpiBranch.is_eligible),
[`traces`](@ref EpiBranch.traces) and
[`draw_trace_delay`](@ref EpiBranch.draw_trace_delay). For a single
policy the four-argument form defaults to the three-argument one. A
custom policy timed from another event of the infector defines a
three-argument method, and one timed from the contact defines a
four-argument method.

The combinators take their time from the wrapped conditions, as decided
by [`is_eligible`](@ref EpiBranch.is_eligible). Each condition is either
not met, met at a time, or met with no time of its own:

- A negation that holds, such as `!OnIsolation()` for an infector not yet
  isolated, is met with no time of its own. A negation that does not hold
  is not met. A negated combinator is timed as its De Morgan form, so
  `!(a & b)` is timed as `!a | !b`, and `!!a` as `a`.
- [`AllOf`](@ref) is not met if any condition is not met. Otherwise it is
  met at the latest time among its timed conditions, so
  `OnSymptomOnset() & !OnIsolation()` traces from onset. If none of its
  conditions is timed, the `AllOf` is met with no time of its own.
- [`AnyOf`](@ref) is not met if no condition is met. If one of its
  conditions is met with no time of its own, so is the `AnyOf`, and inside
  an `AllOf` it sets no time. Otherwise it is met at the earliest time
  among its met conditions.
- A policy met with no time of its own starts the trace at the default
  trigger time, the infector's isolation as for [`TraceEveryone`](@ref),
  or earlier if one of its timed branches is met earlier. So
  `OnSymptomOnset() | !OnIsolation()` traces a case that is never
  isolated from its onset.

A combinator that is not met gives `Inf` (never), and a `NaN` trigger
time from a wrapped condition counts as never met. A single policy's
`trigger_time` does not check [`is_eligible`](@ref EpiBranch.is_eligible),
so for a policy that is not met it still returns the policy's usual time.
`ContactTracing` only times a contact once `is_eligible` holds.

Two policies that are met get the same trigger time if one is rewritten
into the other by De Morgan's laws, double negation, commutativity,
associativity or distributing `&` over `|` outside a negation, and
`TraceNobody() | p` is timed as `p`. The exception is a
policy whose own trigger time is `NaN`, which gives `Inf` once wrapped in
a combinator.

Other logically equal policies can differ, for two reasons. A negation
that holds has no time of its own, so a rewrite that adds or removes such
a branch changes which times count, and `TraceEveryone()` is timed at
isolation, so adding or removing it can move the time. With
`S = OnSymptomOnset()`, `L = OnLabConfirmation()` and `I = OnIsolation()`,
for a case with onset at 4 that is isolated at 9 and never lab-confirmed:

- Distributing inside a negation: `!(L & (!I | !S))` triggers at 9 and
  `!((L & !I) | (L & !S))` at 4.
- Absorption by a branch with no time of its own: `!L | (!L & S)`
  triggers at 4 and `!L` at 9.
- A condition joined with its negation: `S & (I | !I)` triggers at 9 and
  `S` at 4.
- Joining `TraceEveryone()`: `S & TraceEveryone()` triggers at 9 and `S`
  at 4, and `S | TraceEveryone()` triggers at 4 and `TraceEveryone()`
  at 9.
- `!TraceNobody()` behaves as `TraceEveryone()` only at the top level:
  `S & !TraceNobody()` triggers at 4 and `S & TraceEveryone()` at 9.

The four-argument form checks each wrapped condition against the contact
and times it with its four-argument method. The three-argument form times
each wrapped condition with its three-argument method and checks it with
`nothing` in place of the contact, so `trigger_time(p, infector, state)`,
`trigger_time(!!p, infector, state)` and
`trigger_time(AllOf(p), infector, state)` agree for any policy `p` that
is met and whose own trigger time is not `NaN`. A combinator that wraps
a policy whose `is_eligible` reads the contact therefore cannot be
evaluated through the three-argument form; use the four-argument form.
"""
trigger_time(::TraceEligibility, infector, state) = isolation_time(infector)
trigger_time(::OnSymptomOnset, infector, state) = onset_time(infector)
function trigger_time(e::Union{AnyOf, AllOf, NoneOf}, infector, state)
    _combined_time(_WithoutContact(), e, infector, nothing, state)
end

function trigger_time(e::TraceEligibility, infector, contact, state)
    trigger_time(e, infector, state)
end

function trigger_time(e::Union{AnyOf, AllOf, NoneOf}, infector, contact, state)
    _combined_time(_WithContact(), e, infector, contact, state)
end

# Which `trigger_time` method times the policies inside a combinator: the
# form the combinator was called with, so that wrapping a policy does not
# change which of its methods is used.
struct _WithContact end
struct _WithoutContact end
function _single_time(::_WithContact, e, infector, contact, state)
    trigger_time(e, infector, contact, state)
end
function _single_time(::_WithoutContact, e, infector, contact, state)
    trigger_time(e, infector, state)
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
    negated ? _all_met(cx, e.conditions, true) : _any_met(cx, e.conditions, false)
end

function _met_time(cx::_TimingContext, e::AllOf, negated::Bool)
    negated ? _any_met(cx, e.conditions, true) : _all_met(cx, e.conditions, false)
end

# `NoneOf(a, b)` is `!a & !b`, and its negation is `a | b`.
function _met_time(cx::_TimingContext, e::NoneOf, negated::Bool)
    negated ? _any_met(cx, e.conditions, false) : _all_met(cx, e.conditions, true)
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
    _all_met(cx, conditions, negated, oftype(cx.never, -Inf), cx.never, false)
end
function _all_met(cx, ::Tuple{}, negated, latest, earliest, timed)
    timed ? (latest, false) : (earliest, true)
end
function _all_met(cx, conditions::Tuple, negated, latest, earliest, timed)
    tc, uc = _met_time(cx, first(conditions), negated)
    rest = Base.tail(conditions)
    uc && return _all_met(cx, rest, negated, latest, min(earliest, tc), timed)
    return _all_met(cx, rest, negated, max(latest, tc), earliest, true)
end

"""
    TraceRate

Trait deciding whether tracing happens for an eligible
contact. Implementations override
[`traces(rate, infector, contact, state, rng)`](@ref).
"""
abstract type TraceRate end

"""
    traces(rate, infector, contact, state, rng) -> Bool
"""
traces(::TraceRate, infector, contact, state, rng) = false

"""Bernoulli with constant probability `p`."""
struct ConstantRate <: TraceRate
    p::Float64
end
traces(r::ConstantRate, infector, contact, state, rng) = rand(rng) < r.p

"""
    TraceDelay

Trait giving the delay between the infector's isolation and the
contact being traced. Implementations override
[`draw_trace_delay(delay, infector, contact, state, rng)`](@ref).
"""
abstract type TraceDelay end

"""
    draw_trace_delay(delay, infector, contact, state, rng) -> Float64
"""
draw_trace_delay(::TraceDelay, infector, contact, state, rng) = 0.0

"""Delay drawn from a fixed distribution."""
struct ConstantDelay{D <: Distribution} <: TraceDelay
    dist::D
end
draw_trace_delay(d::ConstantDelay, infector, contact, state, rng) = float(rand(rng, d.dist))

"""
    TraceAction

Trait describing what happens to a contact once tracing happens.
Implementations override
[`apply_trace!(action, contact, state, trace_time, rng)`](@ref).
"""
abstract type TraceAction end

"""
    apply_trace!(action, contact, state, trace_time, rng)
"""
apply_trace!(::TraceAction, contact, state, trace_time, rng) = nothing

"""Quarantine the traced contact: set `:traced`, `:quarantined`, and
isolate them at the trace time (or the earlier of the trace time and
any pre-existing self-reporting isolation time)."""
struct Quarantine <: TraceAction end
function apply_trace!(::Quarantine, contact, state, trace_time, rng)
    contact.state[:traced] = true
    contact.state[:quarantined] = true
    if is_isolated(contact)
        set_isolated!(contact, min(isolation_time(contact), trace_time))
    else
        set_isolated!(contact, trace_time)
    end
    return nothing
end

"""Flag the contact as traced without quarantining them. If the
contact has a known onset time, record a `:traced_isolation_time` so
[`Isolation`](@ref) can later pick the earlier of self-reporting and
tracing."""
struct FlagOnly <: TraceAction end
function apply_trace!(::FlagOnly, contact, state, trace_time, rng)
    contact.state[:traced] = true
    contact.state[:quarantined] = false
    ind_onset = onset_time(contact)
    if !isnan(ind_onset)
        traced_iso = max(ind_onset, trace_time)
        contact.state[:traced_isolation_time] = traced_iso
    end
    return nothing
end

# ── ContactTracing intervention ──────────────────────────────────────

"""
Trace contacts based on when the infector becomes eligible for tracing.

## Eligibility policies

Atomic predicates on the infector:

- `OnSymptomOnset()` — infector is symptomatic
- `OnLabConfirmation()` — infector has tested positive
- `OnIsolation()` — infector has been isolated
- `TraceEveryone()` / `TraceNobody()` — trace all / none

Combine them with the boolean operators `&`, `|`, `!`:

```julia
OnSymptomOnset() | OnLabConfirmation()    # suspected or confirmed
OnSymptomOnset() & !OnIsolation()         # symptomatic, not yet isolated
```

## Examples

The terse positional form takes an eligibility policy, a trace
probability, a delay distribution, and (optionally) an action:

```julia
# Trace on symptoms (no wait for confirmation)
ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0))

# Standard protocol (wait for lab confirmation)
ContactTracing(OnLabConfirmation(), 0.6, Exponential(2.0))

# Belt and braces (trace suspected OR confirmed)
ContactTracing(OnSymptomOnset() | OnLabConfirmation(), 0.7, Exponential(1.5))
```

The keyword form keeps the original default (`SymptomaticParent`,
i.e. symptomatic and isolated) and is convenient when only the
probability and delay vary:

```julia
ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(1.0))
```

## Custom eligibility

Users can define custom eligibility types. Per-individual attributes
live in `infector.state` (see [`clinical_presentation`](@ref) /
[`demographics`](@ref)), so read them with `get`:

```julia
struct SymptomaticOver65 <: TraceEligibility end

function is_eligible(::SymptomaticOver65, infector, contact, state)
    !is_asymptomatic(infector) && get(infector.state, :age, 0) >= 65
end
```

## Ring depth

`depth` sets how many contact hops out the trace reaches (default `1`,
direct contacts only; must be at least `1`). With `depth = 2` the traced contacts of a case
keep generating their own contacts, which are traced in turn: the
contacts-of-contacts that a level-2 ring vaccination targets. Each
infected, eligible case seeds a fresh ring of radius `depth`; uninfected
ring members stay active for one more generation so the ring can grow
past them (see [`keep_active`](@ref EpiBranch.keep_active)), without
infecting their contacts (the [`InfectiousSource`](@ref
EpiBranch.InfectiousSource) default). Pair with [`RingVaccination`](@ref)
to vaccinate the whole ring:

```julia
[ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0); depth = 2),
 RingVaccination(efficacy = 0.9)]
```

Needs `:asymptomatic`, `:onset_time` from `clinical_presentation()` and optionally
`:isolated`, `:isolation_time`, `:test_positive` depending on eligibility type.
Sets `:traced`, `:quarantined`. With `depth > 1` also sets `:trace_time`
and `:ring_remaining` to carry the ring outward.
"""
struct ContactTracing{
    E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction} <:
       AbstractIntervention
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A
    depth::Int

    # Validate the ring radius once, here, so every construction path
    # (all the convenience constructors below funnel through this) rejects
    # depth < 1 rather than silently behaving as depth 1.
    function ContactTracing(eligibility::E, trace_rate::F,
            isolation_to_trace_delay::D, action::A,
            depth::Integer) where {
            E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction}
        depth >= 1 ||
            throw(ArgumentError("ContactTracing depth must be at least 1, got $depth"))
        return new{E, F, D, A}(
            eligibility, trace_rate, isolation_to_trace_delay, action, Int(depth))
    end
end

# Fully-typed form (eligibility + trait objects) with a default depth, so
# callers that build the traits directly need not pass `depth`.
function ContactTracing(eligibility::TraceEligibility, trace_rate::TraceRate,
        isolation_to_trace_delay::TraceDelay, action::TraceAction;
        depth::Integer = 1)
    return ContactTracing(
        eligibility, trace_rate, isolation_to_trace_delay, action, Int(depth))
end

function ContactTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        quarantine_on_trace::Bool = true,
        eligibility::TraceEligibility = SymptomaticParent(),
        depth::Integer = 1)
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
function ContactTracing(eligibility::TraceEligibility, probability::Real,
        isolation_to_trace_delay::Distribution, action::TraceAction = Quarantine();
        depth::Integer = 1)
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
    reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end
function required_fields(e::AllOf)
    reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end
function required_fields(e::NoneOf)
    reduce(union, (required_fields(c) for c in e.conditions); init = Symbol[])
end

function reset!(::ContactTracing, ind::Individual)
    ind.state[:traced] = false
    ind.state[:quarantined] = false
    haskey(ind.state, :traced_by) && delete!(ind.state, :traced_by)
    haskey(ind.state, :trace_level) && delete!(ind.state, :trace_level)
    is_isolated(ind) && clear_isolated!(ind)
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
    propagate = ct.depth > 1 && !seed && is_traced(infector) &&
                get(infector.state, :ring_remaining, 0)::Int > 0
    (seed || propagate) || return nothing
    traces(ct.trace_rate, infector, ind, state, rng) || return nothing

    trace_delay = draw_trace_delay(
        ct.isolation_to_trace_delay, infector, ind, state, rng)
    base = seed ? trigger_time(ct.eligibility, infector, ind, state) :
           get(infector.state, :trace_time, isolation_time(infector))
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

    # Record how far the ring can still grow from this contact, and
    # when it was traced, so a contact-of-contact one hop further out
    # can time its own trace from here. Only `depth > 1` rings expand.
    if ct.depth > 1
        ind.state[:trace_time] = trace_time
        ind.state[:ring_remaining] = seed ? ct.depth - 1 :
                                     get(infector.state, :ring_remaining, 0)::Int - 1
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

"""Trace the contacts a case reaches on the continuous-time path. The
generation engine reads each contact's infector off its `parent_id`; here the
nodes pre-exist and the infector is the case the race has just finalised, so it
is passed in and the same per-pair policy applied, with each contact traced no
earlier than its `not_before` time when one is given."""
function trace_contacts!(
        ct::ContactTracing, state, infector, contacts, not_before = nothing)
    rng = state.rng
    for (i, ind) in enumerate(contacts)
        ind.id == infector.id && continue
        _trace_pair!(ct, state, infector, ind, rng;
            not_before = not_before === nothing ? -Inf : not_before[i])
    end
    return nothing
end

"""A quarantined contact is out of onward transmission from its quarantine
time, which is how tracing reaches the infectious window on the continuous-time
models. Contacts merely flagged (`FlagOnly`) write `:traced_isolation_time`
instead, and [`Isolation`](@ref) turns that into the removal, exactly as on the
generation-based path."""
function infectious_removal_time(::ContactTracing, ind::Individual)
    get(ind.state, :quarantined, false) ? isolation_time(ind) : Inf
end

"""Keep uninfected ring members generating contacts so the ring can
reach contacts-of-contacts. A traced contact with ring budget left
stays active for one more generation; the [`InfectiousSource`](@ref
EpiBranch.InfectiousSource) default keeps it from infecting those
contacts. Infected cases stay active regardless, so only the uninfected
fringe is returned. Empty for `depth == 1` (direct contacts only)."""
function keep_active(ct::ContactTracing, state, targets, is_new)
    ct.depth > 1 || return ()
    ids = Int[]
    for t in targets
        is_infected(t) && continue
        is_traced(t) || continue
        get(t.state, :ring_remaining, 0)::Int > 0 || continue
        push!(ids, t.id)
    end
    return ids
end
