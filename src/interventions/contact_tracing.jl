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
from symptom onset — so tracing fires before, or without, lab
confirmation. See [`trigger_time`](@ref)."""
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
# When the trace fires is read from the same eligibility policy that
# decides whether it fires: the trace is timed from when the infector
# first *meets* the condition. So `OnSymptomOnset` times from symptom
# onset (suspicion-based tracing, which fires before — or without — lab
# confirmation), while the historical default and the isolation/
# confirmation policies time from isolation. `ContactTracing` adds its
# delay to this. Combinators compose: an `AllOf` fires once its last
# condition is met (the latest trigger), an `AnyOf` once its first is.

"""
    trigger_time(eligibility, infector, state) -> Float64

The time the trace fires for `infector` under this eligibility policy;
[`ContactTracing`](@ref) adds its delay to it. Defaults to the infector's
isolation time (the historical anchor); [`OnSymptomOnset`](@ref) overrides
to onset time so suspicion-based tracing fires from symptom onset rather
than waiting for isolation/confirmation.
"""
trigger_time(::TraceEligibility, infector, state) = isolation_time(infector)
trigger_time(::OnSymptomOnset, infector, state) = onset_time(infector)
function trigger_time(e::AllOf, infector, state)
    maximum(trigger_time(c, infector, state) for c in e.conditions)
end
function trigger_time(e::AnyOf, infector, state)
    minimum(trigger_time(c, infector, state) for c in e.conditions)
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

Needs `:asymptomatic`, `:onset_time` from `clinical_presentation()` and optionally
`:isolated`, `:isolation_time`, `:test_positive` depending on eligibility type.
Sets `:traced`, `:quarantined`.
"""
struct ContactTracing{
    E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction} <:
       AbstractIntervention
    eligibility::E
    trace_rate::F
    isolation_to_trace_delay::D
    action::A
end

function ContactTracing(;
        probability::Float64,
        isolation_to_trace_delay::Distribution,
        quarantine_on_trace::Bool = true,
        eligibility::TraceEligibility = SymptomaticParent())
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(isolation_to_trace_delay),
        quarantine_on_trace ? Quarantine() : FlagOnly()
    )
end

# Terse positional form: an eligibility policy with a constant trace
# probability and a constant delay distribution. The fully-typed inner
# constructor (taking `TraceRate`/`TraceDelay` objects) is unaffected —
# `probability::Real` and `delay::Distribution` do not match those.
function ContactTracing(eligibility::TraceEligibility, probability::Real,
        isolation_to_trace_delay::Distribution, action::TraceAction = Quarantine())
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(isolation_to_trace_delay),
        action
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
    if is_isolated(ind)
        ind.state[:isolated] = false
        ind.state[:isolation_time] = Inf
    end
    return nothing
end

function initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        ind.parent_id > length(state.individuals) && continue
        infector = state.individuals[ind.parent_id]

        is_eligible(ct.eligibility, infector, ind, state) || continue
        traces(ct.trace_rate, infector, ind, state, rng) || continue

        trace_delay = draw_trace_delay(
            ct.isolation_to_trace_delay, infector, ind, state, rng)
        trace_time = trigger_time(ct.eligibility, infector, state) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
