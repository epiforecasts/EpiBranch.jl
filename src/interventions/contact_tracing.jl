# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. The default built-ins reproduce the original
# `ContactTracing(probability, isolation_to_trace_delay, quarantine_on_trace)`
# behaviour; user-defined subtypes slot in via a single method.

"""
    TraceEligibility

Trait deciding whether a parent → contact pair is eligible to be
traced. Implementations override
[`is_eligible(eligibility, parent, contact, state)`](@ref).
"""
abstract type TraceEligibility end

"""
    is_eligible(eligibility, parent, contact, state) -> Bool
"""
is_eligible(::TraceEligibility, parent, contact, state) = true

"""Every contact is eligible."""
struct AlwaysEligible <: TraceEligibility end
is_eligible(::AlwaysEligible, parent, contact, state) = true

"""Eligible only when the parent is symptomatic and has been isolated.
Reproduces the original `ContactTracing` gate."""
struct SymptomaticParent <: TraceEligibility end
function is_eligible(::SymptomaticParent, parent, contact, state)
    !is_asymptomatic(parent) && is_isolated(parent)
end

# ── Built-in eligibility policies ─────────────────────────────────

"""Trace when parent develops symptoms (clinical suspicion)."""
struct OnSymptomOnset <: TraceEligibility end
function is_eligible(::OnSymptomOnset, parent, contact, state)
    !is_asymptomatic(parent)
end

"""Trace when parent tests positive (lab confirmation)."""
struct OnLabConfirmation <: TraceEligibility end
function is_eligible(::OnLabConfirmation, parent, contact, state)
    !is_asymptomatic(parent) && get(parent.state, :test_positive, false)
end

"""Trace when parent is isolated."""
struct OnIsolation <: TraceEligibility end
function is_eligible(::OnIsolation, parent, contact, state)
    !is_asymptomatic(parent) && is_isolated(parent)
end

"""Trace all contacts regardless of parent status."""
struct TraceEveryone <: TraceEligibility end
function is_eligible(::TraceEveryone, parent, contact, state)
    true
end

"""Never trace any contacts."""
struct TraceNobody <: TraceEligibility end
function is_eligible(::TraceNobody, parent, contact, state)
    false
end

# Backwards compatibility aliases
const SymptomaticParent = OnIsolation  # Original behavior
const AlwaysEligible = TraceEveryone
const NoTracing = TraceNobody

# ── Composition operators ──────────────────────────────────────────

"""Trace if ANY of the eligibility conditions are met (logical OR)."""
struct Any{T <: Tuple} <: TraceEligibility
    conditions::T
    Any(conditions...) = new{typeof(conditions)}(conditions)
end

function is_eligible(eligibility::Any, parent, contact, state)
    for condition in eligibility.conditions
        is_eligible(condition, parent, contact, state) && return true
    end
    return false
end

"""Trace only if ALL eligibility conditions are met (logical AND)."""
struct All{T <: Tuple} <: TraceEligibility
    conditions::T
    All(conditions...) = new{typeof(conditions)}(conditions)
end

function is_eligible(eligibility::All, parent, contact, state)
    for condition in eligibility.conditions
        is_eligible(condition, parent, contact, state) || return false
    end
    return true
end

"""Trace if primary condition is met, unless exclusion condition is also met."""
struct Unless{P <: TraceEligibility, E <: TraceEligibility} <: TraceEligibility
    primary::P
    exclusion::E
end

function is_eligible(eligibility::Unless, parent, contact, state)
    is_eligible(eligibility.primary, parent, contact, state) &&
    !is_eligible(eligibility.exclusion, parent, contact, state)
end

"""
    TraceRate

Trait deciding whether tracing happens for an eligible
contact. Implementations override
[`traces(rate, parent, contact, state, rng)`](@ref).
"""
abstract type TraceRate end

"""
    traces(rate, parent, contact, state, rng) -> Bool
"""
traces(::TraceRate, parent, contact, state, rng) = false

"""Bernoulli with constant probability `p`."""
struct ConstantRate <: TraceRate
    p::Float64
end
traces(r::ConstantRate, parent, contact, state, rng) = rand(rng) < r.p

"""
    TraceDelay

Trait giving the delay between the parent's isolation and the
contact being traced. Implementations override
[`draw_trace_delay(delay, parent, contact, state, rng)`](@ref).
"""
abstract type TraceDelay end

"""
    draw_trace_delay(delay, parent, contact, state, rng) -> Float64
"""
draw_trace_delay(::TraceDelay, parent, contact, state, rng) = 0.0

"""Delay drawn from a fixed distribution."""
struct ConstantDelay{D <: Distribution} <: TraceDelay
    dist::D
end
draw_trace_delay(d::ConstantDelay, parent, contact, state, rng) = float(rand(rng, d.dist))

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
Trace contacts based on when the parent becomes eligible for tracing.

## Eligibility policies

- `OnSymptomOnset()` - trace when parent develops symptoms
- `OnLabConfirmation()` - trace when parent tests positive
- `OnIsolation()` - trace when parent is isolated (default behavior)
- `TraceEveryone()` - trace all contacts regardless of parent status
- `TraceNobody()` - never trace any contacts

## Composition operators

- `Any(...)` - trace if any condition is met (logical OR)
- `All(...)` - trace only if all conditions are met (logical AND)
- `Unless(primary, exclusion)` - trace if primary is met unless exclusion is also met

## Examples

```julia
# Trace on symptoms (Guinea trial expansion)
ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine())

# Standard protocol (wait for lab confirmation)
ContactTracing(OnLabConfirmation(), 0.6, Exponential(2.0), Quarantine())

# Belt and braces (trace suspected OR confirmed)
ContactTracing(Any(OnSymptomOnset(), OnLabConfirmation()), 0.7, Exponential(1.5), Quarantine())
```

## Custom eligibility

Users can define custom eligibility types:

```julia
struct OnAgeSymptomOnset <: TraceEligibility
    min_age::Int
end

function is_eligible(e::OnAgeSymptomOnset, parent, contact, state)
    !is_asymptomatic(parent) && parent.age >= e.min_age
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

required_fields(ct::ContactTracing) = required_fields(ct.eligibility)
intervention_time(::ContactTracing, ind::Individual) = isolation_time(ind)

# Required-field validation dispatches on the eligibility trait
required_fields(::OnSymptomOnset) = [:asymptomatic]
required_fields(::OnLabConfirmation) = [:asymptomatic, :test_positive]
required_fields(::OnIsolation) = [:asymptomatic, :isolated]
required_fields(::TraceEveryone) = Symbol[]
required_fields(::TraceNobody) = Symbol[]
required_fields(::AlwaysEligible) = Symbol[]  # Backward compatibility
required_fields(::SymptomaticParent) = [:asymptomatic, :isolated]  # Backward compatibility
required_fields(::TraceEligibility) = Symbol[]  # Default for custom types

# Composition operators inherit requirements from their components
function required_fields(eligibility::Any)
    reduce(union, [required_fields(c) for c in eligibility.conditions]; init=Symbol[])
end

function required_fields(eligibility::All)
    reduce(union, [required_fields(c) for c in eligibility.conditions]; init=Symbol[])
end

function required_fields(eligibility::Unless)
    union(required_fields(eligibility.primary), required_fields(eligibility.exclusion))
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
        parent = state.individuals[ind.parent_id]

        is_eligible(ct.eligibility, parent, ind, state) || continue
        traces(ct.trace_rate, parent, ind, state, rng) || continue

        trace_delay = draw_trace_delay(ct.isolation_to_trace_delay, parent, ind, state, rng)
        trace_time = isolation_time(parent) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
