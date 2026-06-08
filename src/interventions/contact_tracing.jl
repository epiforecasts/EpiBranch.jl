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

# ── Additional eligibility policies ─────────────────────────────────

"""Trace when parent is isolated (lab-confirmed cases)."""
const OnIsolation = SymptomaticParent  # Alias for backwards compatibility

"""Trace when parent develops symptoms (clinical suspicion, pre-lab)."""
struct OnSymptomOnset <: TraceEligibility end
function is_eligible(::OnSymptomOnset, parent, contact, state)
    !is_asymptomatic(parent)
end

"""Trace any detected case, even if asymptomatic."""
struct OnCaseDetection <: TraceEligibility end
function is_eligible(::OnCaseDetection, parent, contact, state)
    true  # Any detected case can trigger tracing
end

"""Never trace (explicit no-tracing policy)."""
struct NoTracing <: TraceEligibility end
function is_eligible(::NoTracing, parent, contact, state)
    false
end

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
Trace contacts. You can set different rules for when tracing happens.

## Options

- `OnSymptomOnset()` - trace when someone gets sick
- `OnIsolation()` - wait for lab confirmation (default)
- `OnCaseDetection()` - trace any detected case
- `Any(...)`, `All(...)`, `Unless(...)` - combine rules

## Examples

```julia
# Trace on symptoms (faster but less specific)
ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine())

# Wait for lab results (slower but more specific)
ContactTracing(OnIsolation(), 0.6, Exponential(2.0), Quarantine())

# Trace both suspected and confirmed cases
ContactTracing(Any(OnSymptomOnset(), OnIsolation()), 0.7, Exponential(1.5), Quarantine())
```

Needs `:isolated`, `:isolation_time` from `Isolation` and `:asymptomatic`, `:onset_time`
from `clinical_presentation()`. Sets `:traced`, `:quarantined`.
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
required_fields(::SymptomaticParent) = [:isolated, :asymptomatic]
required_fields(::OnSymptomOnset) = [:asymptomatic]
required_fields(::OnCaseDetection) = Symbol[]
required_fields(::NoTracing) = Symbol[]
required_fields(::AlwaysEligible) = Symbol[]
required_fields(::TraceEligibility) = Symbol[]

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
