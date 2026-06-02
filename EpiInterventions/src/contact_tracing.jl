# ── Trait protocol for contact tracing ──────────────────────────────
#
# Contact tracing factors into four independent points of variation,
# each a dispatched seam. The default built-ins reproduce the original
# `ContactTracing(probability, delay, quarantine_on_trace)` behaviour;
# user-defined subtypes slot in via a single method.

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
Trace contacts of isolated symptomatic cases. The intervention is a
thin orchestrator over four traits:

- [`TraceEligibility`](@ref): who is eligible to be traced (default:
  [`SymptomaticParent`](@ref)).
- [`TraceRate`](@ref): whether tracing happens for an eligible
  contact (default: [`ConstantRate`](@ref)).
- [`TraceDelay`](@ref): the delay from parent isolation to the trace
  event (default: [`ConstantDelay`](@ref)).
- [`TraceAction`](@ref): what happens to the contact when tracing
  happens (default: [`Quarantine`](@ref); set
  `quarantine_on_trace = false` for [`FlagOnly`](@ref)).

Each trait is independently overridable. The convenience keyword
constructor preserves the original terse form
(`ContactTracing(probability = 0.8, delay = LogNormal(...))`).

Requires fields set by [`Isolation`](@ref): `:isolated`,
`:isolation_time`. Also requires `:asymptomatic` and `:onset_time`
(from `clinical_presentation()`).

Initialises: `:traced`, `:quarantined`.
"""
struct ContactTracing{
    E <: TraceEligibility, F <: TraceRate, D <: TraceDelay, A <: TraceAction} <:
       AbstractIntervention
    eligibility::E
    trace_rate::F
    delay::D
    action::A
end

function ContactTracing(;
        probability::Float64,
        delay::Distribution,
        quarantine_on_trace::Bool = true,
        eligibility::TraceEligibility = SymptomaticParent())
    return ContactTracing(
        eligibility,
        ConstantRate(probability),
        ConstantDelay(delay),
        quarantine_on_trace ? Quarantine() : FlagOnly()
    )
end

EpiBranchCore.required_fields(ct::ContactTracing) = _required_for_ct_eligibility(ct.eligibility)
EpiBranchCore.intervention_time(::ContactTracing, ind::Individual) = isolation_time(ind)

# Required-field validation dispatches on the eligibility trait so a
# custom eligibility that doesn't read these fields doesn't trip the
# validator.
_required_for_ct_eligibility(::SymptomaticParent) = [:isolated, :asymptomatic]
_required_for_ct_eligibility(::AlwaysEligible) = Symbol[]
_required_for_ct_eligibility(::TraceEligibility) = Symbol[]

function EpiBranchCore.reset!(::ContactTracing, ind::Individual)
    ind.state[:traced] = false
    ind.state[:quarantined] = false
    if is_isolated(ind)
        ind.state[:isolated] = false
        ind.state[:isolation_time] = Inf
    end
    return nothing
end

function EpiBranchCore.initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

function EpiBranchCore.apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        ind.parent_id > length(state.individuals) && continue
        parent = state.individuals[ind.parent_id]

        is_eligible(ct.eligibility, parent, ind, state) || continue
        traces(ct.trace_rate, parent, ind, state, rng) || continue

        trace_delay = draw_trace_delay(ct.delay, parent, ind, state, rng)
        trace_time = isolation_time(parent) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
