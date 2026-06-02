# ── Trait protocol for isolation ────────────────────────────────────
#
# Isolation's three points of variation, each a dispatched seam:
#
# - `IsolationEligibility`: who can be isolated at all
#   (default: symptomatic cases only).
# - `test_sensitivity`: probability an eligible individual tests
#   positive and so reaches isolation (a scalar / distribution /
#   function, sampled per individual at init time).
# - `delay`: time from onset to self-reported isolation (a
#   distribution drawn per individual).
#
# Post-isolation transmission stays a scalar parameter — it modifies
# the competing risk's block probability without changing the
# intervention's policy shape.

"""
    IsolationEligibility

Trait deciding whether an individual is eligible to be isolated based
on the structural gate (e.g. symptomatic vs all-cases). Implementations
override [`is_eligible_for_isolation(elig, individual, state)`](@ref).
Whether eligibility actually leads to isolation also depends on the
intervention's `test_sensitivity`.
"""
abstract type IsolationEligibility end

"""
    is_eligible_for_isolation(eligibility, individual, state) -> Bool
"""
is_eligible_for_isolation(::IsolationEligibility, individual, state) = true

"""Symptomatic cases only. Reproduces the original `Isolation` gate."""
struct SymptomaticOnly <: IsolationEligibility end
is_eligible_for_isolation(::SymptomaticOnly, ind, state) = !is_asymptomatic(ind)

"""Every case is eligible, including asymptomatic ones (mass-testing
scenarios)."""
struct AllCases <: IsolationEligibility end
is_eligible_for_isolation(::AllCases, ind, state) = true

# Required-field validation dispatches on the eligibility trait so a
# custom eligibility that doesn't read `:asymptomatic` doesn't trip
# the validator.
_required_for_eligibility(::SymptomaticOnly) = [:onset_time, :asymptomatic]
_required_for_eligibility(::AllCases) = [:onset_time]
_required_for_eligibility(::IsolationEligibility) = [:onset_time]

# ── Isolation intervention ──────────────────────────────────────────

"""
Isolate cases after a delay from symptom onset.

The structural gate (who can be isolated) is given by `eligibility`,
an [`IsolationEligibility`](@ref) trait. The default
[`SymptomaticOnly`](@ref) reproduces the previous behaviour
(symptomatic cases only).

`test_sensitivity` is the probability that an eligible individual
tests positive and so reaches isolation; it accepts a `Real`, a
`Distribution`, or a function `(rng, ind) -> Real` (sampled once per
individual at init time, stored as `:test_positive`).

`delay` is a `Distribution` from which the time from onset to
self-reported isolation is drawn per individual.

`post_isolation_transmission` ∈ [0, 1] sets the residual transmission
probability after isolation. The competing risk's `block_probability`
is `1 - post_isolation_transmission`.

Initialises: `:isolated`, `:isolation_time`, `:test_positive`.
"""
struct Isolation{E <: IsolationEligibility, D <: Distribution, S} <: AbstractIntervention
    eligibility::E
    delay::D
    test_sensitivity::S
    post_isolation_transmission::Float64
end

function Isolation(;
        delay::Distribution,
        eligibility::IsolationEligibility = SymptomaticOnly(),
        test_sensitivity::Union{Real, Distribution, Function} = 1.0,
        post_isolation_transmission::Real = 0.0)
    return Isolation(eligibility, delay, test_sensitivity,
        Float64(post_isolation_transmission))
end

required_fields(iso::Isolation) = _required_for_eligibility(iso.eligibility)
intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

"""Isolation blocks the parent → contact transmission when the parent's
isolation time is earlier than the contact's transmission time.
Residual transmission is governed by `post_isolation_transmission`:
`block_probability = 1 - post_isolation_transmission`."""
function competing_risk(iso::Isolation, parent, contact, state)
    iso_t = isolation_time(parent)
    isfinite(iso_t) || return nothing
    return Risk(event_time = iso_t,
        block_probability = 1.0 - iso.post_isolation_transmission)
end

function reset!(::Isolation, ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end

function initialise_individual!(iso::Isolation, individual, state)
    individual.state[:isolated] = false
    individual.state[:isolation_time] = Inf
    if is_eligible_for_isolation(iso.eligibility, individual, state)
        sens = _sample_value(iso.test_sensitivity, state.rng, individual)
        individual.state[:test_positive] = rand(state.rng) < sens
    else
        individual.state[:test_positive] = false
    end
    return nothing
end

function resolve_individual!(iso::Isolation, individual, state)
    is_isolated(individual) && return nothing

    # Three isolation pathways, each independent:
    #   - test_isolation_time:  onset + delay, fires iff test_positive
    #   - traced_isolation_time: set by ContactTracing's FlagOnly action
    #     (for symptomatic traced contacts), fires iff contact was traced
    # Isolation fires at the earlier of any active pathway. A
    # test-negative-but-traced contact is still isolated via tracing.
    traced_time = get(individual.state, :traced_isolation_time, Inf)::Float64
    test_time = if is_test_positive(individual)
        onset_time(individual) + rand(state.rng, iso.delay)
    else
        Inf
    end
    final = min(test_time, traced_time)
    isfinite(final) || return nothing
    set_isolated!(individual, final)
    return nothing
end
