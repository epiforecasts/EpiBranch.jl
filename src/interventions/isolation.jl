# ── Trait protocol for isolation ────────────────────────────────────
#
# Isolation's three points of variation, each a dispatched seam:
#
# - `IsolationEligibility`: who can be isolated at all
#   (default: symptomatic cases only).
# - `test_sensitivity`: probability an eligible individual tests
#   positive and so reaches isolation (a scalar / distribution /
#   function, sampled per individual at init time).
# - `onset_to_isolation_delay`: time from onset to self-reported
#   isolation (a distribution drawn per individual).
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

"""
    records_isolation(eligibility, individual, state, isolation_time) -> Bool

Whether a case is recorded as isolated at `isolation_time`, once a pathway has
produced one. The default declines a time at or after the case's own outcome:
isolating a case that has already recovered or died has no effect on
transmission, and recording it would report a detection that did not happen to
tracing, line lists and containment counts.

Override it to record a detection that arrives late anyway. Post-mortem
detection is the case that wants it, as with an Ebola death found at burial,
which triggers tracing and safe burial although it changes no onward
transmission:

```julia
struct DetectAfterOutcome <: EpiBranch.IsolationEligibility end
EpiBranch.is_eligible_for_isolation(::DetectAfterOutcome, ind, state) =
    !is_asymptomatic(ind)
EpiBranch.records_isolation(::DetectAfterOutcome, ind, state, t) = true
```
"""
function records_isolation(::IsolationEligibility, individual, state, isolation_time)
    return isolation_time < outcome_time(individual)
end

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

`onset_to_isolation_delay` is the time from symptom onset to
self-reported isolation; it accepts a `Real`, a `Distribution`, or a
function `(rng, ind) -> Real` (drawn per individual, per resolution). The
function form can read state recorded on the individual by another
intervention earlier in the stack — for example a group's own event time,
switching the delay once a household's first case has been detected.

`post_isolation_transmission` ∈ [0, 1] sets the residual transmission
probability after isolation. The competing risk's `block_probability`
is `1 - post_isolation_transmission`.

A case whose only isolation pathways (self-report or tracing) would fire at
or after its own outcome (recovery, death, or any other terminal
[`Transition`](@ref)) is left unisolated: isolation after the infectious
period has already ended has no effect on transmission, and recording it
would misreport detection to `OnIsolation`, line lists and containment
counts.

Initialises: `:isolated`, `:isolation_time`, `:test_positive`.
"""
struct Isolation{E <: IsolationEligibility, D, S} <: AbstractIntervention
    eligibility::E
    onset_to_isolation_delay::D
    test_sensitivity::S
    post_isolation_transmission::Float64
end

function Isolation(;
        onset_to_isolation_delay,
        eligibility::IsolationEligibility = SymptomaticOnly(),
        test_sensitivity = 1.0,
        post_isolation_transmission::Real = 0.0
    )
    return Isolation(
        eligibility, onset_to_isolation_delay, test_sensitivity,
        Float64(post_isolation_transmission)
    )
end

required_fields(iso::Isolation) = _required_for_eligibility(iso.eligibility)
intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

# Perfect isolation removes a case from onward transmission at its isolation
# time, so a continuous-time model closes the infectious window there. Leaky
# isolation (`post_isolation_transmission > 0`) only reduces transmission, which
# the window cannot express, so it contributes no removal in that setting.
function infectious_removal_time(iso::Isolation, ind::Individual)
    return iso.post_isolation_transmission == 0 ? isolation_time(ind) : Inf
end

"""Isolation blocks the parent → contact transmission when the parent's
isolation time is earlier than the contact's transmission time.
Residual transmission is governed by `post_isolation_transmission`:
`block_probability = 1 - post_isolation_transmission`."""
function competing_risk(iso::Isolation, parent, contact, state)
    iso_t = isolation_time(parent)
    isfinite(iso_t) || return nothing
    return Risk(
        event_time = iso_t,
        block_probability = 1.0 - iso.post_isolation_transmission
    )
end

# Leaky isolation's residual block stands in for the removal perfect isolation
# makes, so it reaches the same routes: those the case is isolated from.
risk_applies(::Isolation, route) = route !== nothing && INTERVENTION_REMOVAL in route.until

function reset!(::Isolation, ind::Individual)
    # Only undo an isolation this Isolation set. `:isolated`/`:isolation_time`
    # are shared keys — ContactTracing's Quarantine writes them directly too —
    # so resetting unconditionally would un-quarantine a validly-traced contact
    # when a Scheduled(Isolation) sees its pre-start isolation time.
    get(ind.state, :isolated_by_isolation, false) || return nothing
    # If this isolation was layered over a quarantine, restore that quarantine
    # rather than clearing the individual outright — undoing Isolation's own
    # effect must not also undo another intervention's.
    previous = get(ind.state, :isolation_time_before_isolation, Inf)
    if isfinite(previous)
        set_isolated!(ind, previous)
        delete!(ind.state, :isolation_time_before_isolation)
    else
        clear_isolated!(ind)
    end
    ind.state[:isolated_by_isolation] = false
    return nothing
end

function initialise_individual!(iso::Isolation, individual, state)
    clear_isolated!(individual)
    individual.state[:isolated_by_isolation] = false
    if is_eligible_for_isolation(iso.eligibility, individual, state)
        sens = _sample_value(iso.test_sensitivity, state.rng, individual)
        individual.state[:test_positive] = rand(state.rng) < sens
    else
        individual.state[:test_positive] = false
    end
    return nothing
end

function resolve_individual!(iso::Isolation, individual, state)
    # An isolation already standing on the individual is a quarantine written by
    # `ContactTracing`. On the generation-based path that cannot happen here,
    # because isolation resolves before tracing runs; on the continuous-time
    # path it routinely does, because a case's contacts are traced when the
    # *infector* is finalised, which is before the contact itself resolves. The
    # quarantine is then a competing pathway rather than a reason to stop:
    # without this the contact would keep a trace time later than the onset it
    # would have self-reported on, and tracing would delay isolation instead of
    # advancing it.
    if is_isolated(individual)
        is_test_positive(individual) || return nothing
        self_t = onset_time(individual) +
            _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
        # Whether a self-report reaching past the case's own outcome counts as
        # a detection is the eligibility's call (`records_isolation`); the
        # default declines it.
        self_t < isolation_time(individual) || return nothing
        records_isolation(iso.eligibility, individual, state, self_t) ||
            return nothing
        # Remember what we are overwriting. Claiming provenance below tells a
        # `Scheduled` reset that this isolation is Isolation's to undo, but the
        # standing quarantine underneath it belongs to ContactTracing and must
        # survive that reset, so stash it for `reset!` to restore.
        individual.state[:isolation_time_before_isolation] = isolation_time(individual)
        set_isolated!(individual, self_t)
        individual.state[:isolated_by_isolation] = true
        return nothing
    end

    # Three isolation pathways, each independent:
    #   - test_isolation_time:  onset + delay, fires iff test_positive
    #   - traced_isolation_time: set by ContactTracing's FlagOnly action
    #     for traced contacts, fires iff contact was traced and has an onset
    # Isolation fires at the earlier of any active pathway. A
    # test-negative-but-traced contact is still isolated via tracing.
    #
    # The traced pathway isolates a flagged contact once it has symptoms, so a
    # contact with no onset never isolates through it. That includes a contact
    # whose infection was aborted before onset, after the trace had already
    # recorded its expected onset. A continuous-time model traces a contact
    # before its own infection is settled, when the onset is not yet known, so
    # the recorded time is held back to the onset.
    onset = onset_time(individual)
    traced_time = isnan(onset) ? Inf :
        max(get(individual.state, :traced_isolation_time, Inf), onset)
    test_time = if is_test_positive(individual)
        onset + _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
    else
        Inf
    end
    final = min(test_time, traced_time)
    # Whether an isolation reaching past the case's own outcome counts as a
    # detection is the eligibility's call (`records_isolation`); the default
    # declines it, since by then the case has left the infectious period.
    records_isolation(iso.eligibility, individual, state, final) || return nothing
    set_isolated!(individual, final)
    # Mark provenance so a Scheduled reset undoes only Isolation's own effect.
    individual.state[:isolated_by_isolation] = true
    return nothing
end
