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
# intervention's policy shape. `isolation_duration` (a scalar / distribution /
# function, drawn whenever isolation is set) is the same kind of parameter: it
# modifies the competing risk's release time, so the block it contributes can
# lapse rather than last forever.

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

Whether an isolation at `isolation_time` is recorded as a detection, once a
pathway has reached one. The case is removed from transmission from
`isolation_time` either way; this decides only whether
[`is_isolated`](@ref) reports it, and so whether tracing, group vaccination,
line lists and detection counts see it. The default declines a time at or
after the case's own [`outcome_time`](@ref), since a self-report or trace
reaching a case that has already recovered or died describes a detection that
did not happen.

Override it to record a detection that arrives late anyway. Post-mortem
detection is the case that wants it, as with an Ebola death found at burial,
which triggers tracing:

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

`isolation_duration` is how long the removal lasts before it lapses; it
accepts a `Real`, a `Distribution`, or a function `(rng, ind) -> Real`
(drawn per individual, each time isolation is set). The default `Inf` keeps a
case isolated to the end of its infectious period, and a duration of zero
isolates nobody. A finite duration gives
[`isolation_release_time`](@ref) the time the block lapses.

What that changes depends on the engine. On a generation-based process the
block is a per-contact risk, so a contact after the release is not blocked.

On the continuous-time (Sellke) models a removal that is due to lapse does not
close the window: it stays open on the case's other removal states, if any,
and the per-contact competing risk blocks exactly the isolated interval, so
the case transmits again from the release time, matching the generation
engine. Only a removal standing to the end of the infectious period
(the default `Inf` duration) closes the window there, which is cheaper than
leaving it to the per-contact risk and is exact because nothing is left to
reopen.

An isolation time at or after the case's own outcome (recovery, death, or
any other terminal [`Transition`](@ref)) still removes the case from
transmission, so a route that opens at the outcome, such as a funeral, is cut
as before. It is not recorded as a detection: [`is_isolated`](@ref) stays
`false`, so `OnIsolation` tracing and the line list do not count it.
[`EpiBranch.records_isolation`](@ref) makes that choice and can be overridden
through the eligibility.

Initialises: `:isolated`, `:isolation_time`, `:isolation_release_time`,
`:test_positive`; sets `:_isolation_unrecorded` for an isolation it does not
record.
"""
struct Isolation{E <: IsolationEligibility, D, S, U} <: AbstractIntervention
    eligibility::E
    onset_to_isolation_delay::D
    test_sensitivity::S
    post_isolation_transmission::Float64
    isolation_duration::U
end

function Isolation(;
        onset_to_isolation_delay,
        eligibility::IsolationEligibility = SymptomaticOnly(),
        test_sensitivity = 1.0,
        post_isolation_transmission::Real = 0.0,
        isolation_duration = Inf
    )
    return Isolation(
        eligibility, onset_to_isolation_delay, test_sensitivity,
        Float64(post_isolation_transmission), isolation_duration
    )
end

required_fields(iso::Isolation) = _required_for_eligibility(iso.eligibility)
intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

# Perfect isolation removes a case from onward transmission at its isolation
# time, so a continuous-time model closes the infectious window there. Leaky
# isolation (`post_isolation_transmission > 0`) only reduces transmission, which
# the window cannot express, so it contributes no removal in that setting.
#
# A window cannot reopen once closed (see `_route_close`), so a removal that is
# due to lapse — whether before the case was even infected or partway through
# an already-open window — must not close it: closing it at the isolation time
# would take the case out of transmission for good, when the removal itself
# only takes it out until the release. The per-contact `competing_risk` below
# is release-aware throughout, on every transmission model, and blocks the
# isolated interval instead. The window closes here only for a removal with no
# release to leave it for.
function infectious_removal_time(iso::Isolation, ind::Individual)
    iso.post_isolation_transmission == 0 || return Inf
    return permanent_removal_time(ind)
end

"""Isolation blocks the parent → contact transmission while the parent's
isolation is in force: from its isolation time until its
[`isolation_release_time`](@ref) (`Inf` by default, so the block lasts until
the end of the infectious period). Residual transmission is governed by
`post_isolation_transmission`: `block_probability = 1 - post_isolation_transmission`."""
function competing_risk(iso::Isolation, parent, contact, state)
    return _removal_risks(parent, 1.0 - iso.post_isolation_transmission)
end

# A finite duration leaves the window open and blocks each contact against the
# infector's own isolated stretch. The block then depends on the infector
# whatever the residual is.
function risk_depends_on_infector(iso::Isolation)
    return iso.post_isolation_transmission > 0 || !(iso.isolation_duration === Inf)
end

# The likelihood reads the stretch a lapsing isolation removed the host for.
# A duration of `Inf` leaves no stretch to read, the window closing at the
# isolation's own start, and the layer records nothing extra for it.
function removal_gap_host_times(iso::Isolation)
    iso.isolation_duration === Inf && return ()
    return (REMOVAL_STRETCHES_KEY,)
end

# Leaky isolation's residual block stands in for the removal perfect isolation
# makes, so it reaches the same routes: those the case is isolated from.
risk_applies(::Isolation, route) = route !== nothing && INTERVENTION_REMOVAL in route.until

# Isolation itself reads and writes only the case it resolves, but its
# eligibility receives the whole state, and it answers for that too. The
# built-in eligibilities read only the individual; one written outside the
# package declares its own read.
reads_population_state(iso::Isolation) = reads_population_state(iso.eligibility)
reads_population_state(::IsolationEligibility) = false

function reset!(::Isolation, ind::Individual)
    # Only undo an isolation this Isolation set. `:isolated`/`:isolation_time`
    # are shared keys — ContactTracing's Quarantine writes them directly too —
    # so resetting unconditionally would un-quarantine a validly-traced contact
    # when a Scheduled(Isolation) sees its pre-start isolation time.
    get(ind.state, :_isolated_by_isolation, false) || return nothing
    # If this isolation was layered over a quarantine, restore that quarantine
    # rather than clearing the individual outright — undoing Isolation's own
    # effect must not also undo another intervention's.
    previous = get(ind.state, :_isolation_time_before_isolation, Inf)
    if isfinite(previous)
        previous_release = get(ind.state, :_isolation_release_time_before_isolation, Inf)
        # The recorded stretches are cleared and the quarantine's own re-recorded,
        # so this isolation's stretch is not left for a likelihood to take out of
        # an exposure the simulation never blocked. A quarantine keeps its own
        # record under its own key, which this does not touch.
        clear_isolated!(ind)
        set_isolated!(ind, previous; release_time = previous_release)
        get(ind.state, :_isolation_unrecorded_before_isolation, false) &&
            (ind.state[:_isolation_unrecorded] = true)
        delete!(ind.state, :_isolation_time_before_isolation)
        delete!(ind.state, :_isolation_release_time_before_isolation)
        delete!(ind.state, :_isolation_unrecorded_before_isolation)
    else
        clear_isolated!(ind)
    end
    ind.state[:_isolated_by_isolation] = false
    return nothing
end

function initialise_individual!(iso::Isolation, individual, state)
    clear_isolated!(individual)
    individual.state[:_isolated_by_isolation] = false
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
    if _isolation_in_force(individual)
        is_test_positive(individual) || return nothing
        self_t = onset_time(individual) +
            _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
        self_t < isolation_time(individual) || return nothing
        # Remember what we are overwriting. Claiming provenance below tells a
        # `Scheduled` reset that this isolation is Isolation's to undo, but the
        # standing quarantine underneath it belongs to ContactTracing and must
        # survive that reset, so stash it, and whether it was recorded, for
        # `reset!` to restore.
        individual.state[:_isolation_time_before_isolation] = isolation_time(individual)
        individual.state[:_isolation_release_time_before_isolation] =
            isolation_release_time(individual)
        if _isolation_unrecorded(individual)
            individual.state[:_isolation_unrecorded_before_isolation] = true
        else
            delete!(individual.state, :_isolation_unrecorded_before_isolation)
        end
        _isolate!(iso, individual, state, self_t)
        return nothing
    end

    # Three isolation pathways, each independent:
    #   - test_isolation_time:  onset + delay, occurs iff test_positive
    #   - _traced_isolation_time: set by ContactTracing's FlagOnly action
    #     for traced contacts, occurs iff contact was traced and has an onset
    # Isolation occurs at the earlier of any active pathway. A
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
        max(get(individual.state, :_traced_isolation_time, Inf), onset)
    test_time = if is_test_positive(individual)
        onset + _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
    else
        Inf
    end
    final = min(test_time, traced_time)
    isfinite(final) || return nothing
    _isolate!(iso, individual, state, final)
    return nothing
end

# Remove the case from transmission at `time`, until `time + isolation_duration`,
# recording it as a detection only if the eligibility does. The provenance mark
# lets a Scheduled reset undo only Isolation's own effect.
function _isolate!(iso::Isolation, individual, state, time)
    duration = _removal_duration(
        iso.isolation_duration, state.rng, individual, "`isolation_duration`"
    )
    start, release = time, time + duration
    # A removal already standing is layered under this one by the same rule the
    # trace path uses, so neither side's release is lost.
    was_unrecorded = _isolation_unrecorded(individual)
    if _isolation_in_force(individual)
        start, release = _combine_removal(
            isolation_time(individual), isolation_release_time(individual),
            start, release
        )
    end
    set_isolated!(individual, start; release_time = release)
    individual.state[:_isolated_by_isolation] = true
    # Whether this counts as a detection is a question about the start in
    # force, so a standing start that won keeps its own answer, and only an
    # isolation starting here is put to this eligibility.
    unrecorded = start == time ?
        !records_isolation(iso.eligibility, individual, state, start) : was_unrecorded
    unrecorded && (individual.state[:_isolation_unrecorded] = true)
    return nothing
end
