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
# intervention's policy shape. `duration` (a scalar / distribution /
# function, drawn whenever isolation is set) is the same kind of parameter: it
# modifies the competing risk's release time, so the block it contributes can
# lapse rather than last forever.

"""
    IsolationEligibility

Rule for which cases can be isolated: [`SymptomaticOnly`](@ref) (the default)
or [`AllCases`](@ref). Whether an eligible case is actually isolated also
depends on the intervention's `test_sensitivity`.

To write a new rule, define a subtype and a method of
[`is_eligible_for_isolation`](@ref EpiBranch.is_eligible_for_isolation).
"""
abstract type IsolationEligibility end

"""
    is_eligible_for_isolation(eligibility, individual, state) -> Bool

Whether this case can be isolated under the eligibility rule (`true` by
default).
"""
is_eligible_for_isolation(::IsolationEligibility, individual, state) = true

"""
    records_isolation(eligibility, individual, state, isolation_time) -> Bool

Whether an isolation at `isolation_time` counts as a detection of the case.
The case stops transmitting from `isolation_time` either way; this decides
only whether [`is_isolated`](@ref) reports it, and so whether tracing, group
vaccination, line lists and detection counts see it. By default an isolation
at or after the case's own [`outcome_time`](@ref) (recovery or death) is not
counted, since isolating someone who has already recovered or died is not a
detection.

Define a method to count a late detection anyway. Post-mortem detection is the
usual reason, as with an Ebola death found at burial, which triggers tracing:

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

"""Only symptomatic cases can be isolated (the default for [`Isolation`](@ref))."""
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
    Isolation(; onset_to_isolation_delay, duration,
              eligibility = SymptomaticOnly(), test_sensitivity = 1.0,
              post_isolation_transmission = 0.0)

Isolate cases a delay after symptom onset, so they stop (or reduce) onward
transmission. Isolation applies to cases; traced contacts who are not yet
known cases are quarantined through [`ContactTracing`](@ref).

# Arguments
- `onset_to_isolation_delay`: days from symptom onset to isolation. A number,
  a distribution such as `Exponential(2.0)` (mean 2 days), or a function of
  the random number generator and the individual, `(rng, ind) -> ...`, drawn
  for each case. The function can read information another intervention has
  recorded on the case, for example to shorten the delay once a case in the
  same household has been detected.
- `duration` (required): days a case stays isolated before release.
  A number, a distribution or a function `(rng, ind) -> ...`. `Inf` isolates
  for good; with zero the case is still recorded as isolated, so tracing can
  start from it, but no transmission is prevented. There is no default
  because indefinite isolation is a modelling choice to make explicitly.
- `eligibility`: who can be isolated, [`SymptomaticOnly`](@ref) (default) or
  [`AllCases`](@ref) (for example with mass testing).
- `test_sensitivity`: probability that an eligible case is detected and
  isolated (default 1). A number, a distribution or a function
  `(rng, ind) -> ...`, drawn once per case.
- `post_isolation_transmission`: fraction of normal transmission that
  continues while isolated, between 0 and 1 (default 0, perfect isolation;
  0.25 would be leaky isolation).

After a finite `duration` the case transmits again at its normal
rate from the release time ([`isolation_release_time`](@ref)) until its
infectious period ends. A finite duration therefore gives the same result as
`Inf` only when the case stops being infectious before it is released.

An isolation at or after the case's own outcome (recovery, death or any other
terminal [`Transition`](@ref)) still stops transmission, so a route that
opens at the outcome, such as a funeral, is cut. It does not count as a
detection: [`is_isolated`](@ref) stays `false`, so `OnIsolation` tracing and
the line list do not count it.
[`records_isolation`](@ref EpiBranch.records_isolation) makes that choice and
can be changed through a custom eligibility rule.

Wrap it in [`OnRoutes`](@ref) to isolate a case out of only some of a model's
named routes, for example community contact while household contact
continues.

# Examples
```julia
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf)
model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
spec = ModelSpec(model; interventions = [iso], attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))
state = simulate(spec; max_cases = 500)

# Leaky isolation lasting 14 days, open to all cases
Isolation(
    onset_to_isolation_delay = 1.0, duration = 14.0,
    eligibility = AllCases(), post_isolation_transmission = 0.25,
)
```

For extension authors: the case records `:isolated`, `:isolation_time`,
`:isolation_release_time` and `:test_positive` in `ind.state`.
"""
struct Isolation{E <: IsolationEligibility, D, S, U} <: AbstractIntervention
    eligibility::E
    onset_to_isolation_delay::D
    test_sensitivity::S
    post_isolation_transmission::Float64
    duration::U
end

function Isolation(;
        onset_to_isolation_delay,
        duration,
        eligibility::IsolationEligibility = SymptomaticOnly(),
        test_sensitivity = 1.0,
        post_isolation_transmission::Real = 0.0
    )
    return Isolation(
        eligibility, onset_to_isolation_delay, test_sensitivity,
        Float64(post_isolation_transmission), duration
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

"""An isolated infector does not infect a contact while isolated, from its
isolation time until its [`isolation_release_time`](@ref) (never released when
`duration` is `Inf`). Each such contact is prevented with probability
`1 - post_isolation_transmission`."""
function competing_risk(iso::Isolation, parent, contact, state)
    key = iso.post_isolation_transmission == 0 ?
        REMOVAL_STRETCHES_KEY : LEAKY_REMOVAL_STRETCHES_KEY
    return _removal_risks(parent, 1.0 - iso.post_isolation_transmission, key)
end

# The likelihood reads the stretch a lapsing isolation removed the host for.
# Perfect isolation with a duration of `Inf` leaves no stretch to read, the
# window closing at the isolation's own start instead, and the layer records
# nothing extra for it. A recorded stretch is append-only, so the release it
# reports binds.
binding_release(::Isolation) = true

# Leaky isolation blocks only this fraction of transmission over its
# stretches (see `EpiBranch.removal_block_probability`); perfect isolation
# blocks all of it, the default. `household_offspring` (in `EpiHouseholds`)
# uses this fraction too, to weight a stretch's contribution to a case's
# community-infectious person-time.
removal_block_probability(iso::Isolation) = 1.0 - iso.post_isolation_transmission

function removal_gap_host_times(iso::Isolation)
    # Leaky isolation never closes the window (its block is per contact
    # regardless of duration), so it always has a stretch to read. That
    # stretch is recorded under its own key, keeping a likelihood from
    # mistaking it for a complete removal. Perfect isolation with a duration
    # of `Inf` closes the window instead and has no stretch of its own to add.
    iso.post_isolation_transmission == 0 || return (LEAKY_REMOVAL_STRETCHES_KEY,)
    iso.duration === Inf && return ()
    return (REMOVAL_STRETCHES_KEY,)
end

# Leaky isolation's residual block stands in for the removal perfect isolation
# makes, so it reaches the same routes: those the case is isolated from.
risk_applies(::Isolation, route) = route !== nothing && INTERVENTION_REMOVAL in route.until

function reset!(::Isolation, ind::Individual)
    # Only undo an isolation this Isolation set, named by the handle it kept
    # when it recorded one. Withdrawing exactly that entry leaves any
    # quarantine Isolation was layered over, or layered under, standing on
    # its own record untouched — resetting unconditionally would
    # un-quarantine a validly-traced contact when a Scheduled(Isolation) sees
    # its pre-start isolation time.
    handle = get(ind.state, :_isolation_handle, nothing)
    handle === nothing && return nothing
    withdraw_removal!(ind, handle)
    delete!(ind.state, :_isolation_handle)
    return nothing
end

function initialise_individual!(iso::Isolation, individual, state)
    clear_isolated!(individual)
    delete!(individual.state, :_isolation_handle)
    if is_eligible_for_isolation(iso.eligibility, individual, state)
        rng = stream(state.rng, individual.id, (:isolation, :test))
        sens = sample_value(iso.test_sensitivity, rng, individual)
        individual.state[:test_positive] = rand(rng) < sens
    else
        individual.state[:test_positive] = false
    end
    return nothing
end

function resolve_individual!(iso::Isolation, individual, state)
    # An isolation already standing on the individual is a quarantine written by
    # `ContactTracing`: a case's contacts are traced when the *infector* is
    # settled, which is before the contact itself resolves. Self-isolation from
    # onset applies on top of it regardless of which started first: `_isolate!`
    # below records its own entry either way, read back merged with the
    # quarantine's where they overlap and apart from it where they do not. That
    # way a quarantine that ends, or never covered anything, before onset still
    # gets the case isolated.
    if _isolation_in_force(individual)
        is_test_positive(individual) || return nothing
        self_t = onset_time(individual) +
            sample_value(
            iso.onset_to_isolation_delay,
            stream(state.rng, individual.id, (:isolation, :delay)), individual
        )
        isfinite(self_t) || return nothing
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
        max(_state_real(individual.state, :_traced_isolation_time, Inf), onset)
    test_time = if is_test_positive(individual)
        onset +
            sample_value(
            iso.onset_to_isolation_delay,
            stream(state.rng, individual.id, (:isolation, :delay)), individual
        )
    else
        Inf
    end
    final = min(test_time, traced_time)
    isfinite(final) || return nothing
    _isolate!(iso, individual, state, final)
    return nothing
end

# Remove the case from transmission at `time`, until `time + duration`, as
# its own entry, whether or not a removal already stands: a quarantine
# already in force keeps its own entry untouched, so `isolation_time` reads
# as the union of the two where they overlap or touch, and as this one alone
# where they do not — the later of the two, since a disjoint one earlier than
# `time` is already spent. The handle this entry gets is kept so a `reset!`
# can later withdraw exactly it, leaving any other entry be.
function _isolate!(iso::Isolation, individual, state, time)
    duration = _removal_duration(
        iso.duration, state.rng, individual, "`Isolation`'s `duration`"
    )
    release = time + duration
    # Leaky isolation records under its own key, apart from the shared one,
    # keeping a likelihood from mistaking this stretch for a complete
    # removal's.
    record_key = iso.post_isolation_transmission == 0 ?
        REMOVAL_STRETCHES_KEY : LEAKY_REMOVAL_STRETCHES_KEY
    unrecorded = !records_isolation(iso.eligibility, individual, state, time)
    handle = _set_isolated!(
        individual, time; release_time = release, key = record_key, unrecorded
    )
    individual.state[:_isolation_handle] = handle
    return nothing
end
