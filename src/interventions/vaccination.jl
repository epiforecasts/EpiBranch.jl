"""
Base type for vaccination interventions. A vaccination has an
`efficacy` (per-exposure block probability once immunity is in place)
and a `delay_to_immunity` (time between vaccination and protection).

Concrete subtypes differ only in eligibility — who gets vaccinated
when. They share the [`competing_risk`](@ref) machinery: a vaccinated
contact whose immunity has developed by their transmission time has
their infection blocked with probability `efficacy`.

`efficacy` accepts a `Real`, a `Distribution`, or a function
`(rng, ind) -> Real`. The function/distribution forms sample once per
vaccinated individual at vaccination time and store the result on the
contact; the competing risk reads the stored value.

`mode` is an [`AbstractEffectMode`](@ref): [`LeakyMode`](@ref) (the
default) reduces each exposure's success probability by `efficacy`,
while [`AllOrNothingMode`](@ref) fully protects a fraction `efficacy`
of vaccinated individuals and leaves the rest unaffected.

!!! note "In a pure branching process the two modes are equivalent"
    Every contact in a branching process is a unique exposure, so
    per-exposure and per-individual semantics give the **same**
    per-contact infection probability. Switching between `LeakyMode`
    and `AllOrNothingMode` here will not change simulation results.
    The distinction only starts to matter once network models permit
    multiple exposures per individual (e.g. the planned
    `EpiBranchHouseholds`); the two modes are exposed now so that
    code written for the household model has the right vocabulary.

# Multi-dose vaccination

A `dose_label::Symbol` (default `:default`) namespaces the per-contact
state so multiple vaccinations can stack without colliding. Two
`MassVaccination`s with `dose_label = :prime` and `dose_label = :boost`
write to `:vaccinated_prime` / `:vaccinated_boost` and contribute
independent competing risks. Each dose's protection is composed by
the engine via the standard competing-risks product.

When `dose_label = :default` (single-dose, the common case), state is
written to plain `:vaccinated` / `:vaccination_time` for backwards
compatibility with the `is_vaccinated` accessor.
"""
abstract type AbstractVaccination <: AbstractIntervention end

"""
Effect mode for a vaccination: how `efficacy` translates into
per-exposure infection probability.

Concrete subtypes:

- [`LeakyMode`](@ref): every exposure of a vaccinated individual is
  reduced by `efficacy` (per-exposure semantics).
- [`AllOrNothingMode`](@ref): a fraction `efficacy` of vaccinated
  individuals are fully protected for all exposures; the rest gain
  no protection (per-individual semantics).
"""
abstract type AbstractEffectMode end

"""Per-exposure efficacy: each exposure's transmission is blocked
independently with probability `efficacy`. Default mode."""
struct LeakyMode <: AbstractEffectMode end

"""Per-individual efficacy: a fraction `efficacy` of vaccinated
individuals are fully protected (susceptibility = 0); the rest gain
no protection."""
struct AllOrNothingMode <: AbstractEffectMode end

"""Time between vaccination and the onset of protective immunity. Added
to the vaccination time to give the event time of the competing risk."""
delay_to_immunity(v::AbstractVaccination) = v.delay_to_immunity

"""Label of the dose a contact must already have received before this
vaccination is given, or `nothing` when it requires no earlier dose."""
required_dose(::AbstractVaccination) = nothing

"""Label namespacing the vaccination's per-contact state (`:vaccinated`,
`:vaccination_time`, `:vaccine_efficacy`). `:default` writes to the
unsuffixed keys for backwards compatibility; other labels write to
`:vaccinated_<label>` etc."""
dose_label(v::AbstractVaccination) = v.dose_label

function _vaccinated_key(label::Symbol)
    label === :default ? :vaccinated : Symbol("vaccinated_", label)
end
function _vaccination_time_key(label::Symbol)
    label === :default ? :vaccination_time : Symbol("vaccination_time_", label)
end
function _vaccine_efficacy_key(label::Symbol)
    label === :default ? :vaccine_efficacy : Symbol("vaccine_efficacy_", label)
end

function initialise_individual!(v::AbstractVaccination, individual, state)
    label = dose_label(v)
    individual.state[_vaccinated_key(label)] = false
    individual.state[_vaccination_time_key(label)] = Inf
    return nothing
end

"""Susceptibility-side risk: blocks the parent → contact transmission
iff this dose has been administered to the contact and the contact's
vaccine-induced immunity has developed by their transmission time.

A dose must *precede* the exposure it blocks. Under default tracing a
contact is traced once its infector has been isolated, and that isolation
already blocks every later exposure. For a ring dose this risk therefore
applies only where a contact can still be infected after its trace: under
leaky isolation, when tracing starts at the infector's symptom onset,
or in a `depth > 1` ring passing through members who keep transmitting
after they are traced. For a dose acting on an infection the contact
already has, see `post_exposure_efficacy` and `onward_efficacy` on
[`RingVaccination`](@ref)."""
function _susceptibility_risk(v::AbstractVaccination, contact)
    label = dose_label(v)
    get(contact.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(contact.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    eff = get(contact.state, _vaccine_efficacy_key(label), nothing)
    eff === nothing && return nothing
    # A zero-efficacy dose can never block, and the engine skips such a risk.
    # Returning nothing keeps it out of the returned tuple, so the recommended
    # post-exposure-only setup (`efficacy = 0.0`) does not build and discard a
    # risk for every contact.
    eff <= 0 && return nothing
    return Risk(event_time = vacc_t + delay_to_immunity(v), block_probability = eff)
end

function competing_risk(v::AbstractVaccination, parent, contact, state)
    _susceptibility_risk(v, contact)
end

# Helper for concrete subtypes: write per-dose state on a contact at
# vaccination time. Samples efficacy via `_sample_value` so scalar,
# distribution, and function forms all work.
function _record_vaccination!(v::AbstractVaccination, contact, vacc_t, rng)
    label = dose_label(v)
    contact.state[_vaccinated_key(label)] = true
    contact.state[_vaccination_time_key(label)] = vacc_t
    contact.state[_vaccine_efficacy_key(label)] = _sample_value(v.efficacy, rng, contact)
    return nothing
end

# ── RingVaccination ──────────────────────────────────────────────────

"""
Vaccinate traced contacts. Applied to contacts that have been traced
(`:traced == true`, set by [`ContactTracing`](@ref)).

For post-exposure prophylaxis (PEP, cf.
[pepbp](https://github.com/sophiemeakin/pepbp)), set
`delay_to_immunity = 0.0` (the default); `post_exposure_efficacy` and
`onward_efficacy` (below) set what it does to an infection the contact
already has. For ring vaccination with a vaccine that takes
time to confer protection, set `delay_to_immunity` to the appropriate
delay.

`coverage` is the per-contact probability that a traced contact
actually receives the vaccine, capturing programme reach (consent
refusal, absence, exclusion criteria, logistical gaps). Defaults to
`1.0`. Accepts a `Real`, `Distribution`, or `Function`
`(rng, contact) -> Real` for per-individual coverage (e.g.
age-dependent). Refusal clusters by household or community in practice
rather than falling independently on each contact; give `coverage` a
function reading a value set by [`vaccine_acceptance`](@ref) to draw
that correlation in, so ring members share one acceptance *probability*
rather than each drawing an independent one — each contact still flips
its own coin against it, so the covered fraction varies ring to ring
rather than averaging out.

`eligibility_window` skips vaccination when the time since the
contact's exposure exceeds the window — typical of filovirus-type
protocols where post-exposure vaccination beyond ~21 days is
operationally pointless. Defaults to `Inf` (no window). Accepts a
`Real` or `Function` `(rng, contact) -> Real`. Eligibility is checked
at vaccination time, not at immunity-onset time, so a contact
vaccinated near the window's end with a long `delay_to_immunity` is
still recorded as vaccinated (whether immunity arrives before that
contact's own transmission time is then decided by competing risks).

!!! warning "The window is measured to *this* dose, `dose_delay` included"
    A second dose is administered `dose_delay` days after the trace, and
    the window is checked against that time. Copying a prime's
    `eligibility_window` onto a boost therefore rejects almost every
    boost, since `dose_delay` alone usually exceeds the window. A window
    belongs on the dose whose timing it describes, so leave a later dose
    at the default `Inf` unless the protocol has a deadline on the second
    dose itself.

`post_exposure_efficacy` is the probability that a dose aborts an
infection the contact already has. It can do so when immunity
(vaccination + `delay_to_immunity`) arrives after the exposure but
before the contact's symptom onset. An aborted infection runs until
immunity arrives and ends there: the contact transmits as usual before
then and not at all afterwards. It has no symptom onset, so nothing
triggered by onset happens (isolation, tracing, onset-timed transitions).
Its clinical course stops at the abort: no transition in the
`progression` takes effect at or after that time, whatever it is timed
from, so there is no later hospitalisation, death or other outcome. A
transition that took effect before immunity arrived stands. An aborted
infection is still a case, counted in [`chain_statistics`](@ref) and
listed by [`linelist`](@ref) with no onset date and a
`date_infection_aborted` column (from `:infection_aborted_time`). Where
immunity is already in place at the exposure, the dose instead blocks the
infection with the same probability, which is all it can do for a contact
with no onset to race (asymptomatic, `NaN` incubation period). Unlike
`efficacy`, `post_exposure_efficacy` acts when a contact is traced after
its infector isolates, provided the trace does not quarantine it (see the
warning below). Defaults to `0.0`. Requires `:incubation_period`, set by
[`clinical_presentation`](@ref).

`post_exposure_efficacy` already blocks every exposure that `efficacy`
would. Set one or the other: setting both composes the two blocks
independently, which double-counts.

`onward_efficacy` is the per-exposure probability that a *vaccinated
parent's* onward transmission is blocked once the parent's
vaccine-induced immunity has developed, so it also averts onward cases
from contacts already exposed when vaccinated. Defaults to `0.0` (no
onward effect). Unlike `post_exposure_efficacy`, it leaves the infection
and its disease in place: each transmission after immunity is blocked
with this probability whenever the contact's onset falls.
`post_exposure_efficacy` ends the whole infection, and only for contacts
whose immunity arrives before their onset. The two compose independently,
giving a vaccine that aborts some infections and reduces the infectiousness
of the rest. `efficacy` (the susceptibility-side block applied when the
*contact* is vaccinated) also applies independently; setting it and
`onward_efficacy` to the same value gives a vaccine that acts
symmetrically on susceptibility and infectiousness.

Requires `:traced` (set by [`ContactTracing`](@ref)).

!!! warning "`efficacy` is redundant when contacts are traced after their infector isolates"
    `ContactTracing` by default traces a contact once its infector has been
    isolated, and a non-leaky `Isolation` then already blocks every later
    transmission to the contact. A dose acting only through `efficacy`
    leaves the results unchanged, with or without quarantine
    (`quarantine_on_trace = false`). `efficacy` has infections left to
    prevent only when a contact can still be infected after being traced:
    under leaky isolation (`post_isolation_transmission > 0`), when tracing
    starts before the infector is isolated (for example
    `eligibility = OnSymptomOnset()`), or in a `depth > 1` ring passing
    through members who keep transmitting after they are traced.
    `onward_efficacy` acts on the traced contact's own later transmission,
    which a quarantine already blocks, so it acts under tracing without
    quarantine. So does `post_exposure_efficacy`, which acts on the
    infection the contact already has. Under quarantine
    `post_exposure_efficacy` can lower containment, because an aborted case
    never has symptoms and so no longer triggers tracing of the contacts it
    infected before its dose.

Per-contact state keys are `:vaccinated`, `:vaccination_time`, and
`:vaccine_efficacy` for the default dose label. With a non-default
`dose_label`, the keys carry the label as a suffix.

# Second and later doses

A dose is given at the trace, or `dose_delay` days after it.
`requires_dose` names a dose label the contact must have received by the
time this dose falls due; a contact without it does not get this dose. A
prime-boost schedule is two `RingVaccination`s:

```julia
[
    RingVaccination(efficacy = 0.6, delay_to_immunity = 21.0,
        coverage = 0.8, dose_label = :prime),
    RingVaccination(efficacy = 0.5, dose_delay = 28.0,
        delay_to_immunity = 14.0, coverage = 0.9,
        requires_dose = :prime, dose_label = :boost),
]
```

The boost is given 28 days after the trace to 90% of those primed (the
remaining 10% being lost to follow-up), and protects 14 days later. List
a dose after the dose it requires: the stack is applied in order, so a
boost placed first would see no prime and never be given. The required
dose may come from any vaccination, such as a [`MassVaccination`](@ref)
prime; a contact whose prime falls after the boost's due date is not
boosted. Between two ring doses, a `dose_delay` shorter than the required
dose's is rejected when the `ModelSpec` is built, since such a boost could
never be given.

Doses compose as competing risks, so a schedule reaching 80% protection
in total from a prime at 60% needs `efficacy = 0.5` on the boost
(`(0.8 - 0.6) / (1 - 0.6)`), the protection the second dose adds among
those the first left unprotected.

A dose is recorded when it falls due, whether or not the contact was
infected in the meantime, because infection is resolved after the doses
are given. Dose counts for later doses are therefore counts of doses
scheduled.
"""
Base.@kwdef struct RingVaccination{E, C, W, M <: AbstractEffectMode} <: AbstractVaccination
    efficacy::E
    coverage::C = 1.0
    delay_to_immunity::Float64 = 0.0
    dose_delay::Float64 = 0.0
    requires_dose::Union{Nothing, Symbol} = nothing
    eligibility_window::W = Inf
    post_exposure_efficacy::Float64 = 0.0
    onward_efficacy::Float64 = 0.0
    mode::M = LeakyMode()
    dose_label::Symbol = :default
end

function required_fields(rv::RingVaccination)
    rv.post_exposure_efficacy > 0.0 ? [:traced, :incubation_period] : [:traced]
end
required_dose(rv::RingVaccination) = rv.requires_dose

# Onward-infectiousness risk: blocks the parent → contact transmission
# iff this dose has been administered to the *parent* and the parent's
# immunity has developed by their (the parent's) transmission time. The
# parent's `:vaccination_time` is set by ring vaccination when the parent
# was traced, and the onward immunity takes effect at that time plus
# `delay_to_immunity`, as on the susceptibility side.
function _onward_risk(rv::RingVaccination, parent)
    rv.onward_efficacy > 0.0 || return nothing
    label = dose_label(rv)
    get(parent.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(parent.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    return Risk(event_time = vacc_t + delay_to_immunity(rv),
        block_probability = rv.onward_efficacy)
end

# Contact-side risk. `efficacy` blocks an exposure that comes after immunity,
# and so does `post_exposure_efficacy`: a dose whose immunity is already in
# place when the contact is exposed prevents the infection outright, the limit
# of aborting it the moment it starts. This is also all a dose can do for a
# contact with no onset to race (asymptomatic). Both act at the same immunity
# time, so they combine into one block, drawn once.
function _contact_risk(rv::RingVaccination, contact)
    susceptibility = _susceptibility_risk(rv, contact)
    post = rv.post_exposure_efficacy
    post > 0.0 || return susceptibility
    if susceptibility === nothing
        label = dose_label(rv)
        get(contact.state, _vaccinated_key(label), false) || return nothing
        vacc_t = get(contact.state, _vaccination_time_key(label), Inf)
        isfinite(vacc_t) || return nothing
        return Risk(event_time = vacc_t + delay_to_immunity(rv), block_probability = post)
    end
    return Risk(event_time = susceptibility.event_time,
        block_probability = 1 - (1 - susceptibility.block_probability) * (1 - post))
end

# A dose given after the exposure can still abort the infection, so long as
# immunity arrives before symptom onset. The contact then stays infected up to
# its immunity time and transmits as usual until then; after it, it transmits
# nothing (the engine's `AbortedInfection` risk source). It has no onset, and
# `resolve_transitions!` ends its clinical course at the abort time.
#
# The draw happens in the intervention phase, when the dose is recorded and
# again each time a contact already given the dose is exposed. It cannot happen
# later, because a contact's onset and clinical course are resolved together
# with its infection. The draw is made only when immunity falls between the
# exposure and onset, so a dose that cannot abort anything leaves the random
# stream untouched. The exposure is the contact's provisional infection time,
# its earliest exposure when several infectors reach it. If that exposure turns
# out not to be the infection, the engine removes the abort
# (`_drop_stale_abort!`), and a contact that escaped it gets a fresh draw at its
# next exposure.
#
# Callers check `post_exposure_efficacy > 0` first: the vaccination time comes
# untyped from the contact's state, so an unconditional call would dispatch
# dynamically for every vaccinated contact of a dose that cannot abort.
function _abort_infection!(rv::RingVaccination, contact, vacc_t, rng)
    incubation = get(contact.state, :incubation_period, NaN)
    isnan(incubation) && return nothing
    immunity = vacc_t + delay_to_immunity(rv)
    exposure = contact.infection_time
    exposure < immunity < exposure + incubation || return nothing
    _covers(rv.post_exposure_efficacy, contact, rng) || return nothing
    # An earlier dose may already have aborted it; the infection ends at the
    # first abort.
    contact.state[:infection_aborted_time] = min(
        get(contact.state, :infection_aborted_time, Inf), immunity)
    _set_onset_from_incubation!(contact)
    return nothing
end

# Combine the contact-side risk with the optional onward-infectiousness risk
# (acting on the parent). The engine's `_iter_risks` helper accepts a tuple of
# risks. The end of an aborted infection is not among them: the engine's
# `AbortedInfection` risk source applies it for as long as the abort is
# recorded, so it stays in force when a `Scheduled` wrapper switches this
# intervention off.
function competing_risk(rv::RingVaccination, parent, contact, state)
    exposure = _contact_risk(rv, contact)
    onward = _onward_risk(rv, parent)
    exposure === nothing && return onward
    onward === nothing && return exposure
    return (exposure, onward)
end

# Scalar defaults short-circuit without drawing from the rng so that
# coverage = 1.0 and eligibility_window = Inf reproduce the previous
# deterministic behaviour exactly.
_within_eligibility_window(w::Real, ind, vacc_t, rng) = vacc_t - ind.infection_time <= w
function _within_eligibility_window(w, ind, vacc_t, rng)
    vacc_t - ind.infection_time <= _sample_value(w, rng, ind)
end

_covers(p::Real, ind, rng) = p >= 1.0 || rand(rng) < p
_covers(p, ind, rng) = rand(rng) < _sample_value(p, rng, ind)

# A dose that requires an earlier one is given only to contacts who
# received it by the time this dose falls due, so `coverage` on the later
# dose is the retention among those who had the earlier one. The check
# reads the required dose's stored time, because a vaccination such as
# `MassVaccination` records a dose as soon as it draws a time, and that
# time can fall after the later dose.
function _has_required_dose(v::AbstractVaccination, ind, vacc_t)
    _has_required_dose(required_dose(v), ind, vacc_t)
end
_has_required_dose(::Nothing, ind, vacc_t) = true
function _has_required_dose(label::Symbol, ind, vacc_t)
    required_t = get(ind.state, _vaccination_time_key(label), Inf)
    return isfinite(required_t) && required_t <= vacc_t
end

"""
    _validate_dose_schedule(interventions)

Check that every vaccination requiring an earlier dose is listed after the
dose it requires, and, when both are ring doses, is not scheduled to arrive
before it. The stack is applied in order, so a dose placed before the one it
requires would silently never be given. Two ring doses are timed from the
same trace, so a shorter `dose_delay` on the later dose would likewise mean
it is never given. Other schedules are checked per contact when the dose
falls due.
"""
function _validate_dose_schedule(interventions)
    given = Dict{Symbol, Union{Nothing, Float64}}()
    for iv in interventions
        vacc = _unwrap_scheduled(iv)
        vacc isa AbstractVaccination || continue
        label = dose_label(vacc)
        req = required_dose(vacc)
        offset = _dose_offset(vacc)
        if req !== nothing
            haskey(given, req) || throw(ArgumentError(
                "vaccination with dose_label = :$label requires dose :$req, " *
                "which is not given earlier in the intervention stack. " *
                "List a dose after the dose it requires."))
            req_offset = given[req]
            if offset !== nothing && req_offset !== nothing && offset < req_offset
                throw(ArgumentError(
                    "vaccination with dose_label = :$label requires dose :$req " *
                    "but is scheduled earlier than it ($offset days after the " *
                    "trace, compared with $req_offset). A dose cannot be given " *
                    "before the dose it requires."))
            end
        end
        given[label] = offset
        _warn_double_counted_efficacy(vacc)
    end
    return nothing
end

# Immunity before onset is a weaker condition than immunity before exposure, so
# a dose setting both fields blocks with `1 - (1 - e1)(1 - e2)` for any contact
# vaccinated before its exposure. That is common once isolation is leaky,
# because a contact's trace time comes from its infector's course and not from
# its own. Only the scalar forms can be checked; a function or distribution
# efficacy is left alone.
_warn_double_counted_efficacy(::AbstractVaccination) = nothing
function _warn_double_counted_efficacy(rv::RingVaccination)
    rv.post_exposure_efficacy > 0.0 || return nothing
    rv.efficacy isa Real && rv.efficacy > 0.0 || return nothing
    @warn "RingVaccination sets both `efficacy` and `post_exposure_efficacy`, "*
          "which compose as independent risks and so over-protect any contact "*
          "vaccinated before its exposure. `post_exposure_efficacy` already "*
          "covers those contacts; set one or the other." dose_label=dose_label(rv) maxlog=1
    return nothing
end

"""Days from the trace to this dose, or `nothing` for a vaccination not timed
from the trace. [`RingVaccination`](@ref) is the only one timed from it."""
_dose_offset(::AbstractVaccination) = nothing
_dose_offset(rv::RingVaccination) = rv.dose_delay

# The intervention inside a wrapper; `Scheduled` adds a method.
_unwrap_scheduled(iv) = iv

function apply_post_transmission!(rv::RingVaccination, state, new_contacts)
    label = dose_label(rv)
    vacc_key = _vaccinated_key(label)
    for ind in new_contacts
        is_traced(ind) || continue
        if get(ind.state, vacc_key, false)
            # Dosed in an earlier generation and exposed again: the dose stays,
            # and the abort draw is made against this exposure.
            rv.post_exposure_efficacy > 0.0 &&
                _abort_infection!(rv, ind, ind.state[_vaccination_time_key(label)],
                    state.rng)
            continue
        end
        # The dose is given `dose_delay` days after the tracing team reached the
        # contact. `ContactTracing` records that time as `:trace_time`
        # whatever its trace action, so the isolation-derived times below are
        # reached only when something other than `ContactTracing` set
        # `:traced`, or when the trace time was `NaN` and so was not recorded.
        # A contact recorded as never reached (an infinite trace time) is not
        # vaccinated. The `haskey` branch avoids evaluating a `get` default for
        # every contact.
        trace_t = if haskey(ind.state, :trace_time)
            ind.state[:trace_time]
        else
            min(isolation_time(ind), get(ind.state, :traced_isolation_time, Inf))
        end
        vacc_t = trace_t + rv.dose_delay
        isfinite(vacc_t) || continue
        _has_required_dose(rv, ind, vacc_t) || continue
        _within_eligibility_window(rv.eligibility_window, ind, vacc_t, state.rng) ||
            continue
        _covers(rv.coverage, ind, state.rng) || continue
        _record_vaccination!(rv, ind, vacc_t, state.rng)
        rv.post_exposure_efficacy > 0.0 && _abort_infection!(rv, ind, vacc_t, state.rng)
    end
    return nothing
end

# ── MassVaccination ──────────────────────────────────────────────────

"""
Vaccinate the population on a rolling schedule, independent of contact
tracing. Each contact draws an eligibility time when they are created;
if that time is finite they are recorded as vaccinated. Whether
vaccination actually blocks transmission for that contact is then
decided by the competing-risks resolution, which checks whether the
eligibility time plus `delay_to_immunity` falls before the contact's
own transmission time.

`eligibility_time` accepts:

- a `Real`: every contact becomes eligible at this absolute time.
- a `Distribution`: each contact draws its eligibility time
  independently (e.g. `Exponential(60.0)` for a slow random rollout).
- a `Function` `(rng, ind) -> Real`: per-individual rule; use for
  age-stratified rollout or any other state-dependent schedule.
  Return `Inf` for individuals who never become eligible. Reading a
  value set by [`vaccine_acceptance`](@ref) here clusters refusal by
  ring rather than drawing it independently per contact, e.g.
  `(rng, ind) -> ind.state[:vaccine_acceptance] > 0.5 ? 30.0 : Inf`
  thresholds the shared propensity into a per-ring accept/decline;
  marginal acceptance is then `P(propensity > 0.5)`, not the
  propensity's mean, so pick the threshold against the target uptake
  rather than reading it off the propensity distribution's mean.

`efficacy` accepts the same `Real | Distribution | Function` set,
sampled once per vaccinated contact. Per-individual heterogeneous
efficacy (e.g. age-dependent) is set via the function form.

Per-contact state keys are `:vaccinated`, `:vaccination_time`, and
`:vaccine_efficacy` for the default dose label. With a non-default
`dose_label`, the keys carry the label as a suffix — pass two
`MassVaccination`s with different labels for a multi-dose rollout.

# Examples

Whole population eligible on day 30:

```julia
MassVaccination(efficacy = 0.85, eligibility_time = 30.0)
```

Per-individual rollout draws from a distribution:

```julia
MassVaccination(efficacy = 0.85,
    eligibility_time = Exponential(60.0),
    delay_to_immunity = 14.0)
```

Age-stratified rollout (65+ on day 30, younger on day 90), with
age-dependent efficacy:

```julia
MassVaccination(
    efficacy = (rng, ind) -> ind.state[:age] >= 65 ? 0.7 : 0.9,
    eligibility_time = (rng, ind) -> ind.state[:age] >= 65 ? 30.0 : 90.0,
    delay_to_immunity = 14.0,
)
```

Prime-and-boost schedule (compose two instances with different labels):

```julia
[
    MassVaccination(efficacy = 0.6,  eligibility_time = 30.0,
        delay_to_immunity = 14.0, dose_label = :prime),
    MassVaccination(efficacy = 0.9,  eligibility_time = 60.0,
        delay_to_immunity = 14.0, dose_label = :boost),
]
```
"""
Base.@kwdef struct MassVaccination{E, T, M <: AbstractEffectMode} <: AbstractVaccination
    efficacy::E
    eligibility_time::T
    delay_to_immunity::Float64 = 0.0
    mode::M = LeakyMode()
    dose_label::Symbol = :default
end

required_fields(::MassVaccination) = Symbol[]

function apply_post_transmission!(mv::MassVaccination, state, new_contacts)
    label = dose_label(mv)
    vacc_key = _vaccinated_key(label)
    for ind in new_contacts
        get(ind.state, vacc_key, false) && continue
        vacc_t = _sample_value(mv.eligibility_time, state.rng, ind)
        isfinite(vacc_t) || continue
        _record_vaccination!(mv, ind, vacc_t, state.rng)
    end
    return nothing
end
