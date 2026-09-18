"""
Base type for vaccination interventions. A vaccination has an
`efficacy` (per-exposure block probability once immunity is in place)
and a `delay_to_immunity` (time between vaccination and protection).

Concrete subtypes differ only in eligibility — who gets vaccinated
when. They share the [`competing_risk`](@ref) machinery: a vaccinated
contact whose immunity has developed by their transmission time has
their infection blocked with probability `efficacy`.

`severity_efficacy` is a fourth effect, alongside `efficacy` (contact
susceptibility) and `onward_efficacy`/`post_exposure_efficacy` on
[`RingVaccination`](@ref) (parent infectiousness / abort of an existing
infection): the probability that a vaccinated individual's own disease
course is milder, e.g. a lower chance of death or of severe symptoms,
once their immunity has developed. It has no counterpart in the
competing-risks machinery those act through, because it does not gate
transmission — it is consulted from the `probability` of a
[`Transition`](@ref) (or [`Death`](@ref), [`Hospitalisation`](@ref), or
any other clinical transition in a `progression`) via the
[`severity_efficacy`](@ref) and [`immunity_time`](@ref) accessors, the
same idiom [`Hospitalisation`](@ref) documents for prerequisite-gated
admission. See [`RingVaccination`](@ref) for a worked example.

`efficacy` and `delay_to_immunity` each accept a `Real`, a
`Distribution`, or a function `(rng, ind) -> Real`. The function/
distribution forms sample once per vaccinated individual at vaccination
time and store the result on the contact; the competing risk reads the
stored value, so a single individual keeps the same immunity delay and
efficacy against every exposure it faces.

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

"""Label of the dose a contact must already have received before this
vaccination is given, or `nothing` when it requires no earlier dose."""
required_dose(::AbstractVaccination) = nothing

"""Label namespacing the vaccination's per-contact state (`:vaccinated`,
`:vaccination_time`, `:vaccine_efficacy`, `:immunity_time`,
`:severity_efficacy`). `:default` writes to the unsuffixed keys for
backwards compatibility; other labels write to `:vaccinated_<label>` etc."""
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
function _post_exposure_efficacy_key(label::Symbol)
    label === :default ? :post_exposure_efficacy : Symbol("post_exposure_efficacy_", label)
end
function _onward_efficacy_key(label::Symbol)
    label === :default ? :onward_efficacy : Symbol("onward_efficacy_", label)
end
function _immunity_time_key(label::Symbol)
    label === :default ? :immunity_time : Symbol("immunity_time_", label)
end
function _severity_efficacy_key(label::Symbol)
    label === :default ? :severity_efficacy : Symbol("severity_efficacy_", label)
end

"""Time at which `ind`'s immunity from dose `v`, given at `vacc_t`, develops.
A scalar `delay_to_immunity` is the same for everyone and is added to
`vacc_t` directly; a varying one was drawn when the dose was given and is
read back from the stored `:immunity_time`, or `Inf` if nothing was stored."""
function _immunity_time(v::AbstractVaccination, ind, vacc_t)
    _immunity_time(v.delay_to_immunity, dose_label(v), ind, vacc_t)
end
_immunity_time(delay::Real, label, ind, vacc_t) = vacc_t + delay
function _immunity_time(delay, label, ind, vacc_t)
    get(ind.state, _immunity_time_key(label), Inf)
end

# A scalar per-dose parameter is the same for every individual, so it is read
# straight off the intervention: no dictionary lookup in the competing risk, no
# per-contact state (and so no extra line-list column), and a value carrying a
# derivative reaches the risk unchanged. A distribution or function was drawn
# once when the dose was given, and that stored draw governs every exposure of
# the individual. `key` is applied to the label only on the varying branch, so
# the scalar path builds no `Symbol`.
_dose_value(x::Real, key, label, ind) = x
_dose_value(x, key, label, ind) = get(ind.state, key(label), 0.0)

_store_draw!(x::Real, key, label, ind, rng) = nothing
function _store_draw!(x, key, label, ind, rng)
    ind.state[key(label)] = _sample_value(x, rng, ind)
    return nothing
end

"""Whether `x` could resolve to a positive value for some individual. A
`Real` answers for everyone and a `Distribution` answers from its support.
A function, or a distribution whose support cannot be read, is taken to be
possibly positive: that costs a risk being built, never protection. The
per-individual draw is what gates the risk at use time."""
_maybe_positive(x::Real) = x > 0.0
function _maybe_positive(d::Distribution)
    hi = _support_bound(maximum, d)
    return hi === nothing || hi > 0
end
_maybe_positive(x) = true

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
    return Risk(event_time = _immunity_time(v, contact, vacc_t), block_probability = eff)
end

function competing_risk(v::AbstractVaccination, parent, contact, state)
    _susceptibility_risk(v, contact)
end

# Helper for concrete subtypes: write per-dose state on a contact at
# vaccination time. Samples efficacy, severity efficacy, and the
# immunity delay via `_sample_value` so scalar, distribution, and
# function forms all work. The delay is drawn here, once, rather than
# inside `competing_risk`, so a given individual's immunity time is the
# same against every exposure it faces. Storing the resulting immunity
# time also lets a clinical transition check it without reaching for the
# vaccination object, which it never sees.
function _record_vaccination!(v::AbstractVaccination, contact, vacc_t, rng)
    label = dose_label(v)
    contact.state[_vaccinated_key(label)] = true
    contact.state[_vaccination_time_key(label)] = vacc_t
    contact.state[_vaccine_efficacy_key(label)] = _sample_value(v.efficacy, rng, contact)
    contact.state[_immunity_time_key(label)] = vacc_t +
                                               _sample_value(v.delay_to_immunity, rng, contact)
    contact.state[_severity_efficacy_key(label)] = _sample_value(v.severity_efficacy, rng, contact)
    _record_effect_draws!(v, contact, label, rng)
    return nothing
end

# Draws for effects only some vaccinations have (`post_exposure_efficacy` and
# `onward_efficacy` exist only on `RingVaccination`).
_record_effect_draws!(::AbstractVaccination, contact, label, rng) = nothing

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
age-dependent).

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

`severity_efficacy` is the probability that the vaccinated *contact's own*
disease course is milder once their immunity has developed — lower
mortality, a lower chance of a severe outcome, or whatever a clinical
transition's `probability` reads it for. Unlike `efficacy`,
`onward_efficacy` and `post_exposure_efficacy`, it does not gate
transmission and so is not one of the risks [`competing_risk`](@ref)
returns: it has no effect until a transition in `progression` consults it,
via the [`severity_efficacy`](@ref) and [`immunity_time`](@ref) accessors,
same as any other prerequisite a clinical transition's `probability` gates
on:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```

`immunity_time(ind) <= onset_time(ind)` is the check that a dose whose
immunity has not yet developed by the outcome it would affect confers no
protection — comparing against `:vaccinated` alone, as a naive closure
might, would count a not-yet-immune dose as protective. Defaults to `0.0`
(no severity effect). Accepts a `Real`, `Distribution`, or `Function`
`(rng, ind) -> Real`, sampled once per vaccinated contact alongside
`efficacy`.

Per-contact state keys are `:vaccinated`, `:vaccination_time`,
`:vaccine_efficacy`, `:immunity_time`, and `:severity_efficacy` for the
default dose label, plus `:post_exposure_efficacy` and `:onward_efficacy`
where those are given as a distribution or function. With a non-default
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
    RingVaccination(efficacy = 0.5, dose_delay = Uniform(28.0, 42.0),
        delay_to_immunity = 14.0, coverage = 0.9,
        requires_dose = :prime, dose_label = :boost),
]
```

The boost is given, on average, five weeks after the trace to 90% of
those primed (the remaining 10% being lost to follow-up), and protects
14 days later. List a dose after the dose it requires: the stack is
applied in order, so a boost placed first would see no prime and never
be given. The required
dose may come from any vaccination, such as a [`MassVaccination`](@ref)
prime; a contact whose prime falls after the boost's due date is not
boosted. Between two ring doses, a `dose_delay` shorter than the required
dose's is rejected when the `ModelSpec` is built, since such a boost could
never be given. A `dose_delay` drawn from a distribution is judged on its
support: the boost is rejected when even its longest delay falls before
the required dose's shortest, and warned about when the two supports
overlap, because contacts whose draws come out in the wrong order go
without the boost. A function, or a distribution that reports no
support, cannot be checked when the `ModelSpec` is built; a dose whose
draw falls before the required dose is then declined for that contact.

Doses compose as competing risks, so a schedule reaching 80% protection
in total from a prime at 60% needs `efficacy = 0.5` on the boost
(`(0.8 - 0.6) / (1 - 0.6)`), the protection the second dose adds among
those the first left unprotected.

A dose is recorded when it falls due, whether or not the contact was
infected in the meantime, because infection is resolved after the doses
are given. Dose counts for later doses are therefore counts of doses
scheduled.

`delay_to_immunity`, `post_exposure_efficacy`, and `onward_efficacy`
accept the same `Real | Distribution | Function` forms as `efficacy`, drawn
once per contact when the dose is given (see [`AbstractVaccination`](@ref)).
So does `dose_delay`, drawn once when the dose is scheduled.
"""
Base.@kwdef struct RingVaccination{
    E, C, DI, DD, W, PE, OE, SV, M <: AbstractEffectMode
} <: AbstractVaccination
    efficacy::E
    coverage::C = 1.0
    delay_to_immunity::DI = 0.0
    dose_delay::DD = 0.0
    requires_dose::Union{Nothing, Symbol} = nothing
    eligibility_window::W = Inf
    post_exposure_efficacy::PE = 0.0
    onward_efficacy::OE = 0.0
    severity_efficacy::SV = 0.0
    mode::M = LeakyMode()
    dose_label::Symbol = :default
end

function required_fields(rv::RingVaccination)
    _maybe_positive(rv.post_exposure_efficacy) ? [:traced, :incubation_period] : [:traced]
end
required_dose(rv::RingVaccination) = rv.requires_dose

function _record_effect_draws!(rv::RingVaccination, contact, label, rng)
    _store_draw!(rv.post_exposure_efficacy, _post_exposure_efficacy_key, label, contact,
        rng)
    _store_draw!(rv.onward_efficacy, _onward_efficacy_key, label, contact, rng)
    return nothing
end

# Onward-infectiousness risk: blocks the parent → contact transmission
# iff this dose has been administered to the *parent* and the parent's
# immunity has developed by their (the parent's) transmission time. The
# parent's `:vaccination_time` is set by ring vaccination when the parent
# was traced, and the onward immunity takes effect at that time plus
# `delay_to_immunity`, as on the susceptibility side.
function _onward_risk(rv::RingVaccination, parent, contact)
    # A community introduction has no infector, and the person stands in for it
    # on the continuous-time models. A dose cannot reduce what a source outside
    # the population transmits, so it contributes nothing there.
    parent === contact && return nothing
    label = dose_label(rv)
    onward = _dose_value(rv.onward_efficacy, _onward_efficacy_key, label, parent)
    onward > 0 || return nothing
    get(parent.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(parent.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    return Risk(event_time = _immunity_time(rv, parent, vacc_t),
        block_probability = onward)
end

# Contact-side risk. `efficacy` blocks an exposure that comes after immunity,
# and so does `post_exposure_efficacy`: a dose whose immunity is already in
# place when the contact is exposed prevents the infection outright, the limit
# of aborting it the moment it starts. This is also all a dose can do for a
# contact with no onset to race (asymptomatic). Both act at the same immunity
# time, so they combine into one block, drawn once.
function _contact_risk(rv::RingVaccination, contact)
    susceptibility = _susceptibility_risk(rv, contact)
    label = dose_label(rv)
    # A contact with no dose of this vaccination has no draw stored, so a
    # varying `post_exposure_efficacy` reads zero here.
    post = _dose_value(rv.post_exposure_efficacy, _post_exposure_efficacy_key, label,
        contact)
    post > 0.0 || return susceptibility
    if susceptibility === nothing
        get(contact.state, _vaccinated_key(label), false) || return nothing
        vacc_t = get(contact.state, _vaccination_time_key(label), Inf)
        isfinite(vacc_t) || return nothing
        return Risk(event_time = _immunity_time(rv, contact, vacc_t),
            block_probability = post)
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
# Callers check `_maybe_positive(rv.post_exposure_efficacy)` first: the
# vaccination time comes untyped from the contact's state, so an
# unconditional call would dispatch dynamically for every vaccinated
# contact of a dose that cannot abort.
function _abort_infection!(rv::RingVaccination, contact, vacc_t, rng)
    incubation = get(contact.state, :incubation_period, NaN)
    isnan(incubation) && return nothing
    post = _dose_value(rv.post_exposure_efficacy, _post_exposure_efficacy_key,
        dose_label(rv), contact)
    post > 0.0 || return nothing
    immunity = _immunity_time(rv, contact, vacc_t)
    exposure = contact.infection_time
    exposure < immunity < exposure + incubation || return nothing
    _covers(post, contact, rng) || return nothing
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
    onward = _onward_risk(rv, parent, contact)
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
it is never given. Other schedules, and delays whose bounds cannot be read,
are checked per contact when the dose falls due.
"""
function _validate_dose_schedule(interventions)
    # Each delay is held as it was given, so a `dose_delay` carrying a
    # derivative passes through the schedule unconverted.
    given = Dict{Symbol, Any}()
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
            _check_dose_order(label, req, offset, given[req])
        end
        given[label] = offset
        _warn_double_counted_efficacy(vacc)
    end
    return nothing
end

# Both ring doses are timed from the same trace, so their `dose_delay`s decide
# whether the later dose can be given at all. A delay drawn from a distribution
# is judged on its support: a dose whose latest possible delay still falls before
# the earliest possible delay of the dose it requires could never be given, and
# is rejected as a scalar one would be. Supports that merely overlap boost some
# contacts and skip the rest, which is worth a warning since the reason for the
# missing doses is otherwise invisible.
function _check_dose_order(label, req, offset, req_offset)
    bounds = _delay_bounds(offset)
    req_bounds = _delay_bounds(req_offset)
    (bounds === nothing || req_bounds === nothing) && return nothing
    lo, hi = bounds
    req_lo, req_hi = req_bounds
    hi < req_lo && throw(ArgumentError(
        "vaccination with dose_label = :$label requires dose :$req but is " *
        "scheduled earlier than it (a dose_delay of at most $hi days after the " *
        "trace, against at least $req_lo for dose :$req). A dose cannot be " *
        "given before the dose it requires."))
    lo < req_hi && @warn "This dose's dose_delay $(_reaches_below(lo)), so it can " *
          "fall before dose :$req, which it requires and whose own dose_delay " *
          "$(_reaches_above(req_hi)). Contacts whose draws come out in that " *
          "order go without this dose." dose_label=label
    return nothing
end

# An unbounded support has no number worth quoting, so the warning describes it
# in words instead.
function _reaches_below(lo)
    isfinite(lo) ? "reaches $lo days after the trace" : "has no lower bound"
end
_reaches_above(hi) = isfinite(hi) ? "can reach $hi days" : "has no upper bound"

"""Bounds `(lo, hi)` on a dose delay, or `nothing` where they cannot be read: a
dose not timed from the trace, a delay given as a function, or a distribution
that does not report its support."""
_delay_bounds(x::Real) = (x, x)
function _delay_bounds(d::Distribution)
    lo = _support_bound(minimum, d)
    hi = _support_bound(maximum, d)
    return (lo === nothing || hi === nothing) ? nothing : (lo, hi)
end
_delay_bounds(_) = nothing

# `minimum` and `maximum` are an optional part of the `Distribution` interface: a
# distribution defining only `rand` and `logpdf` (the package's own
# `_TruncatedSkewNormal` among them) falls through to `Base.minimum`, which tries
# to iterate it and throws a `MethodError`. Such a bound comes back as `nothing`
# and the caller makes the cautious assumption. Any other error is the
# distribution failing for its own reasons and is rethrown.
function _support_bound(f, d)
    bound = try
        f(d)
    catch err
        err isa MethodError || rethrow()
        return nothing
    end
    return bound isa Real ? bound : nothing
end

# Immunity before onset is a weaker condition than immunity before exposure, so
# a dose setting both fields blocks with `1 - (1 - e1)(1 - e2)` for any contact
# vaccinated before its exposure. That is common once isolation is leaky,
# because a contact's trace time comes from its infector's course and not from
# its own. Only the scalar forms can be checked; a function or distribution
# efficacy or post_exposure_efficacy is left alone.
_warn_double_counted_efficacy(::AbstractVaccination) = nothing
function _warn_double_counted_efficacy(rv::RingVaccination)
    rv.post_exposure_efficacy isa Real && rv.post_exposure_efficacy > 0.0 || return nothing
    rv.efficacy isa Real && rv.efficacy > 0.0 || return nothing
    @warn "RingVaccination sets both `efficacy` and `post_exposure_efficacy`, "*
          "which compose as independent risks and so over-protect any contact "*
          "vaccinated before its exposure. `post_exposure_efficacy` already "*
          "covers those contacts; set one or the other." dose_label=dose_label(rv) maxlog=1
    return nothing
end

"""Days from the trace to this dose, as the `dose_delay` was given, or `nothing`
for a vaccination not timed from the trace. [`RingVaccination`](@ref) is the only
one timed from it."""
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
            _maybe_positive(rv.post_exposure_efficacy) &&
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
        # vaccinated, and is skipped before its `dose_delay` is drawn so it
        # leaves the random stream untouched. The `haskey` branch avoids
        # evaluating a `get` default for every contact.
        trace_t = if haskey(ind.state, :trace_time)
            ind.state[:trace_time]
        else
            min(isolation_time(ind), get(ind.state, :traced_isolation_time, Inf))
        end
        isfinite(trace_t) || continue
        vacc_t = trace_t + _sample_value(rv.dose_delay, state.rng, ind)
        isfinite(vacc_t) || continue
        _has_required_dose(rv, ind, vacc_t) || continue
        _within_eligibility_window(rv.eligibility_window, ind, vacc_t, state.rng) ||
            continue
        _covers(rv.coverage, ind, state.rng) || continue
        _record_vaccination!(rv, ind, vacc_t, state.rng)
        _maybe_positive(rv.post_exposure_efficacy) &&
            _abort_infection!(rv, ind, vacc_t, state.rng)
    end
    return nothing
end

# ── GroupVaccination ─────────────────────────────────────────────────

"""
Vaccinate every member of the group a confirmed case belongs to — the
fallback an outbreak response reaches for when no ring can be built, such
as a village, a health area, or a household. Individuals carry their
group under `group_key` (`:group` by default; see [`groups`](@ref)), and
every member sharing a triggering case's group is vaccinated at the
trigger time plus `dose_delay`, whether or not it has any traced
connection to that case.

`eligibility` is a [`TraceEligibility`](@ref) policy, exactly as
[`ContactTracing`](@ref) uses it, but tested against a case itself rather
than an infector–contact pair: `OnLabConfirmation()` (the default) fires
once a case in the group has tested positive, `OnSymptomOnset()` fires on
suspicion alone, and the boolean operators `&`, `|`, `!` combine them the
same way. The policy's [`trigger_time`](@ref EpiBranch.trigger_time) sets
when the group is deemed to have a case in it; `dose_delay` is added on
top, standing in for the time a vaccination team takes to reach the
group once notified.

A group is vaccinated as soon as any of its members meets `eligibility`,
at the *earliest* such member's trigger time — later confirmations in the
same group do not push the dose out further. Members created after the
trigger (a case infected later in the same group) are vaccinated at that
same trigger time plus `dose_delay` when they appear, so they are
protected only from exposures after that point, exactly as an
already-present member is. Because the campaign reaches the whole group,
doses scale with group size where [`RingVaccination`](@ref) doses scale
with ring size.

`coverage`, `efficacy`, `severity_efficacy`, `delay_to_immunity`, `mode`,
and `dose_label` mean what they do for [`RingVaccination`](@ref).
`severity_efficacy` defaults to `0.0` (no severity effect) and, as there,
acts only through a clinical transition that reads it via the
[`severity_efficacy`](@ref) and [`immunity_time`](@ref) accessors.
`dose_delay` accepts the same forms, drawn once per member, so members of one
group can be reached at different times.

# Fallback composition

Listing a [`RingVaccination`](@ref) before a `GroupVaccination` with the
same `dose_label` makes the group dose a pure fallback: `coverage`,
`efficacy`, and vaccination state are namespaced by `dose_label` (see
[`AbstractVaccination`](@ref)), so a member the ring already reached is
recorded as vaccinated by the time `GroupVaccination` runs and is
skipped, leaving the group dose to reach only those the ring did not:

```julia
[ContactTracing(OnLabConfirmation(), 0.7, Exponential(1.0)),
 RingVaccination(efficacy = 0.9),
 GroupVaccination(efficacy = 0.6)]
```

Requires `:group` (or `group_key`) and whatever `eligibility` requires,
e.g. `:test_positive` for the default `OnLabConfirmation()`.

!!! note "Seed cases"
    Like [`RingVaccination`](@ref) and [`MassVaccination`](@ref),
    `GroupVaccination` acts through `apply_post_transmission!`, which the
    engine calls only on newly created contacts. A run's seed cases are
    created directly, not through that hook, so a seed is reached only once
    another member of its group is created later and re-triggers the sweep;
    a seed whose chain goes extinct without ever sharing a group with a
    later case is not vaccinated even if it is itself confirmed.

# Examples

Vaccinate the whole village once a case there is lab-confirmed, 2 days
later:

```julia
GroupVaccination(efficacy = 0.7, eligibility = OnLabConfirmation(), dose_delay = 2.0)
```
"""
Base.@kwdef struct GroupVaccination{
    E <: TraceEligibility, Ef, C, SV, DI, DD, M <: AbstractEffectMode} <:
                   AbstractVaccination
    eligibility::E = OnLabConfirmation()
    efficacy::Ef
    coverage::C = 1.0
    severity_efficacy::SV = 0.0
    delay_to_immunity::DI = 0.0
    dose_delay::DD = 0.0
    group_key::Symbol = :group
    mode::M = LeakyMode()
    dose_label::Symbol = :default
end

function required_fields(gv::GroupVaccination)
    union([gv.group_key], required_fields(gv.eligibility))
end

# The group's trigger time: the earliest time any of its members (found by
# scanning every individual created so far, not just this generation's
# `new_contacts`) meets `eligibility`, tested against the member itself in
# both the infector and contact slots since the policy describes a property
# of a case, not a pair. `Inf` if no member has triggered yet.
function _group_trigger_time(gv::GroupVaccination, state, group)
    key = gv.group_key
    t = Inf
    for m in state.individuals
        get(m.state, key, nothing) == group || continue
        is_eligible(gv.eligibility, m, m, state) || continue
        tt = trigger_time(gv.eligibility, m, state)
        isnan(tt) && continue
        t = min(t, tt)
    end
    return t
end

# Two things can happen to a group in a single generation: a member created
# earlier newly meets `eligibility` (the group's first trigger), or a member
# is created into a group that already triggered in an earlier generation.
# Recomputing the trigger time for every group touched by `new_contacts` and
# sweeping the whole population against it handles both in one pass: a fresh
# trigger reaches members already present, and a standing one reaches a
# member only now created. Groups untouched this generation are left alone,
# so nobody outside a triggered group is ever visited.
function apply_post_transmission!(gv::GroupVaccination, state, new_contacts)
    key = gv.group_key
    label = dose_label(gv)
    vacc_key = _vaccinated_key(label)

    groups_here = Set{Any}()
    for ind in new_contacts
        haskey(ind.state, key) && push!(groups_here, ind.state[key])
    end

    for group in groups_here
        trigger = _group_trigger_time(gv, state, group)
        isfinite(trigger) || continue
        for m in state.individuals
            get(m.state, key, nothing) == group || continue
            get(m.state, vacc_key, false) && continue
            _covers(gv.coverage, m, state.rng) || continue
            vacc_t = trigger + _sample_value(gv.dose_delay, state.rng, m)
            _record_vaccination!(gv, m, vacc_t, state.rng)
        end
    end
    return nothing
end

traces_contacts(::RingVaccination) = true

# Whether a ring can dose on the continuous-time race. Its eligibility window
# and its post-exposure abort both measure from the contact's exposure, which a
# contact the race has not settled does not have yet, so a ring using either
# cannot be applied there faithfully.
function _ring_doses_on_race(rv::RingVaccination)
    rv.post_exposure_efficacy == 0.0 &&
        rv.eligibility_window isa Real && rv.eligibility_window == Inf
end

"""Dose the contacts a case reaches on the continuous-time path, where the
generation engine's round of new contacts does not exist. The dosing policy
reads nothing but each contact's own tracing state, so it is the same loop
either way: list ring vaccination after the [`ContactTracing`](@ref) that feeds
it, as on the generation engine, and it doses whoever that trace has just
reached. Whether the dose then blocks an infection is the competing risk's
business, and the continuous-time race resolves that on each contact it
proposes.

A ring with a finite `eligibility_window` or a `post_exposure_efficacy` doses
no one here: both are measured from the contact's own exposure, which a contact
the race has not settled does not have yet. The model warns about such a ring
rather than apply either from the wrong time."""
function trace_contacts!(rv::RingVaccination, state, infector, contacts)
    _ring_doses_on_race(rv) || return nothing
    for ind in contacts
        _advance_ring_dose!(rv, ind)
    end
    # A contact reappears in the trace of every neighbour that settles after it
    # was traced. It is offered the dose once, as a traced contact is on the
    # generation engine, so one that declined is not offered it again.
    offered_key = _ring_offered_key(dose_label(rv))
    candidates = [ind for ind in contacts if !get(ind.state, offered_key, false)]
    apply_post_transmission!(rv, state, candidates)
    for ind in candidates
        is_traced(ind) && isfinite(get(ind.state, :trace_time, Inf)) &&
            (ind.state[offered_key] = true)
    end
    return nothing
end

function _ring_offered_key(label::Symbol)
    label === :default ? :ring_dose_offered : Symbol("ring_dose_offered_", label)
end

# The race settles cases in infection order, not trace order, so a case settled
# later can reach a contact sooner than the one that dosed it. Its trace lowers
# the contact's `:trace_time`, and the dose, given at the trace, moves with it.
# Coverage and efficacy were drawn with the first dose and stay as they are.
function _advance_ring_dose!(rv::RingVaccination, ind)
    label = dose_label(rv)
    get(ind.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(ind.state, :trace_time, Inf) + rv.dose_delay
    vacc_t < ind.state[_vaccination_time_key(label)] || return nothing
    _has_required_dose(rv, ind, vacc_t) || return nothing
    ind.state[_vaccination_time_key(label)] = vacc_t
    ind.state[_immunity_time_key(label)] = vacc_t + delay_to_immunity(rv)
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
  Return `Inf` for individuals who never become eligible.

`efficacy` and `delay_to_immunity` accept the same set, drawn once per
vaccinated contact (see [`AbstractVaccination`](@ref)), for example an
age-dependent efficacy or immunity that takes one to three weeks to develop.

`severity_efficacy` accepts the same set and, like on
[`RingVaccination`](@ref), sets how much milder a vaccinated individual's
own disease course is once their immunity has developed — a clinical
transition's `probability` reads it via the [`severity_efficacy`](@ref)
and [`immunity_time`](@ref) accessors. Defaults to `0.0` (no severity
effect).

Per-contact state keys are `:vaccinated`, `:vaccination_time`,
`:vaccine_efficacy`, `:immunity_time`, and `:severity_efficacy` for the
default dose label. With a non-default `dose_label`, the keys carry the
label as a suffix — pass two `MassVaccination`s with different labels for
a multi-dose rollout.

# Examples

Whole population eligible on day 30:

```julia
MassVaccination(efficacy = 0.85, eligibility_time = 30.0)
```

Per-individual rollout draws from a distribution, for a vaccine whose
immunity takes one to three weeks to develop:

```julia
MassVaccination(efficacy = 0.85,
    eligibility_time = Exponential(60.0),
    delay_to_immunity = Uniform(7.0, 21.0))
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
Base.@kwdef struct MassVaccination{E, T, DI, SV, M <: AbstractEffectMode} <:
                   AbstractVaccination
    efficacy::E
    eligibility_time::T
    delay_to_immunity::DI = 0.0
    severity_efficacy::SV = 0.0
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
