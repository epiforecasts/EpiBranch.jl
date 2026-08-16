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

"""Dose label this vaccination requires a contact to already carry before
it is given, or `nothing` when the dose stands on its own."""
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

Note that a dose has to *precede* the exposure it blocks. On a branching
process each individual enters the simulation at their own exposure, so
this fires only where a trace can outrun an exposure: a wider ring
reaching past infected members who never become eligible to seed a ring
of their own, or an isolation leaky enough that an infector keeps
transmitting after the ring has gone out. For the ordinary case of a
dose given to a contact already exposed, see `post_exposure_efficacy` on
[`RingVaccination`](@ref)."""
function _susceptibility_risk(v::AbstractVaccination, contact)
    label = dose_label(v)
    get(contact.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(contact.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    eff = get(contact.state, _vaccine_efficacy_key(label), nothing)
    eff === nothing && return nothing
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
`delay_to_immunity = 0.0` (the default). For ring vaccination with a
vaccine that takes time to confer protection, set `delay_to_immunity`
to the appropriate delay.

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
    that is the time the window is checked against. Copying a prime's
    `eligibility_window` onto a boost therefore rejects almost every
    boost, since `dose_delay` alone usually exceeds the window. A window
    belongs on the dose whose timing it describes, so leave a later dose
    at the default `Inf` unless the protocol really does have a deadline
    on the second dose itself.

`post_exposure_efficacy` is the probability that a dose given to an
already-exposed contact aborts that infection, which it can do so long
as immunity arrives before the infection declares itself (vaccination +
`delay_to_immunity` before the contact's symptom onset). This is the
mechanism post-exposure ring vaccination works through, and on a
branching process it is normally the parameter you want: a traced
contact was exposed at the moment they entered the simulation, so
`efficacy` — which asks for immunity *before* the exposure — has
nothing to gate. Defaults to `0.0`. A contact with no onset to race
(asymptomatic, `NaN` incubation period) is protected only if immunity
was in place before their exposure. Requires `:incubation_period`, set
by [`clinical_presentation`](@ref).

Immunity before onset is a weaker condition than immunity before
exposure, so `post_exposure_efficacy` already covers every contact
`efficacy` would have protected. Set one or the other: setting both
composes them as independent risks, which double-counts.

`onward_efficacy` is the per-exposure probability that a *vaccinated
parent's* onward transmission is blocked once the parent's
vaccine-induced immunity has developed — the post-exposure
prophylaxis mechanism by which ring vaccination averts onward cases
even for contacts who were already exposed at the time of
vaccination. Defaults to `0.0` (no onward effect). `efficacy` (the
susceptibility-side block applied when the *contact* is vaccinated)
still applies independently; setting both to the same value gives a
vaccine that acts symmetrically on susceptibility and infectiousness,
setting only `onward_efficacy` gives a pure PEP effect.

Requires `:traced` (set by [`ContactTracing`](@ref)).

!!! warning "Redundant on top of an airtight quarantine"
    Against `ContactTracing`'s default `Quarantine` action combined with a
    non-leaky `Isolation`, adding this intervention leaves the results
    unchanged: the dose is given when the contact is traced, which is also
    when the quarantine starts, so isolation already blocks every
    transmission the dose would have. Measure ring vaccination against
    tracing that follows contacts up without confining them
    (`quarantine_on_trace = false`), or against an isolation that is
    delayed or leaky (`post_isolation_transmission > 0`), where a dose
    still has something left to do.

Per-contact state keys are `:vaccinated`, `:vaccination_time`, and
`:vaccine_efficacy` for the default dose label. With a non-default
`dose_label`, the keys carry the label as a suffix.

# Second and later doses

A dose is given at the trace. `dose_delay` moves it later by a fixed
number of days, and `requires_dose` names a dose label the contact must
already carry for this one to be given at all, so a prime-boost schedule
is two `RingVaccination`s:

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
boost placed first sees no prime and never fires.

Doses compose as competing risks, so a schedule reaching 80% protection
in total from a prime at 60% needs `efficacy = 0.5` on the boost
(`(0.8 - 0.6) / (1 - 0.6)`), the protection the second dose adds among
those the first left unprotected.

A dose is recorded when it falls due, whether or not the contact was
infected in the meantime — infection is resolved after the doses are
given and is not knowable at that point. Dose counts for later doses are
therefore doses scheduled.
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
# was traced; the onward immunity comes online at that time plus
# `delay_to_immunity`, exactly as for the susceptibility side.
function _onward_risk(rv::RingVaccination, parent)
    rv.onward_efficacy > 0.0 || return nothing
    label = dose_label(rv)
    get(parent.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(parent.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    return Risk(event_time = vacc_t + delay_to_immunity(rv),
        block_probability = rv.onward_efficacy)
end

# Post-exposure risk: a dose given after the contact was exposed can still
# abort that infection, so long as immunity arrives before the infection
# declares itself. The engine asks whether a risk's `event_time` falls at or
# before the transmission time, so shifting the immunity time back by the
# contact's incubation period turns that test into
#
#     vaccination + delay_to_immunity <= exposure + incubation
#
# which is immunity arriving before symptom onset.
#
# A contact with no onset to race (asymptomatic, `NaN` incubation) falls back
# to the stricter pre-exposure condition rather than dropping out. Dropping out
# would leave asymptomatic contacts reachable only by `efficacy`, so the two
# fields would be complementary and a user would have to set both — and then
# double-count on every symptomatic contact protected before exposure.
function _post_exposure_risk(rv::RingVaccination, contact)
    rv.post_exposure_efficacy > 0.0 || return nothing
    label = dose_label(rv)
    get(contact.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(contact.state, _vaccination_time_key(label), Inf)
    isfinite(vacc_t) || return nothing
    immunity = vacc_t + delay_to_immunity(rv)
    incubation = get(contact.state, :incubation_period, NaN)
    event_time = isnan(incubation) ? immunity : immunity - incubation
    return Risk(event_time = event_time,
        block_probability = rv.post_exposure_efficacy)
end

# Ring vaccination gates a transmission through three mechanisms: protection
# of the contact before exposure, protection of the contact after exposure but
# before their infection declares itself, and reduced onward transmission from
# a vaccinated parent. Returning a tuple of risks is supported by the engine's
# `_iter_risks` helper, which applies each independently.
#
# The branches are written out rather than compacted generically. Each risk is
# a `Union{Nothing, Risk}`, so a generic compaction infers as
# `Tuple{Vararg{Risk}}` — abstract, of unknown length — and the engine then
# iterates a non-inferrable tuple once per contact. Enumerating the cases keeps
# the return type a small union of concrete tuple lengths, which matters
# because this sits on the per-contact hot path of every simulation carrying a
# vaccination.
function competing_risk(rv::RingVaccination, parent, contact, state)
    susceptibility = _susceptibility_risk(rv, contact)
    post_exposure = _post_exposure_risk(rv, contact)
    onward = _onward_risk(rv, parent)
    if susceptibility === nothing
        post_exposure === nothing && return onward
        onward === nothing && return post_exposure
        return (post_exposure, onward)
    elseif post_exposure === nothing
        onward === nothing && return susceptibility
        return (susceptibility, onward)
    elseif onward === nothing
        return (susceptibility, post_exposure)
    end
    return (susceptibility, post_exposure, onward)
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

# A dose that requires an earlier one is given only to contacts already
# carrying it, so `coverage` on the later dose reads as retention among
# those who got the earlier one.
_has_required_dose(v::AbstractVaccination, ind) = _has_required_dose(required_dose(v), ind)
_has_required_dose(::Nothing, ind) = true
_has_required_dose(label::Symbol, ind) = get(ind.state, _vaccinated_key(label), false)

"""
    _validate_dose_schedule(interventions)

Check that every vaccination requiring an earlier dose is listed after the
dose it requires, and is not scheduled to arrive before it. The stack is
applied in order, so a dose placed before the one it requires would silently
never be given; and because the required dose's flag is set at the trace
whatever its own `dose_delay`, a shorter `dose_delay` on the later dose would
otherwise let it be administered first.
"""
function _validate_dose_schedule(interventions)
    given = Dict{Symbol, Float64}()
    for iv in interventions
        vacc = _unwrap_scheduled(iv)
        vacc isa AbstractVaccination || continue
        label = dose_label(vacc)
        req = required_dose(vacc)
        if req !== nothing
            haskey(given, req) || throw(ArgumentError(
                "vaccination with dose_label = :$label requires dose :$req, " *
                "which is not given earlier in the intervention stack. " *
                "List a dose after the dose it requires."))
            if _dose_offset(vacc) < given[req]
                throw(ArgumentError(
                    "vaccination with dose_label = :$label requires dose :$req " *
                    "but is scheduled earlier than it ($(_dose_offset(vacc)) days " *
                    "after the trace against $(given[req])). A dose cannot be " *
                    "given before the dose it requires."))
            end
        end
        given[label] = _dose_offset(vacc)
    end
    return nothing
end

"""Days from the triggering event to this dose being administered. Only
[`RingVaccination`](@ref) delays a dose relative to its trigger."""
_dose_offset(::AbstractVaccination) = 0.0
_dose_offset(rv::RingVaccination) = rv.dose_delay

# The intervention a wrapper stands in for; `Scheduled` adds its method.
_unwrap_scheduled(iv) = iv

function apply_post_transmission!(rv::RingVaccination, state, new_contacts)
    label = dose_label(rv)
    vacc_key = _vaccinated_key(label)
    for ind in new_contacts
        is_traced(ind) || continue
        get(ind.state, vacc_key, false) && continue
        _has_required_dose(rv, ind) || continue
        # Fire when the tracing team reached the contact. `ContactTracing`
        # records that as `:trace_time` whatever its trace action, so the
        # isolation-derived times below are reached only when something
        # other than `ContactTracing` set `:traced` — or when the trace time
        # was not finite and so was not recorded. Branch rather than passing
        # a `get` default, which Julia would evaluate on every contact.
        trace_t = if haskey(ind.state, :trace_time)
            ind.state[:trace_time]
        else
            min(isolation_time(ind), get(ind.state, :traced_isolation_time, Inf))
        end
        vacc_t = trace_t + rv.dose_delay
        isfinite(vacc_t) || continue
        _within_eligibility_window(rv.eligibility_window, ind, vacc_t, state.rng) ||
            continue
        _covers(rv.coverage, ind, state.rng) || continue
        _record_vaccination!(rv, ind, vacc_t, state.rng)
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
  Return `Inf` for individuals who never become eligible.

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
