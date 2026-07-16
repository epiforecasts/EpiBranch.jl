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
vaccine-induced immunity has developed by their transmission time."""
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

Per-contact state keys are `:vaccinated`, `:vaccination_time`, and
`:vaccine_efficacy` for the default dose label. With a non-default
`dose_label`, the keys carry the label as a suffix.
"""
Base.@kwdef struct RingVaccination{E, C, W, M <: AbstractEffectMode} <: AbstractVaccination
    efficacy::E
    coverage::C = 1.0
    delay_to_immunity::Float64 = 0.0
    eligibility_window::W = Inf
    onward_efficacy::Float64 = 0.0
    mode::M = LeakyMode()
    dose_label::Symbol = :default
end

required_fields(::RingVaccination) = [:traced]

# Onward-infectiousness risk: blocks the parent → contact transmission
# iff this dose has been administered to the *parent* and the parent's
# immunity has developed by their (the parent's) transmission time. The
# parent's `:vaccination_time` is set by ring vaccination at the parent's
# isolation time; the onward immunity comes online at that time plus
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

# Combine the susceptibility risk (acting on the contact) with the
# optional onward-infectiousness risk (acting on the parent). Returning
# a tuple of risks is supported by the engine's `_iter_risks` helper.
function competing_risk(rv::RingVaccination, parent, contact, state)
    susceptibility = _susceptibility_risk(rv, contact)
    onward = _onward_risk(rv, parent)
    susceptibility === nothing && return onward
    onward === nothing && return susceptibility
    return (susceptibility, onward)
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

function apply_post_transmission!(rv::RingVaccination, state, new_contacts)
    label = dose_label(rv)
    vacc_key = _vaccinated_key(label)
    for ind in new_contacts
        is_traced(ind) || continue
        get(ind.state, vacc_key, false) && continue
        # Fire at the trace-driven isolation time. Quarantine writes
        # `:isolation_time`; FlagOnly (for a traced contact with a known onset)
        # writes `:traced_isolation_time`. Take whichever is set so ring
        # vaccination is not silently skipped when tracing only flags contacts.
        vacc_t = min(isolation_time(ind),
            get(ind.state, :traced_isolation_time, Inf))
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
