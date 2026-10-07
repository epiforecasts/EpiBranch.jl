"""
Parent type of the vaccination interventions: [`RingVaccination`](@ref)
(traced contacts), [`GroupVaccination`](@ref) (everyone in a case's group)
and [`MassVaccination`](@ref) (the population on a schedule). They differ in
who is vaccinated and when; the dose's effects are the same for all three and are set by these keywords (collected in a [`VaccineEffect`](@ref)):

| Effect | What it does | Keyword |
|:-------|:-------------|:--------|
| Against infection | Probability that an exposure of a vaccinated person does not infect them, once immune | `efficacy` |
| Against severe disease | Probability that a vaccinated person's own illness is milder, once immune | `severity_efficacy` |
| Delay | Days from vaccination to immunity | `delay_to_immunity` |
| Waning | Fraction of protection left a given number of days after immunity | `waning` |
| Leaky or all-or-nothing | How `efficacy` acts | `mode` |
| Dose name | Tells doses of a multi-dose schedule apart | `dose_label` |

[`RingVaccination`](@ref) adds two effects on infections a contact already
has: `post_exposure_efficacy` (stopping the infection) and `onward_efficacy`
(reducing onward transmission).

`efficacy`, `severity_efficacy` and `delay_to_immunity` each take a number, a
distribution or a function of the random number generator and the individual,
`(rng, ind) -> ...`. A distribution or function is drawn once per vaccinated
person when the dose is given, so the same person keeps the same efficacy and
immunity delay against every exposure.

# Leaky and all-or-nothing vaccines

With `mode = LeakyMode()` (the default) every exposure of a vaccinated
person is prevented with probability `efficacy`. With
`mode = AllOrNothingMode()` a share `efficacy` of vaccinated people are fully
protected against infection from their immunity date on and the rest get no
protection against infection. The mode acts on `efficacy` alone:
`severity_efficacy`, `post_exposure_efficacy` and `onward_efficacy` apply to
everyone vaccinated. `AllOrNothingMode` cannot yet be combined with `waning`
(this raises an error).

!!! note "The two modes differ only with repeated exposure"
    In a branching process each contact is a separate person exposed once,
    so the two modes give the same infection probability per contact
    (though not the same random draws). They differ in the homogeneous,
    household and network models, where a person can be exposed more than
    once: under all-or-nothing a protected person stays protected against
    every exposure, while a leaky vaccine is worn down by repeated
    exposure.

# Severity

`severity_efficacy` does not stop transmission. It acts only where a
clinical transition in `progression` (such as [`Death`](@ref) or
[`Hospitalisation`](@ref)) reads it through the [`severity_efficacy`](@ref)
and [`immunity_time`](@ref) functions in its `probability`;
[`RingVaccination`](@ref) has a worked example.

# Waning

`waning` is a function of the days `dt` since the person became immune,
giving the fraction of protection left (`dt = 0` at immunity), or `nothing`
(default) for constant protection. It multiplies `efficacy` (and, for
[`RingVaccination`](@ref), `onward_efficacy` and `post_exposure_efficacy`)
at each exposure, scaling each person's own drawn efficacy from their own
immunity date. It matters most when vaccination comes long before exposure.
Post-exposure protection acts at the moment of immunity and so uses
`waning(0)`. `waning` does not apply to `severity_efficacy`;
[`RingVaccination`](@ref) shows how to make that wane too.

# Multi-dose schedules

Give each dose its own `dose_label` (default `:default`), for example
`:prime` and `:boost`. Each dose then has its own vaccination record and
wanes from its own immunity date. The doses act independently, so an
exposure gets through all of them with probability `prod(1 - eff_i * w_i)`,
where `eff_i` is dose `i`'s efficacy and `w_i` the fraction it retains: a
prime at 0.6 and a boost at 0.7, both at full strength, together prevent
0.88 of exposures. [`is_vaccinated`](@ref) reports the dose with the default
label; pass its `dose_label` keyword to check another.
"""
abstract type AbstractVaccination <: AbstractIntervention end

"""
How a vaccine's `efficacy` acts: [`LeakyMode`](@ref) (each exposure is
prevented with probability `efficacy`) or [`AllOrNothingMode`](@ref) (a share
`efficacy` of vaccinated people are fully protected, the rest not at all).

For extension authors: a new mode defines
[`EpiBranch.realised_efficacy`](@ref) and
[`EpiBranch.realise_prior_dose!`](@ref); see the Extending guide.
"""
abstract type AbstractEffectMode end

"""Leaky vaccine (the default): each exposure of a vaccinated person is
prevented independently with probability `efficacy`."""
struct LeakyMode <: AbstractEffectMode end

"""All-or-nothing vaccine: each vaccinated person is, with probability
`efficacy`, fully protected against infection at every exposure from their
immunity date on, and otherwise not protected against infection at all. The
dose's other effects, such as `severity_efficacy`, apply to everyone
vaccinated."""
struct AllOrNothingMode <: AbstractEffectMode end

"""
    supports_waning(mode::AbstractEffectMode) -> Bool

Whether `waning` can be used with this vaccine `mode`. Waning reduces the
per-exposure protection of a leaky vaccine; an all-or-nothing vaccine has no
such protection to reduce, so `AllOrNothingMode` returns `false` and
[`VaccineEffect`](@ref) rejects `waning` with it. Default: `true`. A new
all-or-nothing style mode returns `false` to get the same check.
"""
supports_waning(::AbstractEffectMode) = true
supports_waning(::AllOrNothingMode) = false

"""
    VaccineEffect(; efficacy, severity_efficacy = 0.0, delay_to_immunity = 0.0,
        waning = nothing, mode = LeakyMode(), dose_label = :default)

What a vaccine dose does once given, whoever receives it and whenever: the
effects shared by every vaccination intervention. The built-in vaccinations
take these as keywords and build the `VaccineEffect` themselves; build one
directly only when writing a new vaccination intervention (see the Extending
guide).

- `efficacy`: for a leaky vaccine, the probability that each exposure is
  prevented once immune; for an all-or-nothing vaccine, the probability of
  being fully protected.
- `severity_efficacy`: probability that the vaccinated person's own illness
  is milder once immune (default 0).
- `delay_to_immunity`: days from vaccination to immunity (default 0).
- `waning`: a function of days since immunity giving the fraction of
  protection left, or `nothing` (default) for constant protection. Not yet
  available with `mode = AllOrNothingMode()`.
- `mode`: [`LeakyMode`](@ref) (default) or [`AllOrNothingMode`](@ref).
- `dose_label`: name of the dose, so several doses can be told apart
  (default `:default`).

The first three each take a number, a distribution or a function
`(rng, ind) -> ...`, drawn once per vaccinated person when the dose is given.
See [`AbstractVaccination`](@ref) for how the effects combine.
"""
struct VaccineEffect{E, SV, D, W, M <: AbstractEffectMode}
    efficacy::E
    severity_efficacy::SV
    delay_to_immunity::D
    waning::W
    mode::M
    dose_label::Symbol
end

function VaccineEffect(;
        efficacy, severity_efficacy = 0.0, delay_to_immunity = 0.0,
        waning = nothing, mode = LeakyMode(), dose_label = :default
    )
    # `waning` decays a per-exposure block, which a mode without one (a
    # responder is blocked with certainty from immunity onward, not at a
    # strength that fades) has nothing to apply to. Reject the combination
    # rather than silently ignoring `waning`.
    waning === nothing || supports_waning(mode) ||
        throw(
        ArgumentError(
            "`waning` is not yet supported together with `mode = $(mode)`. " *
                "Use `LeakyMode`, or another mode with " *
                "`supports_waning(mode) == true`, or drop `waning`."
        )
    )
    return VaccineEffect(
        efficacy, severity_efficacy, delay_to_immunity, waning, mode, dose_label
    )
end

"""
    vaccine_effect(v::AbstractVaccination) -> VaccineEffect

The [`VaccineEffect`](@ref) of a vaccination: what its dose does. Every
vaccination intervention defines this, and the package reads efficacy, delay,
mode and dose label through it.
"""
function vaccine_effect end

"""Efficacy of the vaccination's dose against infection, as given to
[`VaccineEffect`](@ref): per exposure for a leaky vaccine, the probability of
full protection for an all-or-nothing one."""
efficacy(v::AbstractVaccination) = vaccine_effect(v).efficacy

"""Efficacy of the vaccination's dose against severe disease, as given to
[`VaccineEffect`](@ref). The method for an individual returns the value drawn
for that person."""
severity_efficacy(v::AbstractVaccination) = vaccine_effect(v).severity_efficacy

"""Days from vaccination to immunity, as given to [`VaccineEffect`](@ref): a
number, a distribution or a function. A distribution or function is drawn once
per person when the dose is given."""
delay_to_immunity(v::AbstractVaccination) = vaccine_effect(v).delay_to_immunity

"""Whether the vaccination is leaky or all-or-nothing (its
[`AbstractEffectMode`](@ref))."""
effect_mode(v::AbstractVaccination) = vaccine_effect(v).mode
"""The dose's waning function (fraction of protection left `dt` days after
immunity), or `nothing` if protection does not wane. See
[`AbstractVaccination`](@ref)."""
waning(v::AbstractVaccination) = vaccine_effect(v).waning

"""Name of the dose a person must already have received before this
vaccination is given, or `nothing` if no earlier dose is needed."""
required_dose(::AbstractVaccination) = nothing

"""Name of the dose (for example `:prime` or `:boost`), so several doses can
be told apart. Each person's record of the dose is stored under keys that include the name (`:vaccinated_prime`, `:vaccination_time_prime` and so on); the
default `:default` uses the plain keys (`:vaccinated`, `:vaccination_time`,
`:vaccine_efficacy`, `:immunity_time`, `:severity_efficacy`)."""
dose_label(v::AbstractVaccination) = vaccine_effect(v).dose_label

# The built-in vaccinations expose the effect parameters as properties
# (`rv.efficacy`) next to their own fields, matching the keywords their
# constructors take, and `show` prints those keywords. Both are derived from
# the fields, so a parameter added to `VaccineEffect` or to one vaccination
# type appears without further edits. The keyword constructors pass their
# effect keywords on to `VaccineEffect` for the same reason, checking them
# first, which reports a misspelt keyword against the vaccination the caller
# named.
const _VACCINE_EFFECT_FIELDS = fieldnames(VaccineEffect)

function _effect_getproperty(v, name::Symbol)
    if name in _VACCINE_EFFECT_FIELDS
        return getfield(getfield(v, :effect), name)
    end
    return getfield(v, name)
end

_effect_propertynames(v) = (fieldnames(typeof(v))..., _VACCINE_EFFECT_FIELDS...)

function _show_keywords(io::IO, v)
    own = filter(!=(:effect), fieldnames(typeof(v)))
    print(io, nameof(typeof(v)), "(")
    for (i, name) in enumerate((_VACCINE_EFFECT_FIELDS..., own...))
        i > 1 && print(io, ", ")
        print(io, name, " = ")
        show(io, getproperty(v, name))
    end
    return print(io, ")")
end

# The effect keywords a vaccination's own constructor does not name are passed
# on to `VaccineEffect`, whose error for an unknown keyword would name a type
# the caller never wrote. Checking them here reports the vaccination and the
# keywords it takes instead.
function _check_effect_keywords(T, own, effect)
    for name in keys(effect)
        name in _VACCINE_EFFECT_FIELDS && continue
        throw(
            ArgumentError(
                "$T has no keyword argument `$name`. It takes " *
                    join(
                    string.("`", (own..., _VACCINE_EFFECT_FIELDS...), "`"),
                    ", "
                ) * "."
            )
        )
    end
    return nothing
end

function _vaccinated_key(label::Symbol)
    return label === :default ? :vaccinated : Symbol("vaccinated_", label)
end
function _vaccination_time_key(label::Symbol)
    return label === :default ? :vaccination_time : Symbol("vaccination_time_", label)
end
function _vaccine_efficacy_key(label::Symbol)
    return label === :default ? :vaccine_efficacy : Symbol("vaccine_efficacy_", label)
end
function _post_exposure_efficacy_key(label::Symbol)
    return label === :default ? :post_exposure_efficacy : Symbol("post_exposure_efficacy_", label)
end
function _onward_efficacy_key(label::Symbol)
    return label === :default ? :onward_efficacy : Symbol("onward_efficacy_", label)
end
function _coverage_declined_key(label::Symbol)
    return label === :default ? :coverage_declined : Symbol("coverage_declined_", label)
end
function _immunity_time_key(label::Symbol)
    return label === :default ? :immunity_time : Symbol("immunity_time_", label)
end
function _severity_efficacy_key(label::Symbol)
    return label === :default ? :severity_efficacy : Symbol("severity_efficacy_", label)
end

# Time dose `label` was given to `ind`, or `nothing` if it has not been.
function _dose_time(label::Symbol, ind)
    get(ind.state, _vaccinated_key(label), false) || return nothing
    vacc_t = get(ind.state, _vaccination_time_key(label), Inf)
    return isfinite(vacc_t) ? vacc_t : nothing
end

"""Full-strength efficacy of dose `v` against infection of `ind`, as sampled
when the dose was given, or `nothing` if none was recorded."""
function _vaccine_efficacy(v::AbstractVaccination, ind)
    return get(ind.state, _vaccine_efficacy_key(dose_label(v)), nothing)
end

# Fraction of a dose's efficacy still in force `dt` after immunity develops.
_retained(::Nothing, dt) = 1.0
_retained(w, dt) = w(dt)

# Block probability of a dose whose immunity develops at `imm_t`, given as
# `block(retained)`, a function of the fraction of efficacy retained. Without
# waning the fraction stays at 1 and the block is a fixed number. With waning it
# is read at the exposure under evaluation: the engine sets
# `contact.infection_time` to that transmission time before resolving competing
# risks (`_resolve!`), so the closure reads it fresh for every edge.
_waned_block(block, ::Nothing, imm_t) = block(1.0)
function _waned_block(block, w, imm_t)
    return (rng, parent, contact, state) -> block(w(contact.infection_time - imm_t))
end

"""Time at which `ind`'s immunity from dose `v`, given at `vacc_t`, develops.
A scalar `delay_to_immunity` is the same for everyone and is added to
`vacc_t` directly; a varying one was drawn when the dose was given and is
read back from the stored `:immunity_time`, or `Inf` if nothing was stored."""
function _immunity_time(v::AbstractVaccination, ind, vacc_t)
    return _immunity_time(delay_to_immunity(v), dose_label(v), ind, vacc_t)
end
_immunity_time(delay::Real, label, ind, vacc_t) = vacc_t + delay
function _immunity_time(delay, label, ind, vacc_t)
    return get(ind.state, _immunity_time_key(label), Inf)
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

"""Record a person as not yet vaccinated with this dose, unless the model's
`attributes` already gave them the dose (for example from a campaign before the
simulation starts), in which case that dose and its date are kept.

For an all-or-nothing vaccine, such an earlier dose's efficacy is turned into
full or no protection here, once, as for a dose given during the simulation.
A recorded efficacy of 0 or 1 is kept as it is."""
function initialise_individual!(v::AbstractVaccination, individual, state)
    label = dose_label(v)
    get!(individual.state, _vaccinated_key(label), false)
    get!(individual.state, _vaccination_time_key(label), Inf)
    realise_prior_dose!(effect_mode(v), individual, label, state)
    return nothing
end

"""
    realise_prior_dose!(mode::AbstractEffectMode, individual, label, state)

For a dose the model's `attributes` gave a person before the simulation
started, apply [`realised_efficacy`](@ref EpiBranch.realised_efficacy) to its
recorded efficacy, so it gets the same per-person draw as a dose given during
the simulation. Changes `individual.state` in place.

A new mode usually needs no method of its own: the default applies the mode's
`realised_efficacy` to the recorded efficacy, which is what
[`AllOrNothingMode`](@ref) needs. [`LeakyMode`](@ref) does nothing, since its
recorded efficacy is already the value it uses.
"""
function realise_prior_dose!(mode::AbstractEffectMode, individual, label, state)
    key = _vaccine_efficacy_key(label)
    eff = get(individual.state, key, nothing)
    (eff isa Real && 0 < eff < 1) || return nothing
    individual.state[key] = realised_efficacy(mode, eff, state.rng)
    return nothing
end
realise_prior_dose!(::LeakyMode, individual, label, state) = nothing

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
    vacc_t = _dose_time(dose_label(v), contact)
    vacc_t === nothing && return nothing
    eff = _vaccine_efficacy(v, contact)
    eff === nothing && return nothing
    # A zero-efficacy dose can never block, and the engine skips such a risk.
    # Returning nothing keeps it out of the returned tuple, so the recommended
    # post-exposure-only setup (`efficacy = 0.0`) does not build and discard a
    # risk for every contact.
    eff <= 0 && return nothing
    imm_t = _immunity_time(v, contact, vacc_t)
    return Risk(
        event_time = imm_t,
        block_probability = _waned_block(retained -> eff * retained, waning(v), imm_t)
    )
end

function competing_risk(v::AbstractVaccination, parent, contact, state)
    return _susceptibility_risk(v, contact)
end

# The protection above reads only the contact. A subtype with a risk of its own
# is taken to read the infector, as any other intervention is.
function risk_depends_on_infector(v::AbstractVaccination)
    return _has_own_method(competing_risk, typeof(v), AbstractVaccination)
end

"""
    realised_efficacy(mode::AbstractEffectMode, eff, rng) -> Real

Turn a vaccinated person's drawn efficacy `eff` into the probability that
each of their exposures is prevented once immune. Called once, when the dose
is given; the result applies at every exposure. [`LeakyMode`](@ref) keeps
`eff`. [`AllOrNothingMode`](@ref) draws once whether the person is protected,
giving `1.0` (fully protected) with probability `eff` and `0.0` (no
protection) otherwise. A new mode defines this method.
"""
realised_efficacy(::LeakyMode, eff, rng) = eff
realised_efficacy(::AllOrNothingMode, eff, rng) = float(rand(rng, Bernoulli(eff)))

# Helper for concrete subtypes: write per-dose state on a contact at
# vaccination time. Samples efficacy, severity efficacy, and the
# immunity delay via `_sample_value` so scalar, distribution, and
# function forms all work. The delay is drawn here, once, rather than
# inside `competing_risk`, so a given individual's immunity time is the
# same against every exposure it faces. Storing the resulting immunity
# time also lets a clinical transition check it without reaching for the
# vaccination object, which it never sees.
#
# `realised_efficacy` is where the two effect modes part ways (see its
# docstring); `rand(rng, Bernoulli(e))` there is what a StochasticAD pass
# differentiates without bias, so a continuous relaxation of that draw would
# reintroduce leaky semantics.
function _record_vaccination!(v::AbstractVaccination, contact, vacc_t, rng)
    label = dose_label(v)
    contact.state[_vaccinated_key(label)] = true
    contact.state[_vaccination_time_key(label)] = vacc_t
    contact.state[_vaccine_efficacy_key(label)] = realised_efficacy(
        effect_mode(v), _sample_value(efficacy(v), rng, contact), rng
    )
    contact.state[_immunity_time_key(label)] = vacc_t +
        _sample_value(
        delay_to_immunity(v), rng, contact
    )
    contact.state[_severity_efficacy_key(label)] = _sample_value(
        severity_efficacy(v), rng, contact
    )
    _record_effect_draws!(v, contact, label, rng)
    return nothing
end

# Draws for effects only some vaccinations have (`post_exposure_efficacy` and
# `onward_efficacy` exist only on `RingVaccination`).
_record_effect_draws!(::AbstractVaccination, contact, label, rng) = nothing

# ── RingVaccination ──────────────────────────────────────────────────

"""
    RingVaccination(; efficacy, coverage = 1.0, delay_to_immunity = 0.0,
                    post_exposure_efficacy = 0.0, onward_efficacy = 0.0,
                    severity_efficacy = 0.0, eligibility_window = Inf,
                    dose_delay = 0.0, requires_dose = nothing, waning = nothing,
                    mode = LeakyMode(), dose_label = :default)

Vaccinate the contacts found by [`ContactTracing`](@ref) (ring vaccination),
including post-exposure prophylaxis (PEP, as in
[pepbp](https://github.com/sophiemeakin/pepbp)). Needs `ContactTracing`
among the interventions, listed first.

# Examples
```julia
# Ring vaccination of contacts and contacts of contacts
[ContactTracing(OnSymptomOnset(), 0.8, Exponential(1.0), Quarantine(duration = 7.0); depth = 2),
 RingVaccination(efficacy = 0.9, delay_to_immunity = 10.0, coverage = 0.8)]

# Post-exposure prophylaxis: a dose that takes effect before onset
# stops the infection with probability 0.6
RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.6)
```

# Arguments
- `efficacy`: probability that each later exposure of the vaccinated contact
  is prevented once immune (see [`AbstractVaccination`](@ref) for leaky and
  all-or-nothing vaccines).
- `coverage`: probability that a traced contact is actually vaccinated
  (consent, absence, exclusion criteria, logistics). Default 1. A number, a
  distribution or a function `(rng, contact) -> ...`, for example age
  dependent, or reading [`vaccine_acceptance`](@ref) to cluster refusal by
  group.
- `delay_to_immunity`: days from vaccination to immunity. Default 0, as for
  PEP; set it for a vaccine that takes time to protect.
- `post_exposure_efficacy`: probability that the dose stops an infection the
  contact already has (default 0). Needs `incubation_period` from
  [`clinical_presentation`](@ref).
- `onward_efficacy`: probability that each onward transmission by the
  vaccinated contact is prevented once immune (default 0).
- `severity_efficacy`: probability that the vaccinated contact's own illness is
  milder once immune (default 0). See "Severity" below.
- `eligibility_window`: days after exposure beyond which a contact is no longer
  vaccinated (default `Inf`), as in filovirus protocols where vaccination
  more than about 21 days after exposure is pointless. A number or a function
  `(rng, contact) -> ...`.
- `dose_delay`, `requires_dose`: for second and later doses (below).
- `waning`, `mode`, `dose_label`: see [`AbstractVaccination`](@ref).

`delay_to_immunity`, `post_exposure_efficacy`, `onward_efficacy` and
`severity_efficacy` take the same forms as `efficacy` (a number, a
distribution or a function `(rng, ind) -> ...`), drawn once per contact when
the dose is given. `dose_delay` is drawn once when the dose is scheduled.

The window is checked on the day of vaccination, so a contact vaccinated just
inside it with a long `delay_to_immunity` is still vaccinated, and whether
immunity comes in time is then decided at each exposure.

!!! warning "The window is measured to *this* dose, `dose_delay` included"
    A second dose is given `dose_delay` days after the trace, and the window
    is checked against that day. Copying a first dose's
    `eligibility_window` onto a booster therefore rejects almost every
    booster, since `dose_delay` alone usually exceeds the window. Leave a
    later dose at the default `Inf` unless the protocol has a deadline on
    that dose itself.

# Post-exposure and onward effects

`post_exposure_efficacy` can stop an infection when immunity (vaccination
plus `delay_to_immunity`) arrives after the exposure but before the contact's
symptom onset. A stopped infection transmits as usual until immunity arrives
and not at all afterwards. It never has symptoms, so nothing triggered by
onset happens (isolation, tracing, onset-timed transitions), and no
transition in the `progression` takes effect at or after that time, so there
is no later hospitalisation, death or other outcome; transitions before it
stand. A stopped infection is still a case: it is counted by
[`chain_statistics`](@ref) and listed by [`linelist`](@ref) with no onset date
and a `date_infection_aborted` column. Where immunity is already in place at
the exposure, the dose instead prevents the infection with the same
probability. That is all it does for a contact who never has symptoms, so such
a contact exposed before immunity is not protected. `post_exposure_efficacy`
therefore already covers every exposure `efficacy` would; set one or the
other, since setting both counts the protection twice.

`onward_efficacy` leaves the contact's infection and illness in place but
prevents each of their transmissions after immunity with this probability,
whenever their onset is. It therefore also reduces onward cases from contacts
already infected when vaccinated. `post_exposure_efficacy` and
`onward_efficacy` act independently, giving a vaccine that stops some
infections and reduces the infectiousness of the rest. Setting `efficacy` and
`onward_efficacy` to the same value gives a vaccine that acts equally on
susceptibility and infectiousness.

!!! warning "`efficacy` changes nothing when contacts are traced after their infector is isolated"
    By default `ContactTracing` reaches a contact once its infector has
    been isolated, and perfect `Isolation` already prevents every later
    transmission to that contact. A dose acting only through `efficacy`
    then leaves results unchanged, with or without quarantine
    (`action = FlagOnly()`). `efficacy` has infections to prevent
    only when a contact can still be infected after being traced: with
    leaky isolation (`post_isolation_transmission > 0`), when tracing
    starts before isolation (for example `eligibility = OnSymptomOnset()`),
    or in a `depth > 1` ring through people who keep transmitting after
    being traced. `onward_efficacy` acts on the contact's own later
    transmission, which quarantine already prevents, so it matters when
    tracing does not quarantine; `post_exposure_efficacy` acts on the
    infection the contact already has. Under quarantine,
    `post_exposure_efficacy` can lower containment, because a stopped
    infection never has symptoms and so no longer triggers tracing of the
    people it infected before the dose.

# Severity

`severity_efficacy` does not change transmission. It acts only where a
transition in the `progression` reads it, through the
[`severity_efficacy`](@ref) and [`immunity_time`](@ref) functions, for example
a lower case fatality ratio among vaccinated cases:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```

The check `immunity_time(ind) <= onset_time(ind)` gives protection only to a
person immune by their onset; checking `is_vaccinated` alone would count a
dose whose immunity had not yet developed.

`severity_efficacy` does not wane: `waning` applies only to the protection
against infection, so with `waning` set, protection against infection fades
while protection against severe outcomes stays. To make it fade too, apply
the decay in the transition, from immunity to onset:

```julia
decay(dt) = exp(-dt / 180)
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) -> begin
          dt = onset_time(ind) - immunity_time(ind)
          dt >= 0 ? 0.7 * (1 - severity_efficacy(ind) * decay(dt)) : 0.7
      end)
```

`dt >= 0` is the same immunity check, and is false for a person with no onset
(`NaN`), who keeps the unvaccinated probability. Passing `decay` as the
dose's `waning` too makes both effects fade at the same rate.

# Second and later doses

A dose is given at the trace, or `dose_delay` days after it.
`requires_dose` names a dose the contact must have received by the time this
dose is due; a contact without it does not get this dose. A prime-boost
schedule is two `RingVaccination`s:

```julia
[
    RingVaccination(efficacy = 0.6, delay_to_immunity = 21.0,
        coverage = 0.8, dose_label = :prime),
    RingVaccination(efficacy = 0.5, dose_delay = Uniform(28.0, 42.0),
        delay_to_immunity = 14.0, coverage = 0.9,
        requires_dose = :prime, dose_label = :boost),
]
```

The boost is given on average five weeks after the trace to 90% of those
primed (the other 10% lost to follow-up) and protects 14 days later. Doses
act independently, so reaching 80% protection in total from a 60% prime
needs `efficacy = 0.5` on the boost (`(0.8 - 0.6) / (1 - 0.6)`), the
protection it adds among those the prime left unprotected.

List a dose after the dose it requires: interventions are applied in order,
so a boost listed first would find no prime and never be given. The required
dose may come from any vaccination, such as a [`MassVaccination`](@ref)
prime; a contact whose prime falls after the boost is due is not boosted.
Between two ring doses, a boost `dose_delay` shorter than the prime's is
rejected when the [`ModelSpec`](@ref) is built, since the boost could never
be given. A `dose_delay` drawn from a distribution is rejected when even its
longest delay falls before the prime's shortest, and gives a warning when the
two ranges overlap, because contacts whose draws come out in the wrong order
go without the boost. A function, or a distribution without a known range,
cannot be checked in advance; a dose whose draw falls before the required dose
is then skipped for that contact.

A dose is recorded when it is due, whether or not the contact was infected
in the meantime, so counts of later doses are counts of doses scheduled.

`waning` scales `efficacy`, `post_exposure_efficacy` and `onward_efficacy` by
the time since this contact became immune. Post-exposure protection acts the
moment immunity arrives and uses `waning(0)`, so a `waning` that builds up
first, such as `dt -> min(1, dt / 14)`, stops no infections.

For extension authors: each contact's dose is recorded in `ind.state` under
`:vaccinated`, `:vaccination_time`, `:vaccine_efficacy`, `:immunity_time` and
`:severity_efficacy` (with the `dose_label` as a suffix for a non-default
label), plus `:post_exposure_efficacy` and `:onward_efficacy` when those vary
between people.
"""
struct RingVaccination{V <: VaccineEffect, C, DD, W, PE, OE} <: AbstractVaccination
    effect::V
    coverage::C
    dose_delay::DD
    requires_dose::Union{Nothing, Symbol}
    eligibility_window::W
    post_exposure_efficacy::PE
    onward_efficacy::OE
end

function RingVaccination(;
        coverage = 1.0, dose_delay = 0.0, requires_dose = nothing,
        eligibility_window = Inf, post_exposure_efficacy = 0.0, onward_efficacy = 0.0,
        effect...
    )
    _check_effect_keywords(
        RingVaccination,
        (
            :coverage, :dose_delay, :requires_dose, :eligibility_window,
            :post_exposure_efficacy, :onward_efficacy,
        ), effect
    )
    return RingVaccination(
        VaccineEffect(; effect...), coverage, dose_delay,
        requires_dose, eligibility_window, post_exposure_efficacy, onward_efficacy
    )
end

vaccine_effect(rv::RingVaccination) = getfield(rv, :effect)
Base.getproperty(rv::RingVaccination, name::Symbol) = _effect_getproperty(rv, name)
Base.propertynames(rv::RingVaccination, ::Bool = false) = _effect_propertynames(rv)
Base.show(io::IO, rv::RingVaccination) = _show_keywords(io, rv)

function required_fields(rv::RingVaccination)
    return _maybe_positive(rv.post_exposure_efficacy) ? [:traced, :incubation_period] : [:traced]
end
required_dose(rv::RingVaccination) = rv.requires_dose

# Full-strength post-exposure and onward efficacies of dose `rv` for `ind`: the
# scalar off the intervention, or the draw stored when the dose was given. A
# contact with no dose of this vaccination has no draw stored and reads zero.
function _post_exposure_efficacy(rv::RingVaccination, ind)
    return _dose_value(
        rv.post_exposure_efficacy, _post_exposure_efficacy_key,
        dose_label(rv), ind
    )
end
function _onward_efficacy(rv::RingVaccination, ind)
    return _dose_value(rv.onward_efficacy, _onward_efficacy_key, dose_label(rv), ind)
end

function _record_effect_draws!(rv::RingVaccination, contact, label, rng)
    _store_draw!(
        rv.post_exposure_efficacy, _post_exposure_efficacy_key, label, contact,
        rng
    )
    _store_draw!(rv.onward_efficacy, _onward_efficacy_key, label, contact, rng)
    return nothing
end

# Onward-infectiousness risk: blocks the parent → contact transmission
# iff this dose has been administered to the *parent* and the parent's
# immunity has developed by their (the parent's) transmission time. The
# parent's `:vaccination_time` is set by ring vaccination when the parent
# was traced, and the onward immunity takes effect at that time plus
# `delay_to_immunity`, as on the susceptibility side.
_onward_risk(rv::RingVaccination, parent) = _onward_risk(rv, parent, nothing)

function _onward_risk(rv::RingVaccination, parent, contact)
    # A community introduction has no infector within the population, so
    # vaccination cannot reduce its source's onward transmission.
    parent === contact && return nothing
    onward = _onward_efficacy(rv, parent)
    onward > 0.0 || return nothing
    vacc_t = _dose_time(dose_label(rv), parent)
    vacc_t === nothing && return nothing
    imm_t = _immunity_time(rv, parent, vacc_t)
    return Risk(
        event_time = imm_t,
        block_probability = _waned_block(retained -> onward * retained, waning(rv), imm_t)
    )
end

# Contact-side risk. `efficacy` blocks an exposure that comes after immunity,
# and so does `post_exposure_efficacy`: a dose whose immunity is already in
# place when the contact is exposed prevents the infection outright, the limit
# of aborting it the moment it starts. This is also all a dose can do for a
# contact with no onset to race (asymptomatic). Both act at the same immunity
# time, so they combine into one block, drawn once, and `waning` scales both by
# the same fraction.
function _contact_risk(rv::RingVaccination, contact)
    post = _post_exposure_efficacy(rv, contact)
    post > 0.0 || return _susceptibility_risk(rv, contact)
    vacc_t = _dose_time(dose_label(rv), contact)
    vacc_t === nothing && return nothing
    eff = something(_vaccine_efficacy(rv, contact), 0.0)
    imm_t = _immunity_time(rv, contact, vacc_t)
    block(retained) = 1 - (1 - eff * retained) * (1 - post * retained)
    return Risk(
        event_time = imm_t,
        block_probability = _waned_block(block, waning(rv), imm_t)
    )
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
# out not to be the infection, the engine discards the abort (see
# `abort_infection!`), and a contact that escaped it gets a fresh draw at its
# next exposure. `action_draw!` caches the draw against this exposure, so a
# continuous-time race, which settles a contact's infection before checking a
# dose already recorded against it and can then reconsider the same contact at
# the same exposure through its usual candidate discovery, does not draw twice
# for one exposure.
#
# Callers check `_maybe_positive(rv.post_exposure_efficacy)` first: the
# vaccination time comes untyped from the contact's state, so an
# unconditional call would dispatch dynamically for every vaccinated
# contact of a dose that cannot abort.
function _abort_infection!(rv::RingVaccination, contact, vacc_t, rng)
    incubation = get(contact.state, :incubation_period, NaN)
    isnan(incubation) && return nothing
    post = _post_exposure_efficacy(rv, contact)
    post > 0.0 || return nothing
    immunity = _immunity_time(rv, contact, vacc_t)
    exposure = contact.infection_time
    exposure < immunity < exposure + incubation || return nothing
    # The abort acts the moment immunity arrives and takes the protection the
    # dose retains then: the block `_contact_risk` applies to an exposure
    # coinciding with immunity.
    post *= _retained(waning(rv), 0.0)
    # Waning can take the retained efficacy to zero, and a dose that cannot
    # abort anything must not draw: the draw would never succeed and would still
    # move every later draw in the run.
    post > 0.0 || return nothing
    covered = action_draw!(contact, (rv, :abort, exposure)) do
        _covers(post, contact, rng)
    end
    covered || return nothing
    abort_infection!(contact, immunity)
    return nothing
end

# A dose given to a pending, still-uninfected member of a continuous-time race
# (household, network, routed network) is recorded on it before its own
# infection is settled, so `_abort_infection!` finds nothing to check against
# yet (`contact.infection_time` is still `NaN`) and does not draw. Once the
# race settles that member's infection, this reconsiders the dose against the
# now-final exposure, which suppresses onset and the clinical transitions
# exactly as a dose given after the exposure does on the generation engine.
#
# The protection derives from the recorded dose rather than from a schedule's
# clock, as `persistent_competing_risks` says, so a `Scheduled` wrapper that
# has since switched off does not withdraw it.
function on_infection_settled!(rv::RingVaccination, ind, state, rng)
    _maybe_positive(rv.post_exposure_efficacy) || return nothing
    vacc_t = _dose_time(dose_label(rv), ind)
    vacc_t === nothing && return nothing
    _abort_infection!(rv, ind, vacc_t, rng)
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

# Only the onward risk reads the infector.
risk_depends_on_infector(rv::RingVaccination) = rv.onward_efficacy > 0

# A ring doses the infector's own traced contacts: delivery depends on the
# individual being resolved. A shared budget wrapped around one is
# `CapacityConstrained`'s own declaration.
reads_population_state(::RingVaccination) = false

# Scalar defaults short-circuit without drawing from the rng so that
# coverage = 1.0 and eligibility_window = Inf reproduce the previous
# deterministic behaviour exactly.
_within_eligibility_window(w::Real, ind, vacc_t, rng) = _within_window(w, ind, vacc_t)
function _within_eligibility_window(w, ind, vacc_t, rng)
    return _within_window(_sample_value(w, rng, ind), ind, vacc_t)
end

# A contact with no exposure yet (a `NaN` infection time) has not exceeded any
# window, so a pre-exposure dose is always within it.
function _within_window(w, ind, vacc_t)
    return isnan(ind.infection_time) || vacc_t - ind.infection_time <= w
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
    return _has_required_dose(required_dose(v), ind, vacc_t)
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
            haskey(given, req) || throw(
                ArgumentError(
                    "vaccination with dose_label = :$label requires dose :$req, " *
                        "which is not given earlier in the intervention stack. " *
                        "List a dose after the dose it requires."
                )
            )
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
    hi < req_lo && throw(
        ArgumentError(
            "vaccination with dose_label = :$label requires dose :$req but is " *
                "scheduled earlier than it (a dose_delay of at most $hi days after the " *
                "trace, against at least $req_lo for dose :$req). A dose cannot be " *
                "given before the dose it requires."
        )
    )
    lo < req_hi && @warn "This dose's dose_delay $(_reaches_below(lo)), so it can " *
        "fall before dose :$req, which it requires and whose own dose_delay " *
        "$(_reaches_above(req_hi)). Contacts whose draws come out in that " *
        "order go without this dose." dose_label = label
    return nothing
end

# An unbounded support has no number worth quoting, so the warning describes it
# in words instead.
function _reaches_below(lo)
    return isfinite(lo) ? "reaches $lo days after the trace" : "has no lower bound"
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
    efficacy(rv) isa Real && efficacy(rv) > 0.0 || return nothing
    @warn "RingVaccination sets both `efficacy` and `post_exposure_efficacy`, " *
        "which compose as independent risks and so over-protect any contact " *
        "vaccinated before its exposure. `post_exposure_efficacy` already " *
        "covers those contacts; set one or the other." dose_label = dose_label(rv) maxlog = 1
    return nothing
end

"""Days from the trace to this dose, as the `dose_delay` was given, or `nothing`
for a vaccination not timed from the trace. [`RingVaccination`](@ref) is the only
one timed from it."""
_dose_offset(::AbstractVaccination) = nothing
_dose_offset(rv::RingVaccination) = rv.dose_delay

# The intervention inside a wrapper; `InterventionWrapper` adds a method.
_unwrap_scheduled(iv) = iv

function apply_post_transmission!(rv::RingVaccination, state, new_contacts)
    return apply_actions!(rv, state, new_contacts)
end

# ── GroupVaccination ─────────────────────────────────────────────────

"""
    GroupVaccination(; efficacy, eligibility = OnLabConfirmation(),
                     coverage = 1.0, dose_delay = 0.0, group_key = :group,
                     severity_efficacy = 0.0, delay_to_immunity = 0.0,
                     waning = nothing, mode = LeakyMode(), dose_label = :default)

Vaccinate everyone in a case's group (a community, health area or household)
once a case there is confirmed or suspected: the usual fallback when no ring
of contacts can be traced. Every member is vaccinated, whether or not they
have any traced link to the case. Groups come from the [`groups`](@ref)
population characteristic (stored under `group_key`, default `:group`).

# Examples
```julia
# Vaccinate the whole community 2 days after a case there is lab-confirmed
GroupVaccination(efficacy = 0.7, eligibility = OnLabConfirmation(), dose_delay = 2.0)
```

# Arguments
- `eligibility`: which cases trigger vaccination of their group, using the
  same rules as [`ContactTracing`](@ref) but applied to the case:
  `OnLabConfirmation()` (default) once a case has tested positive,
  `OnSymptomOnset()` on suspicion alone, combined with `&`, `|` and `!`. The
  rule's [`trigger_time`](@ref EpiBranch.trigger_time) sets when the group
  counts as having a case.
- `dose_delay`: days from that trigger to vaccination, for example the time a
  vaccination team needs to reach the group. A number, a distribution or a
  function, drawn once per member, so members can be reached on different
  days.
- `coverage`, `efficacy`, `severity_efficacy`, `delay_to_immunity`, `waning`,
  `mode`, `dose_label`: as for [`RingVaccination`](@ref). Pairing `coverage`
  with [`vaccine_acceptance`](@ref) on the same `group_key` clusters refusal by
  group. `severity_efficacy` acts only through a clinical transition that
  reads it and does not wane.

A group is vaccinated at the trigger time of its *earliest* eligible case;
later cases in the same group do not move the date. Members who appear later
(a case infected later in the same group) are vaccinated at that same date
plus `dose_delay`, so they are protected only from exposures after that, as
for members already present. Doses grow with group size, where
[`RingVaccination`](@ref) doses grow with ring size.

# Group vaccination as a fallback to ring vaccination

Listing a [`RingVaccination`](@ref) before a `GroupVaccination` with the
same `dose_label` makes the group dose a fallback: a member the ring has
already vaccinated is skipped, so the group dose reaches only those the ring
did not:

```julia
[ContactTracing(OnLabConfirmation(), 0.7, Exponential(1.0), Quarantine(duration = 7.0)),
 RingVaccination(efficacy = 0.9),
 GroupVaccination(efficacy = 0.6)]
```

Needs the group characteristic and whatever `eligibility` needs, such as the
test result from [`Isolation`](@ref) for the default `OnLabConfirmation()`.

!!! note "Index cases in a branching process"
    Like [`RingVaccination`](@ref) and [`MassVaccination`](@ref),
    `GroupVaccination` acts on new contacts as they are made. In a
    branching process the index cases are not contacts, so an index case
    is vaccinated only once another member of its group appears later; an
    index case whose chain dies out without anyone else in its group being
    infected is not vaccinated, even if it is itself confirmed.

In network and household models, a group is checked after each case's
infection is settled, index cases included, and the dose can reach that case
and members not yet infected, subject to scheduling and capacity. Cases there
are settled in order of infection, not of eligibility, so a member can turn
out eligible earlier than the person who infected them. A member whose
infection is not yet settled has their dose moved earlier if an earlier
trigger is found later; a member already settled keeps their dose date, as
does a dose another vaccination already gave.

For a campaign with repeat visits, `coverage` is the final probability of being
reached, and `dose_delay` can describe the first successful visit. A person
not covered is not marked as having refused: record willingness as a
population characteristic when that distinction matters. See
[Repeat campaign visits](@ref) for an example using these inputs.
"""
struct GroupVaccination{V <: VaccineEffect, E <: TraceEligibility, C, DD} <:
    AbstractVaccination
    effect::V
    eligibility::E
    coverage::C
    dose_delay::DD
    group_key::Symbol
end

function GroupVaccination(;
        eligibility = OnLabConfirmation(), coverage = 1.0,
        dose_delay = 0.0, group_key = :group, effect...
    )
    _check_effect_keywords(
        GroupVaccination,
        (:eligibility, :coverage, :dose_delay, :group_key), effect
    )
    return GroupVaccination(
        VaccineEffect(; effect...), eligibility, coverage,
        dose_delay, group_key
    )
end

vaccine_effect(gv::GroupVaccination) = getfield(gv, :effect)
Base.getproperty(gv::GroupVaccination, name::Symbol) = _effect_getproperty(gv, name)
Base.propertynames(gv::GroupVaccination, ::Bool = false) = _effect_propertynames(gv)
Base.show(io::IO, gv::GroupVaccination) = _show_keywords(io, gv)

function required_fields(gv::GroupVaccination)
    return union([gv.group_key], required_fields(gv.eligibility))
end

# A group's members under `key`, from an index this intervention keeps in the
# run's `scratch` and grows incrementally as `state.individuals` grows: only
# the tail added since the last query is scanned, rather than every individual
# on every call. Correct because membership is set once, at creation, and never
# changes afterwards (see `groups`/`group_attribute`), so an id already indexed
# never needs revisiting.
function _group_members(state::SimulationState, key::Symbol, group)
    seen, index = get!(state.scratch, (:group_members, key)) do
        (Ref(0), Dict{Any, Vector{Int}}())
    end::Tuple{Base.RefValue{Int}, Dict{Any, Vector{Int}}}
    n = length(state.individuals)
    for id in (seen[] + 1):n
        g = get(state.individuals[id].state, key, nothing)
        g === nothing && continue
        push!(get!(index, g, Int[]), id)
    end
    seen[] = n
    return get(index, group, Int[])
end

# The group's trigger time: the earliest time any of its members (found
# through the group-to-members index, not just this generation's
# `new_contacts`) meets `eligibility`, tested against the member itself in
# both the infector and contact slots since the policy describes a property
# of a case, not a pair. `Inf` if no member has triggered yet.
# A group's trigger is the earliest eligible time among every member
# under the group key, wherever they live. The trigger a race sees
# therefore depends on which cliques have raced already.
reads_population_state(::GroupVaccination) = true

function _group_trigger_time(gv::GroupVaccination, state, group)
    key = gv.group_key
    t = Inf
    for id in _group_members(state, key, group)
        m = state.individuals[id]
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
# visiting that group's members (via the index) against it handles both in
# one pass: a fresh trigger reaches members already present, and a standing
# one reaches a member only now created. Groups untouched this generation are
# left alone, so nobody outside a triggered group is ever visited.
function apply_post_transmission!(gv::GroupVaccination, state, new_contacts)
    return apply_actions!(gv, state, new_contacts)
end

# ── MassVaccination ──────────────────────────────────────────────────

"""
    MassVaccination(; efficacy, eligibility_time, severity_efficacy = 0.0,
                    delay_to_immunity = 0.0, waning = nothing,
                    mode = LeakyMode(), dose_label = :default)

Vaccinate the population on a schedule, independently of contact tracing.

# Arguments
- `eligibility_time`: the day each person is vaccinated; `Inf` means never. A
  number (everyone on that day), a distribution such as `Exponential(60.0)`
  for a slow random rollout (drawn per person), or a function
  `(rng, ind) -> ...` for an age-stratified or other rule. Reading
  [`vaccine_acceptance`](@ref) here clusters refusal by group: for example
  `(rng, ind) -> ind.state[:vaccine_acceptance] > 0.5 ? 30.0 : Inf` leaves out
  whole groups whose acceptance is at most 0.5, so uptake is
  `P(acceptance > 0.5)`.
- `efficacy`, `delay_to_immunity`, `severity_efficacy`: take the same forms,
  drawn once per vaccinated person, for example an age-dependent efficacy or
  immunity taking one to three weeks. See [`AbstractVaccination`](@ref).
- `waning`: decline of protection against infection from the day of immunity
  (see [`AbstractVaccination`](@ref)); matters most when people are vaccinated
  long before exposure. It does not apply to `severity_efficacy`, which acts
  only through a clinical transition that reads it, as for
  [`RingVaccination`](@ref).
- `mode`, `dose_label`: see [`AbstractVaccination`](@ref). Use two
  `MassVaccination`s with different `dose_label`s for a multi-dose rollout.

Each contact's vaccination day is drawn when a branching process creates
the contact (index cases are not vaccinated), and they count as vaccinated
if it is finite. They are protected against an exposure only if their
vaccination day plus `delay_to_immunity` comes before it.

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

Prime-and-boost schedule (two doses with different labels):

```julia
[
    MassVaccination(efficacy = 0.6,  eligibility_time = 30.0,
        delay_to_immunity = 14.0, dose_label = :prime),
    MassVaccination(efficacy = 0.9,  eligibility_time = 60.0,
        delay_to_immunity = 14.0, dose_label = :boost),
]
```
"""
struct MassVaccination{V <: VaccineEffect, T} <: AbstractVaccination
    effect::V
    eligibility_time::T
end

function MassVaccination(; eligibility_time, effect...)
    _check_effect_keywords(MassVaccination, (:eligibility_time,), effect)
    return MassVaccination(VaccineEffect(; effect...), eligibility_time)
end

vaccine_effect(mv::MassVaccination) = getfield(mv, :effect)

# Each contact's eligibility time is drawn when the contact is created:
# delivery depends on the individual being resolved.
reads_population_state(::MassVaccination) = false
Base.getproperty(mv::MassVaccination, name::Symbol) = _effect_getproperty(mv, name)
Base.propertynames(mv::MassVaccination, ::Bool = false) = _effect_propertynames(mv)
Base.show(io::IO, mv::MassVaccination) = _show_keywords(io, mv)

required_fields(::MassVaccination) = Symbol[]

function apply_post_transmission!(mv::MassVaccination, state, new_contacts)
    return apply_actions!(mv, state, new_contacts)
end
