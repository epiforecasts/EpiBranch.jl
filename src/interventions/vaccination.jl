"""
Base type for vaccination interventions. A vaccination has an
`efficacy` (per-exposure block probability once immunity is in place)
and a `delay_to_immunity` (time between vaccination and protection).

Concrete subtypes differ only in eligibility — who gets vaccinated
when. They share the [`competing_risk`](@ref) machinery: a vaccinated
contact whose immunity has developed by their transmission time has
their infection blocked with probability `efficacy`.

`mode` (`:leaky` or `:all_or_nothing`) is recorded for completeness
but does not change the per-contact infection probability in a pure
branching tree, where every contact is a unique exposure. The
distinction starts to matter once network models permit multiple
exposures per individual.
"""
abstract type AbstractVaccination <: AbstractIntervention end

efficacy(v::AbstractVaccination) = v.efficacy
delay_to_immunity(v::AbstractVaccination) = v.delay_to_immunity

"""Vaccination's competing risk: blocks the parent → contact
transmission iff the contact has been vaccinated and immunity has
developed by their transmission time."""
function competing_risk(v::AbstractVaccination, parent, contact, state)
    is_vaccinated(contact) || return nothing
    vacc_t = get(contact.state, :vaccination_time, Inf)::Float64
    isfinite(vacc_t) || return nothing
    return Risk(event_time = vacc_t + delay_to_immunity(v),
        block_probability = efficacy(v))
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

Requires `:traced` (set by [`ContactTracing`](@ref)).

Initialises: `:vaccinated`, `:vaccination_time`.
"""
Base.@kwdef struct RingVaccination <: AbstractVaccination
    efficacy::Float64
    delay_to_immunity::Float64 = 0.0
    mode::Symbol = :leaky
end

required_fields(::RingVaccination) = [:traced]

function initialise_individual!(::RingVaccination, individual, state)
    individual.state[:vaccinated] = false
    individual.state[:vaccination_time] = Inf
    return nothing
end

function apply_post_transmission!(::RingVaccination, state, new_contacts)
    for ind in new_contacts
        is_traced(ind) || continue
        is_vaccinated(ind) && continue
        vacc_t = isolation_time(ind)
        isfinite(vacc_t) || continue
        ind.state[:vaccinated] = true
        ind.state[:vaccination_time] = vacc_t
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

Initialises: `:vaccinated`, `:vaccination_time`.

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

Age-stratified rollout (65+ on day 30, younger on day 90):

```julia
MassVaccination(
    efficacy = 0.85,
    eligibility_time = (rng, ind) -> ind.state[:age] >= 65 ? 30.0 : 90.0,
    delay_to_immunity = 14.0,
)
```
"""
Base.@kwdef struct MassVaccination{E} <: AbstractVaccination
    efficacy::Float64
    eligibility_time::E
    delay_to_immunity::Float64 = 0.0
    mode::Symbol = :leaky
end

required_fields(::MassVaccination) = Symbol[]

function initialise_individual!(::MassVaccination, individual, state)
    individual.state[:vaccinated] = false
    individual.state[:vaccination_time] = Inf
    return nothing
end

function apply_post_transmission!(mv::MassVaccination, state, new_contacts)
    for ind in new_contacts
        is_vaccinated(ind) && continue
        vacc_t = _draw_eligibility_time(mv.eligibility_time, state.rng, ind)
        isfinite(vacc_t) || continue
        ind.state[:vaccinated] = true
        ind.state[:vaccination_time] = vacc_t
    end
    return nothing
end

_draw_eligibility_time(t::Real, rng, ind) = float(t)
_draw_eligibility_time(d::Distribution, rng, ind) = float(rand(rng, d))
_draw_eligibility_time(f, rng, ind) = float(f(rng, ind))
