# ── Individual show + state accessors ───────────────────────────────
# Typed read/write helpers over `Individual.state`. Each is a small
# wrapper around `get(ind.state, ..., default)::T` so that the rest of
# the codebase doesn't sprinkle that pattern everywhere.

function Base.show(io::IO, ind::Individual)
    infected_str = is_infected(ind) ? "infected" : "contact-only"
    isolated_str = is_isolated(ind) ? ", isolated" : ""
    return print(
        io,
        "Individual(id=$(ind.id), gen=$(ind.generation), chain=$(ind.chain_id), t=$(round(ind.infection_time, digits = 1)), $(infected_str)$(isolated_str))"
    )
end

"""Symptom onset time (`NaN` if asymptomatic or not set); a dual under AD."""
function onset_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :onset_time, T(NaN)))::T
end

"""
Incubation period: time from infection to symptom onset (Float64, NaN if
asymptomatic or onset is not set). Useful inside a `generation_time`
function that links an individual's generation time to their own
incubation period.
"""
incubation_period(ind::Individual) = onset_time(ind) - ind.infection_time

"""
Time of the individual's terminal outcome — the earliest terminal
[`Transition`](@ref) to fire, e.g. recovery or death (`Inf` if none has
fired, whether because the case is still ongoing or the progression has no
terminal transition); a dual under AD.
"""
function outcome_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :outcome_time, T(Inf)))::T
end

"""Whether the individual is recorded as isolated, which is what tracing,
group vaccination and the line list read as a detection. An isolation that
[`Isolation`](@ref) does not record (see
[`EpiBranch.records_isolation`](@ref)) still removes the case from
transmission at [`isolation_time`](@ref) but leaves this `false`."""
is_isolated(ind::Individual) = _isolation_in_force(ind) && !_isolation_unrecorded(ind)

"""Time from which isolation or quarantine removes the individual from
transmission (`Inf` if never), whether or not the isolation is recorded as a
detection (see [`is_isolated`](@ref)); a dual under AD."""
function isolation_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :isolation_time, T(Inf)))::T
end

# Whether an isolation or quarantine stands on the individual, recorded or not.
# Interventions layering one isolation over another read this.
_isolation_in_force(ind::Individual) = get(ind.state, :isolated, false)::Bool

# Whether the standing isolation removes the case from transmission without
# counting as a detection.
_isolation_unrecorded(ind::Individual) = get(ind.state, :isolation_unrecorded, false)::Bool

# The time a detection reader sees: the isolation time, or `Inf` for an
# isolation that is not recorded.
function _recorded_isolation_time(ind::Individual{T}) where {T}
    return _isolation_unrecorded(ind) ? T(Inf) : isolation_time(ind)
end

"""Whether the individual was traced via contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated under the given `dose_label`. The
default label reads the plain `:vaccinated` key; a non-default label reads the
namespaced key an `AbstractVaccination` with that `dose_label` writes."""
function is_vaccinated(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _vaccinated_key(dose_label), false)::Bool
end

"""Time the individual's vaccine-induced immunity develops under the given
`dose_label` (`Inf` if not vaccinated); a dual under AD. This is
`vaccination_time + delay_to_immunity`, recorded by [`AbstractVaccination`](@ref)
at vaccination time so a clinical transition can check it without reaching
for the vaccination object, which it never sees. Compare against the time an
outcome would take effect (e.g. `onset_time(ind)` for the default `Death`) to
decide whether that dose's [`severity_efficacy`](@ref) applies: a dose whose
immunity develops after that time confers no protection."""
function immunity_time(ind::Individual{T}; dose_label::Symbol = :default) where {T}
    return convert(T, get(ind.state, _immunity_time_key(dose_label), T(Inf)))::T
end

"""Probability that the individual's own disease course is milder — e.g. a
lower chance of death — once their vaccine-induced immunity has developed
(`0.0` if not vaccinated, or if the dose carries no severity effect). Sampled
once per vaccinated individual, alongside `efficacy`, from the
`severity_efficacy` of the dose's [`VaccineEffect`](@ref). Read this from a
clinical transition's `probability`, gated on [`immunity_time`](@ref) having
passed:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```
"""
function severity_efficacy(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _severity_efficacy_key(dose_label), 0.0)::Float64
end

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""Whether the individual's infection was aborted before symptom onset, as a
post-exposure dose of [`RingVaccination`](@ref) can do. The infection lasts
until `:infection_aborted_time` and ends there, before any onset. The abort is
recorded against the contact's exposure at the time the dose is given, or at a
later exposure of a contact already given it. It is removed when infection is
resolved if that exposure does not infect the contact before the abort time."""
_infection_aborted(ind::Individual) = haskey(ind.state, :infection_aborted_time)

"""Whether the individual develops symptoms: it is not asymptomatic and its
infection was not aborted before onset."""
_develops_symptoms(ind::Individual) = !is_asymptomatic(ind) && !_infection_aborted(ind)

"""Whether the individual tested positive."""
is_test_positive(ind::Individual) = get(ind.state, :test_positive, false)::Bool

"""Whether the individual was successfully infected (vs contact only)."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""Type index for multi-type branching processes (default 1)."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

"""Mark an individual as isolated at the given time (any `Real`, so an AD
dual isolation time flows through).

The time is stored under `:isolation_time`. A route window that isolation should
end lists [`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, which
respects leaky isolation. `:isolated` in an `until` refers to a
`Transition(:isolated, …)` in the natural history."""
function set_isolated!(ind::Individual, time::Real)
    ind.state[:isolated] = true
    delete!(ind.state, :isolation_unrecorded)
    return ind.state[:isolation_time] = time
end

"""Clear an individual's isolation, the inverse of [`set_isolated!`](@ref)."""
function clear_isolated!(ind::Individual)
    ind.state[:isolated] = false
    delete!(ind.state, :isolation_unrecorded)
    ind.state[:isolation_time] = Inf
    return nothing
end
