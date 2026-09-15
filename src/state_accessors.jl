# ── Individual show + state accessors ───────────────────────────────
# Typed read/write helpers over `Individual.state`. Each is a small
# wrapper around `get(ind.state, ..., default)::T` so that the rest of
# the codebase doesn't sprinkle that pattern everywhere.

function Base.show(io::IO, ind::Individual)
    infected_str = is_infected(ind) ? "infected" : "contact-only"
    isolated_str = is_isolated(ind) ? ", isolated" : ""
    print(io,
        "Individual(id=$(ind.id), gen=$(ind.generation), chain=$(ind.chain_id), t=$(round(ind.infection_time, digits=1)), $(infected_str)$(isolated_str))")
end

"""Symptom onset time (`NaN` if asymptomatic or not set); a dual under AD."""
onset_time(ind::Individual{T}) where {T} = convert(T, get(ind.state, :onset_time, T(NaN)))

"""
Incubation period: time from infection to symptom onset (Float64, NaN if
asymptomatic or onset is not set). Useful inside a `generation_time`
function that links an individual's generation time to their own
incubation period.
"""
incubation_period(ind::Individual) = onset_time(ind) - ind.infection_time

"""Whether the individual is isolated."""
is_isolated(ind::Individual) = get(ind.state, :isolated, false)::Bool

"""Time of isolation (Inf if not isolated); a dual under AD."""
function isolation_time(ind::Individual{T}) where {T}
    convert(T, get(ind.state, :isolation_time, T(Inf)))
end

"""Whether the individual was traced via contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated under the given `dose_label`. The
default label reads the plain `:vaccinated` key; a non-default label reads the
namespaced key an `AbstractVaccination` with that `dose_label` writes."""
function is_vaccinated(ind::Individual; dose_label::Symbol = :default)
    get(ind.state, _vaccinated_key(dose_label), false)::Bool
end

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""Whether the individual's infection was aborted before symptom onset, as a
post-exposure dose of [`RingVaccination`](@ref) can do. The infection stands up
to `:infection_aborted_time` and ends there, before any onset. Recorded when the
dose is given, or when a contact already given it is exposed again, against the
exposure the contact has then, and removed when infection is resolved if that
exposure does not infect the contact before the abort time."""
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
    ind.state[:isolation_time] = time
end

"""Clear an individual's isolation, the inverse of [`set_isolated!`](@ref)."""
function clear_isolated!(ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end
