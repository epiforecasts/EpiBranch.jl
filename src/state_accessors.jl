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

"""Time of isolation (Float64, Inf if not isolated)."""
isolation_time(ind::Individual) = get(ind.state, :isolation_time, Inf)::Float64

"""Whether the individual was traced via contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated."""
is_vaccinated(ind::Individual) = get(ind.state, :vaccinated, false)::Bool

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""Whether the individual tested positive."""
is_test_positive(ind::Individual) = get(ind.state, :test_positive, false)::Bool

"""Whether the individual was successfully infected (vs contact only)."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""Type index for multi-type branching processes (default 1)."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

"""Mark an individual as isolated at the given time."""
function set_isolated!(ind::Individual, time::Float64)
    ind.state[:isolated] = true
    ind.state[:isolation_time] = time
end
