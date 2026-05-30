# ── Individual show + engine-owned state accessors ──────────────────
# Typed read/write helpers over `Individual.state`. Only the accessors
# for keys the engine itself owns or that are part of the generic
# clinical-attribute surface live in Base; intervention-specific
# accessors (`is_isolated`, `is_traced`, `is_vaccinated`, …) live in
# the owning submodule.

function Base.show(io::IO, ind::Individual)
    infected_str = is_infected(ind) ? "infected" : "contact-only"
    print(io,
        "Individual(id=$(ind.id), gen=$(ind.generation), chain=$(ind.chain_id), t=$(round(ind.infection_time, digits=1)), $(infected_str))")
end

"""Whether the individual was successfully infected (vs contact only).
Set by the engine's competing-risks resolution."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""Type index for multi-type branching processes (default 1).
Set by the engine when offspring are allocated to types."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

"""Symptom onset time (Float64, NaN if asymptomatic or not set).
Set by `clinical_presentation` or any equivalent attributes builder."""
onset_time(ind::Individual) = get(ind.state, :onset_time, NaN)::Float64

"""Whether the individual is asymptomatic. Set by `clinical_presentation`
or any equivalent attributes builder."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool
