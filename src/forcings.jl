# ── Model forcings: the population, policy and observation a model carries ──
# A process stores its interventions, population attributes and
# observation model alongside its core dynamics, so the engine and the
# likelihood read them off the model itself. The accessors below are
# defined once against the abstract `TransmissionModel` and shared by
# every process type, so a new process (say a future `HouseholdProcess`)
# picks up the forcings, the engine integration and the simulation-based
# likelihood by holding a `Forcings` field and defining `forcings`.

"""No observation model: latent cases are observed exactly."""
struct NoObservation <: ObservationModel end

# Internal container, never constructed by users — they pass
# `interventions` / `attributes` / `observation` to a process
# constructor and these are gathered here.
struct Forcings{A, I <: AbstractVector, O <: ObservationModel}
    attributes::A
    interventions::I
    observation::O
end

const _NO_FORCINGS = Forcings(NoAttributes(), AbstractIntervention[], NoObservation())

function make_forcings(; attributes = NoAttributes(),
        interventions = AbstractIntervention[],
        observation::ObservationModel = NoObservation())
    ivs = interventions isa AbstractVector ? interventions : [interventions]
    return Forcings(attributes,
        convert(Vector{AbstractIntervention}, ivs), observation)
end

# Shared accessors. A process opts in by defining `forcings`; the three
# component accessors are defined once here against `TransmissionModel`.
forcings(::TransmissionModel) = _NO_FORCINGS
_interventions(m::TransmissionModel) = forcings(m).interventions
_attributes(m::TransmissionModel) = forcings(m).attributes
_observation(m::TransmissionModel) = forcings(m).observation
