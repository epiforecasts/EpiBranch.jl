# ── Model inputs: the population, policy and observation a model carries ──
# A process stores its interventions, population attributes and
# observation model alongside its core dynamics, so the engine and the
# likelihood read them off the model itself. The accessors below are
# defined once against the abstract `TransmissionModel` and shared by
# every process type, so a new process (say a future `HouseholdProcess`)
# picks up the engine integration and the simulation-based likelihood by
# holding the three fields and defining `interventions`, `attributes` and
# `observation`.

"""No observation model: latent cases are observed exactly."""
struct NoObservation <: ObservationModel end

# Shared accessors with abstract-type defaults, so a model that carries
# none of these defines nothing. A process opts in by defining whichever
# of `interventions`, `attributes` and `observation` it stores.
interventions(::TransmissionModel) = AbstractIntervention[]
attributes(::TransmissionModel) = NoAttributes()
observation(::TransmissionModel) = NoObservation()

# Normalise an interventions keyword (a single intervention or a vector)
# to a `Vector{AbstractIntervention}`, for constructors to use.
function _intervention_vector(ivs)
    convert(Vector{AbstractIntervention}, ivs isa AbstractVector ? ivs : [ivs])
end
