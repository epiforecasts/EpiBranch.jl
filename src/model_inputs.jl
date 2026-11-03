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
# of `interventions`, `attributes`, `observation` and `recorder` it stores.
"""Control measures `model` applies to cases and their contacts, as a vector
of [`AbstractIntervention`](@ref)s, or `[]` if it has none (the
default). A process that stores its own interventions defines this to
return them; composed onto any process with a [`ModelSpec`](@ref)."""
interventions(::TransmissionModel) = AbstractIntervention[]

"""Population characteristics `model` draws for each person it creates, such
as [`clinical_presentation`](@ref) or [`demographics`](@ref), or
[`NoAttributes`](@ref) if it has none (the default). A process that
stores its own attributes defines this to return them; composed onto any
process with a [`ModelSpec`](@ref)."""
attributes(::TransmissionModel) = NoAttributes()

"""The observation model `model` has, such as [`MinimumSize`](@ref), or
[`NoObservation`](@ref) if it has none (the default). A process that
stores its own observation model defines this to return it; composed onto
any process with a [`ModelSpec`](@ref)."""
observation(::TransmissionModel) = NoObservation()

"""The [`ContactRecorder`](@ref) `model` has, or
[`NoContactRecorder`](@ref) if it has none (the default); see
[`ContactRecorder`](@ref) for when one is needed."""
recorder(::TransmissionModel) = NoContactRecorder()

# The within-host natural history: the clinical-state transitions a case moves
# through. Composed onto a process with a `ModelSpec`; empty for a bare process.
"""The within-host natural history `model` has: a vector of
[`Transition`](@ref)s and other clinical steps, or `[]` if it has none
(the default, for a bare process with no [`ModelSpec`](@ref) around it)."""
progression(::TransmissionModel) = AbstractClinicalTransition[]

# Normalise an interventions keyword (a single intervention or a vector)
# to a `Vector{AbstractIntervention}`, for constructors to use.
function _intervention_vector(ivs)
    return convert(Vector{AbstractIntervention}, ivs isa AbstractVector ? ivs : [ivs])
end
