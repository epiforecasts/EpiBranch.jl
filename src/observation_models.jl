# ── Observation models ─────────────────────────────────────────────
# State-space framework: the process model (TransmissionModel)
# describes the latent epidemiological dynamics; an ObservationModel
# describes how the latent state generates observed data. The two are
# combined by `Surveilled(process, observation)` so that a single
# `loglikelihood(data, model)` dispatch covers any process /
# observation pairing for which a method is defined.

"""
Abstract supertype for observation models. Subtypes describe how
underlying transmission events generate observable data — per-case
detection, reporting delays, aggregation, multi-stream surveillance,
etc. Composed with a `TransmissionModel` via [`Surveilled`](@ref).
"""
abstract type ObservationModel end

"""
    PerCaseObservation(; detection_prob = 1.0, delay = Dirac(0.0))

Independent per-case observation: each case is reported with
probability `detection_prob`, and reports lag the underlying
transmission event by an independent draw from `delay`. The
defaults reproduce instantaneous full reporting.

`detection_prob = 1.0, delay = Dirac(0.0)` ↔ no observation effect.
`detection_prob = 1.0, delay = D` ↔ full reporting with delay `D`.
`detection_prob = ρ, delay = Dirac(0.0)` ↔ binomial thinning, no delay.
"""
struct PerCaseObservation{D <: Distribution} <: ObservationModel
    detection_prob::Float64
    delay::D

    function PerCaseObservation(detection_prob::Real, delay::D) where {D <: Distribution}
        0.0 < detection_prob <= 1.0 || throw(ArgumentError(
            "detection_prob must be in (0, 1], got $detection_prob"))
        new{D}(Float64(detection_prob), delay)
    end
end
function PerCaseObservation(; detection_prob::Real = 1.0,
        delay::Distribution = Dirac(0.0))
    PerCaseObservation(detection_prob, delay)
end

function Base.show(io::IO, o::PerCaseObservation)
    print(io, "PerCaseObservation(detection_prob=$(o.detection_prob), ",
        "delay=$(o.delay))")
end
