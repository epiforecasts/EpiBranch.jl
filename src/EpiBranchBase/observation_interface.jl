# ── Observation models ─────────────────────────────────────────────
# State-space framework: the process model (TransmissionModel)
# describes the latent epidemiological dynamics; an ObservationModel
# describes how the latent state generates observed data. Concrete
# observation models live in `Observation`.

"""
Abstract supertype for observation models. Subtypes describe how
underlying transmission events generate observable data — per-case
detection, reporting delays, aggregation, multi-stream surveillance,
etc. Composed with a `TransmissionModel` via [`Observed`](@ref EpiBranch.Observation.Observed).
"""
abstract type ObservationModel end
