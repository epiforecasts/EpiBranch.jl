# ── Observation models (abstract) ──────────────────────────────────
# State-space framework: the process model (TransmissionModel)
# describes the latent epidemiological dynamics; an ObservationModel
# describes how the latent state generates observed data. The two are
# combined by `Observed(process, observation)` so that a single
# `loglikelihood(data, model)` dispatch covers any process /
# observation pairing for which a method is defined.
#
# Concrete observation models live in `EpiBranchObservation`. The abstract
# type is declared here so any package that needs to subtype it can
# depend on `EpiBranchCore` alone.

"""
Abstract supertype for observation models. Subtypes describe how
underlying transmission events generate observable data — per-case
detection, reporting delays, aggregation, multi-stream surveillance,
etc. Composed with a `TransmissionModel` via `Observed`.
"""
abstract type ObservationModel end
