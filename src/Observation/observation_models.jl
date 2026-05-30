# ── Concrete observation models ─────────────────────────────────────
# The abstract `ObservationModel` type lives in `EpiBranchBase`; this
# file defines per-case detection-and-delay observation, which is the
# canonical implementation. Composed with a `TransmissionModel` via
# `Observed(process, observation)`.

"""
    PerCaseObservation(; detection_prob = 1.0, delay = Dirac(0.0))

Independent per-case observation: each case is reported with
probability `detection_prob`, and reports lag the underlying
transmission event by an independent draw from `delay`. The
defaults reproduce instantaneous full reporting.

`detection_prob` and `delay` both accept the standard
`Real | Distribution | Function` trio:

- a `Real` (or `Distribution` for `delay`) reproduces the original
  behaviour;
- a `Function` `(rng, ind) -> Real` lets the value depend on
  per-individual state (e.g. age-conditional reporting probability).

Per-individual variation is honoured by the simulation path
(`simulate(::Observed{..., PerCaseObservation})`). The closed-form
analytical helpers — `ThinnedChainSize`, `chain_size_distribution` on
`Observed{..., PerCaseObservation}` — require a scalar `detection_prob`
and will throw when given a `Distribution` or `Function`; fall back
to the simulation likelihood for per-individual reporting.

`detection_prob = 1.0, delay = Dirac(0.0)` ↔ no observation effect.
`detection_prob = 1.0, delay = D` ↔ full reporting with delay `D`.
`detection_prob = ρ, delay = Dirac(0.0)` ↔ binomial thinning, no delay.
"""
struct PerCaseObservation{P, D} <: ObservationModel
    detection_prob::P
    delay::D
end

function PerCaseObservation(;
        detection_prob::Union{Real, Distribution, Function} = 1.0,
        delay::Union{Real, Distribution, Function} = Dirac(0.0))
    if detection_prob isa Real
        0.0 < detection_prob <= 1.0 || throw(ArgumentError(
            "detection_prob must be in (0, 1], got $detection_prob"))
    end
    PerCaseObservation(detection_prob, delay)
end

function Base.show(io::IO, o::PerCaseObservation)
    print(io, "PerCaseObservation(detection_prob=$(o.detection_prob), ",
        "delay=$(o.delay))")
end

"""Extract a scalar `detection_prob` for analytical paths that need it
(e.g. `ThinnedChainSize`). Throws if the observation model uses
per-individual variation (`Distribution` or `Function`)."""
scalar_detection_prob(o::PerCaseObservation{<:Real}) = float(o.detection_prob)
function scalar_detection_prob(o::PerCaseObservation)
    throw(ArgumentError(
        "Closed-form analytics require a scalar detection_prob; " *
        "got $(typeof(o.detection_prob)). Use the simulation-based " *
        "likelihood instead, or pass a Real value."))
end
