# ── Observation models ─────────────────────────────────────────────
# State-space framework: the process model (TransmissionModel)
# describes the latent epidemiological dynamics; an ObservationModel
# describes how the latent state generates observed data. The two are
# combined by `Observed(process, observation)` so that a single
# `loglikelihood(data, model)` dispatch covers any process /
# observation pairing for which a method is defined.

"""
Abstract supertype for observation models. Subtypes describe how
underlying transmission events generate observable data — per-case
detection, reporting delays, aggregation, multi-stream surveillance,
etc. Composed with a `TransmissionModel` via [`Observed`](@ref).
"""
abstract type ObservationModel end

"""
    PerCaseObservation(; detection_prob = 1.0, delay = Dirac(0.0),
                       from = :onset_time)

Independent per-case observation: each case is reported with
probability `detection_prob`, and reports lag the anchor time given by
`from` by an independent draw from `delay`. `from` defaults to
`:onset_time` because real surveillance lags symptom onset, not
infection. Set `from = ind -> ind.infection_time` to anchor on
infection time instead.

If the anchor evaluates to `NaN` (e.g. an asymptomatic case under
`clinical_presentation`), reporting falls back to the infection time so
the report time is still well-defined.

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
struct PerCaseObservation{P, D, F} <: ObservationModel
    detection_prob::P
    delay::D
    from::F
end

function PerCaseObservation(;
        detection_prob::Union{Real, Distribution, Function} = 1.0,
        delay::Union{Real, Distribution, Function} = Dirac(0.0),
        from::Union{Symbol, Function} = :onset_time)
    if detection_prob isa Real
        0.0 < detection_prob <= 1.0 || throw(ArgumentError(
            "detection_prob must be in (0, 1], got $detection_prob"))
    end
    PerCaseObservation(detection_prob, delay, from)
end

# Two-argument positional form preserved for terse callers — uses the
# default :onset_time anchor.
function PerCaseObservation(detection_prob::Union{Real, Distribution, Function},
        delay::Union{Real, Distribution, Function})
    PerCaseObservation(; detection_prob, delay)
end

function Base.show(io::IO, o::PerCaseObservation)
    print(io, "PerCaseObservation(detection_prob=$(o.detection_prob), ",
        "delay=$(o.delay), from=$(o.from))")
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
