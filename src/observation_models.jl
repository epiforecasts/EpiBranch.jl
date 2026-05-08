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

"""
    Snapshot(time_since)

Cluster-level observation timing. For each cluster, an inner vector
of times since each known reported case (in time units matching the
generation-time and reporting-delay distributions; days are
conventional). Encoding:

- `[Inf]` — cluster is concluded; no further reports possible.
- `[τ]` for finite `τ` — single-most-recent-case approximation.
- `[τ_1, …, τ_x]` for x finite values — exact per-case product.
- `[]` (empty) — observational claim that the cluster is ongoing
  with no timing data; the likelihood uses the chain-size right-tail
  only. Equivalent to the binary "ongoing" classification in the
  Endo-style threshold rule.

Convenience constructors accept either a vector of scalars (one
representative time per cluster, wrapped automatically) or a vector
of vectors (full per-cluster timing, preserved as-is).
"""
struct Snapshot{T <: AbstractFloat}
    time_since::Vector{Vector{T}}

    function Snapshot{T}(time_since::Vector{Vector{T}}) where {T <: AbstractFloat}
        isempty(time_since) && throw(ArgumentError("Snapshot must be non-empty"))
        for v in time_since
            all(>=(0), v) ||
                throw(ArgumentError("times since cases must be non-negative"))
        end
        new{T}(time_since)
    end
end

# Vector of scalars per cluster (single-most-recent-case mode).
Snapshot(τ::AbstractVector{<:Real}) = Snapshot{Float64}([[Float64(t)] for t in τ])

# Vector of vectors per cluster (per-case mode, with arbitrary inner
# lengths — including 0 for ongoing/right-tail clusters).
function Snapshot(τ::AbstractVector{<:AbstractVector{<:Real}})
    Snapshot{Float64}([Float64.(v) for v in τ])
end

function Base.show(io::IO, s::Snapshot)
    n = length(s.time_since)
    print(io, "Snapshot($n cluster$(n == 1 ? "" : "s"))")
end

Base.length(s::Snapshot) = length(s.time_since)
