# ── Observation models ─────────────────────────────────────────────
# State-space framework: the process model (TransmissionModel)
# describes the latent epidemiological dynamics; an ObservationModel
# describes how the latent state generates observed data. An observation
# is a forcing on the process (passed as `observation = …` to the
# constructor), so a single `loglikelihood(data, model)` dispatch covers
# any process / observation pairing for which the protocol is defined.

"""
How true cases become observed data: which cases are detected, how long
reporting takes, or which outbreaks are recorded at all. Pass one to a model
as `observation = ...`. Built in: [`NoObservation`](@ref) (every case seen),
[`PerCaseObservation`](@ref) (each case detected with some probability, after
a reporting delay) and [`MinimumSize`](@ref) (only chains of at least a
given size are recorded).

To write a new one, define [`observe`](@ref) (for the likelihood) and
[`apply_observation!`](@ref EpiBranch.apply_observation!) (for simulation)
for it; see "Adding an observation model" on the New transmission structures
page of the Extending EpiBranch guide.
"""
abstract type ObservationModel end

"""
    PerCaseObservation(; detection_prob = 1.0, delay = Dirac(0.0),
                       from = :onset_time)

Under-reporting and reporting delay: each case is reported independently with
probability `detection_prob`, `delay` days after symptom onset (or after
`from`). Each simulated case records `:reported` and `:report_time`.

- `detection_prob`: the probability a case is reported, a number, a
  distribution, or a function of the random number generator and the
  individual, `(rng, ind) -> ...` (for example to make reporting depend on
  age).
- `delay`: days from `from` to report, as a number, distribution or such a
  function.
- `from`: the time reporting is measured from. Symptom onset by default,
  because surveillance follows onset; `from = ind -> ind.infection_time`
  measures from infection. A case without an onset time (for example an
  asymptomatic case from `clinical_presentation`) is measured from its
  infection time.

`detection_prob = 1.0, delay = Dirac(0.0)` (the defaults) means every case is
seen at once; `detection_prob = ρ, delay = Dirac(0.0)` is under-reporting
alone.

!!! note
    The closed-form chain-size results ([`observe`](@ref),
    [`ThinnedChainSize`](@ref)) need `detection_prob` to be a single number
    and give an error otherwise. For reporting that varies between cases, use
    the simulation-based likelihood.

# Examples

```julia
# 60% of cases reported, on average 3 days after onset
PerCaseObservation(detection_prob = 0.6, delay = Gamma(3.0, 1.0))
```
"""
struct PerCaseObservation{P, D, F} <: ObservationModel
    detection_prob::P
    delay::D
    from::F
end

function PerCaseObservation(;
        detection_prob = 1.0,
        delay = Dirac(0.0),
        from = :onset_time
    )
    if detection_prob isa Real
        0.0 < detection_prob <= 1.0 || throw(
            ArgumentError(
                "detection_prob must be in (0, 1], got $detection_prob"
            )
        )
    end
    return PerCaseObservation(detection_prob, delay, from)
end

# Two-argument positional form preserved for terse callers — uses the
# default :onset_time anchor.
function PerCaseObservation(detection_prob, delay)
    return PerCaseObservation(; detection_prob, delay)
end

function Base.show(io::IO, o::PerCaseObservation)
    return print(
        io, "PerCaseObservation(detection_prob=$(o.detection_prob), ",
        "delay=$(o.delay), from=$(o.from))"
    )
end

"""Extract a scalar `detection_prob` for analytical paths that need it
(e.g. `ThinnedChainSize`). Throws if the observation model uses
per-individual variation (a `Distribution` or callable)."""
scalar_detection_prob(o::PerCaseObservation{<:Real}) = float(o.detection_prob)
function scalar_detection_prob(o::PerCaseObservation)
    throw(
        ArgumentError(
            "Closed-form analytics require a scalar detection_prob; " *
                "got $(typeof(o.detection_prob)). Use the simulation-based " *
                "likelihood instead, or pass a Real value."
        )
    )
end

"""
    MinimumSize(min_size)

Only chains of at least `min_size` cases are recorded, as when only clusters
of two or more cases are investigated. The exact likelihood conditions chain
sizes on being at least `min_size` ([`TruncatedChainSize`](@ref)), and the
simulation-based likelihood leaves out simulated chains below it, so both
describe the same recorded data. [`simulate`](@ref) and
[`chain_statistics`](@ref) still return every chain.

It acts on whole chains rather than on individual cases, so it leaves the
simulated cases themselves unchanged. A model has one observation model, so
`MinimumSize` cannot be combined with [`PerCaseObservation`](@ref).
"""
struct MinimumSize <: ObservationModel
    min_size::Int
    function MinimumSize(min_size::Integer)
        min_size >= 1 || throw(ArgumentError("min_size must be >= 1, got $min_size"))
        return new(Int(min_size))
    end
end
