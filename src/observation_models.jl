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
[`MinimumSize`](@ref) (only chains of at least a given size are recorded),
[`ChainDetection`](@ref) (a whole chain is recorded with a probability that
grows with its size). To report each case with some probability after a
delay, use a [`Reporting`](@ref) step in the model's `progression`.

To write a new one, define [`observe`](@ref) (for the likelihood) and
[`apply_observation!`](@ref EpiBranch.apply_observation!) (for simulation)
for it; see "Adding an observation model" in the Extending
EpiBranch guide.
"""
abstract type ObservationModel end

"""
    MinimumSize(min_size)

Only chains of at least `min_size` cases are recorded, as when only clusters
of two or more cases are investigated. `min_size` is at least 1. The exact
likelihood conditions chain
sizes on being at least `min_size` ([`TruncatedChainSize`](@ref)), and the
simulation-based likelihood leaves out simulated chains below it, so both
describe the same recorded data. [`simulate`](@ref) and
[`chain_statistics`](@ref) still return every chain.

When the reproduction number exceeds 1, some chains never end. The event "the
chain reached at least `min_size`" includes those chains as well as finite
ones.

It acts on whole chains rather than on individual cases, so it leaves the
simulated cases themselves unchanged.
"""
struct MinimumSize <: ObservationModel
    min_size::Int
    function MinimumSize(min_size::Integer)
        min_size >= 1 || throw(ArgumentError("min_size must be >= 1, got $min_size"))
        return new(Int(min_size))
    end
end

"""
    ChainDetection(recording_probability)
    ChainDetection(q)

A chain is either found and investigated in full, or missed altogether, as
when surveillance relies on spotting a cluster rather than each case on its
own. `recording_probability` is a function of chain size: `n -> P(a chain of
n cases is recorded)`. Any detection rule that grows with size can be passed
this way.

`ChainDetection(q::Real)` builds the common case directly: each case
independently brings its chain to attention with probability `q`
(`0 < q <= 1`); a chain of `n` cases is then recorded with probability
`1 - (1 - q)^n`.

It acts on whole chains rather than on individual cases: every case in a
recorded chain is marked `:reported`, and every case in a missed one is not.
Because a model has only one observation model, `ChainDetection` cannot be
combined with a [`Reporting`](@ref) progression step.

# Examples

```julia
# each case has an independent 30% chance of bringing its chain to light
ChainDetection(0.3)

# a detection rule of your own, as a function of chain size
ChainDetection(n -> min(1.0, 0.1 * n))
```
"""
struct ChainDetection{F} <: ObservationModel
    recording_probability::F
end

function ChainDetection(q::Real)
    0.0 < q <= 1.0 || throw(ArgumentError("q must be in (0, 1], got $q"))
    return ChainDetection(n -> -expm1(n * log1p(-q)))
end
