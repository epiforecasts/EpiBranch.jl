# ── Observation models ──────────────────────────────────────────────
# Wrapper models that describe how a transmission process is observed.
# Compose with any TransmissionModel; dispatch picks up analytical
# likelihoods where available and falls back to simulation otherwise.

"""
    PartiallyObserved(model, detection_prob)

Wrap a `TransmissionModel` with independent per-case detection. Each
case in a chain is detected independently with probability
`detection_prob`; only chains with at least one detected case are
observed.

Likelihood methods marginalise over the true (unobserved) chain sizes
using the underlying model's analytical chain size distribution when
available, or simulation otherwise.

# Examples

```julia
base = BranchingProcess(NegBin(0.8, 0.5))
model = PartiallyObserved(base, 0.7)
loglikelihood(ChainSizes([1, 1, 2, 3]), model)
```
"""
struct PartiallyObserved{M <: TransmissionModel} <: TransmissionModel
    model::M
    detection_prob::Float64

    function PartiallyObserved(model::TransmissionModel, detection_prob::Real)
        0.0 < detection_prob <= 1.0 ||
            throw(ArgumentError("detection_prob must be in (0, 1], got $detection_prob"))
        new{typeof(model)}(model, Float64(detection_prob))
    end
end

function Base.show(io::IO, m::PartiallyObserved)
    print(io, "PartiallyObserved($(m.model), detection_prob=$(m.detection_prob))")
end

population_size(m::PartiallyObserved) = population_size(m.model)
latent_period(m::PartiallyObserved) = latent_period(m.model)
n_types(m::PartiallyObserved) = n_types(m.model)

# Wrappers delegate offspring extraction through to the wrapped model.
_single_type_offspring(m::PartiallyObserved) = _single_type_offspring(m.model)

# Pure per-case detection thins Binomially; stacking compounds the
# probabilities. This keeps the underlying generative model free of
# observation state and makes repeated wrapping mathematically correct.
function PartiallyObserved(inner::PartiallyObserved, detection_prob::Real)
    PartiallyObserved(inner.model, detection_prob * inner.detection_prob)
end

"""
    PartiallyObserved(detection_prob) -> m -> PartiallyObserved(m, detection_prob)

Curried form for pipe composition: `model |> PartiallyObserved(0.7)`.
"""
PartiallyObserved(detection_prob::Real) = m -> PartiallyObserved(m, detection_prob)
