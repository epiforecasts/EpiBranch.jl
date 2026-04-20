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

"""
    ThinnedChainSize(base, detection_prob)

Chain size distribution under independent per-case detection with
probability `detection_prob`, applied to a base chain size distribution.
Nesting is meaningful: each level applies an additional Binomial thinning.

Generic `logpdf` marginalises over the true (unobserved) chain size by
summing `P(true = n) * Binomial(n, p, obs)` for `n >= obs`. Nesting
therefore composes correctly for any base without per-pair specialisations.
"""
struct ThinnedChainSize{D <: DiscreteUnivariateDistribution} <:
       DiscreteUnivariateDistribution
    base::D
    detection_prob::Float64
end

Distributions.minimum(::ThinnedChainSize) = 1
Distributions.maximum(::ThinnedChainSize) = Inf
Distributions.insupport(::ThinnedChainSize, n::Integer) = n >= 1

function Distributions.logpdf(d::ThinnedChainSize, obs::Integer)
    obs < 1 && return -Inf
    p = d.detection_prob
    # Sum terms logpdf(base, n) + logpdf(Binomial(n, p), obs) for n >= obs
    # until the running log-sum-exp stabilises within `tol`. This avoids
    # a heuristic truncation (e.g. obs*5) that under-integrates heavy-tailed
    # base distributions like GammaBorel with low k.
    tol = 1e-12
    acc = Float64[logpdf(d.base, obs) + logpdf(Binomial(obs, p), obs)]
    prev = logsumexp(acc)
    n = obs + 1
    max_n = 100_000
    while n <= max_n
        push!(acc, logpdf(d.base, n) + logpdf(Binomial(n, p), obs))
        cur = logsumexp(acc)
        if isfinite(cur) && isfinite(prev) && abs(cur - prev) < tol &&
           n - obs >= 20
            return cur
        end
        prev = cur
        n += 1
    end
    return prev
end

Distributions.pdf(d::ThinnedChainSize, n::Integer) = exp(logpdf(d, n))

"""
    chain_size_distribution(m::PartiallyObserved)

Compositional chain size distribution for a partially-observed model.
Recursively extracts the wrapped model's chain size distribution and
wraps it in `ThinnedChainSize`. Nested `PartiallyObserved` therefore
compose via the generic logpdf — no per-pair specialisations required.
"""
function chain_size_distribution(m::PartiallyObserved)
    ThinnedChainSize(chain_size_distribution(m.model), m.detection_prob)
end

# Optimisation only: stacking per-case detection collapses algebraically
# to a single Binomial with the product of detection probabilities.
# Without this constructor, nesting still produces a correct (but nested)
# ThinnedChainSize — the correctness comes from the chain_size_distribution
# dispatch above, not from this specialisation.
function PartiallyObserved(inner::PartiallyObserved, detection_prob::Real)
    PartiallyObserved(inner.model, detection_prob * inner.detection_prob)
end

"""
    PartiallyObserved(detection_prob) -> m -> PartiallyObserved(m, detection_prob)

Curried form for pipe composition: `model |> PartiallyObserved(0.7)`.
"""
PartiallyObserved(detection_prob::Real) = m -> PartiallyObserved(m, detection_prob)
