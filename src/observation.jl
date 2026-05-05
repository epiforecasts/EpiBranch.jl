# ── Observation models ──────────────────────────────────────────────
# Wrappers around a TransmissionModel that describe how it is observed.
# Dispatch uses the analytical likelihood when one is defined; otherwise
# it falls back to simulation.

"""
    Surveilled(process, observation)

State-space combiner: pair a `TransmissionModel` with an
[`ObservationModel`](@ref). Dispatch on `Surveilled{P, O}` is the
single surface for likelihoods that need to know about both the
latent dynamics and how they are observed.

Forwards process-model accessors (`population_size`, `latent_period`,
`n_types`, `_single_type_offspring`) to `process`, so analytical
helpers that route through them work transparently on the combined
model.
"""
struct Surveilled{P <: TransmissionModel, O <: ObservationModel} <: TransmissionModel
    process::P
    observation::O
end

function Base.show(io::IO, m::Surveilled)
    print(io, "Surveilled($(m.process), $(m.observation))")
end

population_size(m::Surveilled) = population_size(m.process)
latent_period(m::Surveilled) = latent_period(m.process)
n_types(m::Surveilled) = n_types(m.process)
_single_type_offspring(m::Surveilled) = _single_type_offspring(m.process)

"""
    PartiallyObserved(model, detection_prob)

Wrap a `TransmissionModel` with independent per-case detection. Each
case in a chain is detected with probability `detection_prob`. Chains
with zero detected cases are unobserved.

Likelihoods marginalise over the true (unobserved) chain sizes. If the
wrapped model has an analytical chain size distribution, that is used;
otherwise the likelihood falls back to simulation.

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

# Forward offspring extraction so analytical helpers that route through
# `_single_type_offspring` (extinction_probability, probability_contain,
# superspreading metrics) work on wrapped models. Per-case detection
# does not change transmission dynamics, so these helpers delegate to
# the wrapped model.
_single_type_offspring(m::PartiallyObserved) = _single_type_offspring(m.model)

"""
    Reported(model, delay)

Wrap a `TransmissionModel` with a per-case reporting delay. `delay` is
a `Distribution` over the time between infection and report; combined
with the model's generation time it determines when secondary cases
of an observed case can themselves still be reported.

Used by the real-time cluster-size likelihood to fold reporting delay
into the end-of-outbreak probability `π(τ)`. The bare
`BranchingProcess` likelihood treats reports as instantaneous; wrap
with `Reported` to account for delay.

# Examples

```julia
base = BranchingProcess(NegBin(2.5, 0.1), Gamma(2.0, 2.5))
model = Reported(base, LogNormal(1.6, 0.4))
loglikelihood(real_time_data, model)
```
"""
struct Reported{M <: TransmissionModel, D <: Distribution} <: TransmissionModel
    model::M
    delay::D
end

function Base.show(io::IO, m::Reported)
    print(io, "Reported($(m.model), delay=$(m.delay))")
end

population_size(m::Reported) = population_size(m.model)
latent_period(m::Reported) = latent_period(m.model)
n_types(m::Reported) = n_types(m.model)

# Reporting delay does not change transmission dynamics, so offspring-
# based analytical helpers (extinction probability, etc.) delegate to
# the wrapped model.
_single_type_offspring(m::Reported) = _single_type_offspring(m.model)

"""
    ThinnedChainSize(base, detection_prob)

Distribution of observed chain sizes when each case in a `base` chain
is detected with probability `detection_prob`. Nesting applies a second
round of Binomial thinning.

`logpdf(d, obs)` sums `logpdf(base, n) + logpdf(Binomial(n, p), obs)`
over `n >= obs` until the tail is negligible. The computation only
needs `logpdf` on the base, so a nested `ThinnedChainSize` gives the
same answer as a single thinning with a compounded probability without
any specialised method.
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
    # Streaming log-sum-exp: maintain running max `m` and `S = Σ exp(xᵢ - m)`.
    # Stop when the accumulated value stops changing (within `tol`) and at
    # least 20 further terms have been added, to avoid false early
    # convergence on heavy-tailed bases (e.g. GammaBorel with low k).
    tol = 1e-12
    max_n = 100_000
    first = logpdf(d.base, obs) + logpdf(Binomial(obs, p), obs)
    m = first
    S = 1.0
    prev = m
    n = obs + 1
    while n <= max_n
        x = logpdf(d.base, n) + logpdf(Binomial(n, p), obs)
        if x > m
            S = 1.0 + S * exp(m - x)
            m = x
        else
            S += exp(x - m)
        end
        cur = m + log(S)
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

Return `ThinnedChainSize` wrapping the chain size distribution of the
inner model. Because the call recurses, a nested `PartiallyObserved`
produces a nested `ThinnedChainSize`, and `logpdf` on that gives the
right likelihood without any pairwise dispatch.
"""
function chain_size_distribution(m::PartiallyObserved)
    ThinnedChainSize(chain_size_distribution(m.model), m.detection_prob)
end

# Binomial thinning compounds: two rounds with p1, p2 equal one round
# with p1*p2. We collapse at construction time to avoid a needlessly
# nested ThinnedChainSize. This is a performance shortcut; without it
# the generic chain_size_distribution dispatch still gives the right
# likelihood, just via a deeper nested sum.
function PartiallyObserved(inner::PartiallyObserved, detection_prob::Real)
    PartiallyObserved(inner.model, detection_prob * inner.detection_prob)
end

"""
    PartiallyObserved(detection_prob) -> m -> PartiallyObserved(m, detection_prob)

Call with only the detection probability to get a function that wraps
a model. Enables pipe syntax like `model |> PartiallyObserved(0.7)`.
"""
PartiallyObserved(detection_prob::Real) = m -> PartiallyObserved(m, detection_prob)
