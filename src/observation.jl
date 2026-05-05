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

Convenience constructor that builds
`Surveilled(model, PerCaseObservation(detection_prob, Dirac(0.0)))`.
Equivalent to per-case binomial thinning with no reporting delay.
Each case in a chain is detected with probability `detection_prob`;
chains with zero detected cases are unobserved.

# Examples

```julia
base = BranchingProcess(NegBin(0.8, 0.5))
model = PartiallyObserved(base, 0.7)
loglikelihood(ChainSizes([1, 1, 2, 3]), model)
```
"""
function PartiallyObserved(model::TransmissionModel, detection_prob::Real)
    Surveilled(model, PerCaseObservation(detection_prob, Dirac(0.0)))
end

"""
    Reported(model, delay)

Convenience constructor that builds
`Surveilled(model, PerCaseObservation(1.0, delay))`. Equivalent to
full per-case reporting with the given delay distribution.

# Examples

```julia
base = BranchingProcess(NegBin(2.5, 0.1), Gamma(2.0, 2.5))
model = Reported(base, LogNormal(1.6, 0.4))
loglikelihood(real_time_data, model)
```
"""
function Reported(model::TransmissionModel, delay::Distribution)
    Surveilled(model, PerCaseObservation(1.0, delay))
end

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
    chain_size_distribution(m::Surveilled{<:Any, <:PerCaseObservation})

Return `ThinnedChainSize` wrapping the chain size distribution of the
inner process. Recurses through any nested wrappers, so e.g.
`PartiallyObserved(PartiallyObserved(model, p1), p2)` (which becomes
`Surveilled(Surveilled(model, ...), ...)` in the type) gives the same
nested-thinning likelihood without pairwise dispatch.
"""
function chain_size_distribution(m::Surveilled{<:Any, <:PerCaseObservation})
    ThinnedChainSize(chain_size_distribution(m.process),
        m.observation.detection_prob)
end

# Binomial thinning compounds: two rounds with p1, p2 equal one round
# with p1·p2. Collapse at construction time to avoid a needlessly
# nested Surveilled. Performance shortcut; without it the generic
# chain_size_distribution dispatch still gives the right likelihood,
# just via a deeper nested sum.
function PartiallyObserved(inner::Surveilled{<:Any, <:PerCaseObservation},
        detection_prob::Real)
    inner_p = inner.observation.detection_prob
    PartiallyObserved(inner.process, detection_prob * inner_p)
end

"""
    PartiallyObserved(detection_prob) -> m -> PartiallyObserved(m, detection_prob)

Call with only the detection probability to get a function that wraps
a model. Enables pipe syntax like `model |> PartiallyObserved(0.7)`.
"""
PartiallyObserved(detection_prob::Real) = m -> PartiallyObserved(m, detection_prob)
