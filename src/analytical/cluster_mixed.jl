# ── Cluster-level heterogeneity ───────────────────────────────────────
# Offspring specification where the offspring distribution parameters
# vary across chains (clusters). For each chain a parameter θ is drawn
# from `mixing`; within a chain the offspring distribution is `build(θ)`.

"""
Marker type for Poisson offspring, used so that
`ClusterMixed(Poisson, mixing)` constructs a `ClusterMixed` with a
statically known offspring family, enabling dispatch to closed-form
likelihoods where available (e.g. Poisson + Gamma → `PoissonGammaChainSize`).
"""
struct PoissonFamily end
(::PoissonFamily)(λ) = Poisson(λ)

"""
    ClusterMixed(build, mixing)

Offspring specification with cluster-level heterogeneity: the offspring
distribution's parameters vary across chains. For each chain a value
`θ` is sampled from `mixing`, and the offspring distribution for that
chain is `build(θ)`.

When `build` is a distribution family type (e.g. `Poisson`) and a
closed form exists for the resulting chain size PMF, dispatch picks
the closed form automatically. Otherwise the likelihood falls back to
numerical quadrature over `mixing`.

# Examples

```julia
# Poisson offspring with Gamma-distributed rate — uses closed form
# (PoissonGammaChainSize) via dispatch.
o = ClusterMixed(Poisson, Gamma(2.0, 0.4))
loglikelihood(ChainSizes([1, 2, 1, 5]), o)

# NegBin offspring with Gamma-distributed R (fixed k) — no closed form,
# evaluated by quadrature.
o = ClusterMixed(R -> NegBin(R, 0.5), Gamma(2.0, 0.3))
loglikelihood(ChainSizes([1, 1, 3, 2]), o)
```
"""
struct ClusterMixed{F, D <: Distribution}
    build::F
    mixing::D
end

# Convenience: ClusterMixed(Poisson, mixing) maps to the marker builder
ClusterMixed(::Type{Poisson}, m::Distribution) = ClusterMixed(PoissonFamily(), m)

function Base.show(io::IO, o::ClusterMixed)
    build_str = o.build isa PoissonFamily ? "Poisson" : "Function"
    print(io, "ClusterMixed(build=$(build_str), mixing=$(typeof(o.mixing)))")
end

"""
    ChainSizeMixture(build, mixing)

Chain size distribution obtained by integrating the chain size PMF of
`build(θ)` over `mixing`. `logpdf(d, n)` is evaluated by adaptive
Gauss-Kronrod quadrature on the 0.001-0.999 quantile range of
`mixing`.

This is the generic (non-closed-form) chain size distribution for a
[`ClusterMixed`](@ref) offspring. Closed forms, when they exist
(e.g. [`PoissonGammaChainSize`](@ref) for Poisson + Gamma), are
dispatched to directly via `chain_size_distribution`.
"""
struct ChainSizeMixture{F, D <: Distribution} <: DiscreteUnivariateDistribution
    build::F
    mixing::D
end

Distributions.minimum(::ChainSizeMixture) = 1
Distributions.maximum(::ChainSizeMixture) = Inf
Distributions.insupport(::ChainSizeMixture, n::Integer) = n >= 1

function Distributions.logpdf(d::ChainSizeMixture, n::Integer)
    n < 1 && return -Inf
    lo = quantile(d.mixing, 1e-3)
    hi = quantile(d.mixing, 1 - 1e-3)
    integrand = θ -> pdf(chain_size_distribution(d.build(θ)), n) * pdf(d.mixing, θ)
    prob, _ = quadgk(integrand, lo, hi)
    return prob > 0.0 ? log(prob) : -Inf
end

Distributions.pdf(d::ChainSizeMixture, n::Integer) = exp(logpdf(d, n))

"""
    chain_size_distribution(o::ClusterMixed)

Chain size distribution for a cluster-mixed offspring specification.
Dispatches to a closed form when one is known (e.g. Poisson + Gamma
→ [`PoissonGammaChainSize`](@ref)); otherwise returns a
[`ChainSizeMixture`](@ref) that evaluates the PMF by numerical
quadrature at each point.
"""
chain_size_distribution(o::ClusterMixed) = ChainSizeMixture(o.build, o.mixing)

# Closed form: Poisson offspring with Gamma-mixed rate
function chain_size_distribution(o::ClusterMixed{PoissonFamily, <:Gamma})
    k = shape(o.mixing)
    R = k * scale(o.mixing)  # mean of Gamma(shape=k, scale=θ)
    return PoissonGammaChainSize(k, R)
end

function loglikelihood(data::ChainSizes, o::ClusterMixed)
    d = chain_size_distribution(o)
    return sum(logpdf(d, n) for n in data.data)
end

"""
    BranchingProcess(offspring::ClusterMixed; population_size=NoPopulation())

Wrap a cluster-mixed offspring specification in a `BranchingProcess` so
that it composes with other `TransmissionModel` machinery
(e.g. `PartiallyObserved`). Only the likelihood path is supported;
`simulate` is not yet implemented for cluster-level heterogeneity
(per-chain parameter sampling requires engine changes).
"""
function BranchingProcess(offspring::ClusterMixed;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess(offspring, nothing, population_size, 0.0, 1, NoTypeLabels())
end
