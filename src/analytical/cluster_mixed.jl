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
    BranchingProcess(offspring::ClusterMixed, gt; population_size=NoPopulation(), latent_period=0.0)
    BranchingProcess(offspring::ClusterMixed; population_size=NoPopulation())

Wrap a cluster-mixed offspring specification in a `BranchingProcess`.
Simulation samples `θ` from `mixing` once per chain (at the index case)
and reuses it for all individuals in that chain via `parent_id` lookup.
The per-individual offspring draw is `rand(build(θ))`.
"""
function BranchingProcess(offspring::ClusterMixed, gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        latent_period::Real = 0.0)
    BranchingProcess(
        offspring, gt, population_size, Float64(latent_period), 1, NoTypeLabels())
end

function BranchingProcess(offspring::ClusterMixed;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess(offspring, nothing, population_size, 0.0, 1, NoTypeLabels())
end

"""
    _draw_offspring(rng, offspring::ClusterMixed, individual, state)

Draw offspring for an individual under a cluster-mixed specification.
Samples `θ ~ mixing` once per chain, caches it on the index case's
state, and looks it up via `parent_id` for descendants so every
individual in a chain shares the same `θ`.
"""
function _draw_offspring(rng::AbstractRNG, offspring::ClusterMixed,
        individual, state::SimulationState)
    θ = get!(individual.state, :cluster_theta) do
        if individual.parent_id == 0
            rand(rng, offspring.mixing)
        else
            state.individuals[individual.parent_id].state[:cluster_theta]
        end
    end
    return rand(rng, offspring.build(θ))
end
