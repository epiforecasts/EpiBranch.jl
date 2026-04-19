# ── Cluster-level heterogeneity ───────────────────────────────────────
# Offspring specification where the offspring distribution parameters
# vary across chains (clusters). For each chain a parameter θ is drawn
# from `mixing`; within a chain the offspring distribution is `build(θ)`.

"""
    ClusterMixed(build, mixing)

Offspring specification with cluster-level heterogeneity: the offspring
distribution's parameters vary across chains. For each chain a value
`θ` is sampled from `mixing`, and the offspring distribution for that
chain is `build(θ)`.

# Examples

```julia
# Cluster-level variation in R for NegBin offspring (fixed k = 0.5)
o = ClusterMixed(R -> NegBin(R, 0.5), Gamma(2.0, 1.0))
loglikelihood(ChainSizes([1, 1, 3, 2]), o)

# Cluster-level variation in λ for Poisson offspring
o = ClusterMixed(λ -> Poisson(λ), Gamma(2.0, 0.4))
loglikelihood(ChainSizes([1, 2, 1, 5]), o)
```
"""
struct ClusterMixed{F, D <: Distribution}
    build::F
    mixing::D
end

function Base.show(io::IO, o::ClusterMixed)
    print(io, "ClusterMixed(build=Function, mixing=$(typeof(o.mixing)))")
end

"""
    loglikelihood(data::ChainSizes, o::ClusterMixed)

Log-likelihood of observed chain sizes under a cluster-mixed offspring
specification. For each observed chain size, the likelihood is
obtained by integrating the chain size PMF (conditional on θ) over the
mixing distribution.

Integration uses adaptive Gauss-Kronrod quadrature on the 0.001-0.999
quantile range of the mixing distribution. For subcritical inference,
the mixing distribution should have support restricted so that all
drawn offspring distributions have mean < 1 (chains with mean ≥ 1
have infinite size with positive probability and contribute no density
to finite observed chain sizes).

Known closed forms (e.g. Poisson offspring with Gamma-mixed rate) are
available via dedicated chain size distributions such as
[`PoissonGammaChainSize`](@ref) and avoid the quadrature cost.
"""
function loglikelihood(data::ChainSizes, o::ClusterMixed)
    lo = quantile(o.mixing, 1e-3)
    hi = quantile(o.mixing, 1 - 1e-3)

    ll = 0.0
    for n in data.data
        integrand = function (θ)
            dist = o.build(θ)
            pdf(chain_size_distribution(dist), n) * pdf(o.mixing, θ)
        end
        prob, _ = quadgk(integrand, lo, hi)
        prob <= 0.0 && return -Inf
        ll += log(prob)
    end
    return ll
end
