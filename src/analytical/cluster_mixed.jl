# ── Cluster-level heterogeneity ───────────────────────────────────────
# Offspring specification where the offspring distribution parameters
# vary across chains (clusters). For each chain a parameter θ is drawn
# from `mixing`; within a chain the offspring distribution is `build(θ)`.

"""
Marker type for Poisson offspring. The type parameter makes
`ClusterMixed(Poisson, mixing)` statically known, so dispatch can route
Poisson + Gamma to the closed form `PoissonGammaChainSize`.
"""
struct PoissonFamily end
(::PoissonFamily)(λ) = Poisson(λ)

"""
    ClusterMixed(build, mixing)

Offspring specification with cluster-level heterogeneity: each chain
draws `θ` from `mixing`, and the offspring distribution within that
chain is `build(θ)`.

If `build` is a distribution family type (e.g. `Poisson`) and a closed
form exists for the combination, dispatch uses it automatically. For
everything else the likelihood falls back to numerical quadrature over
`mixing`.

# Examples

```julia
# Poisson offspring with Gamma-distributed rate uses the closed-form
# PoissonGammaChainSize via dispatch.
o = ClusterMixed(Poisson, Gamma(2.0, 0.4))
loglikelihood(ChainSizes([1, 2, 1, 5]), o)

# NegBin offspring with Gamma-distributed R (fixed k) has no closed
# form and is evaluated by quadrature.
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

Chain size distribution defined by integrating the chain size PMF of
`build(θ)` over `mixing`. `logpdf(d, n)` uses adaptive Gauss-Kronrod
quadrature on the 0.001-0.999 quantile range of `mixing`.

This is the generic chain size distribution for a [`ClusterMixed`](@ref)
offspring. When a closed form exists (e.g. [`PoissonGammaChainSize`](@ref)
for Poisson + Gamma), `chain_size_distribution` dispatches to it directly
instead.
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

Return the chain size distribution for a cluster-mixed offspring. Uses
the closed form when one is known (e.g. Poisson + Gamma returns
[`PoissonGammaChainSize`](@ref)); otherwise returns
[`ChainSizeMixture`](@ref), which evaluates the PMF pointwise by
numerical quadrature.
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
    return _chain_size_loglik(d, data)
end

"""
    BranchingProcess(offspring::ClusterMixed, gt; population_size=NoPopulation())
    BranchingProcess(offspring::ClusterMixed; population_size=NoPopulation())

Wrap a cluster-mixed offspring in a `BranchingProcess`. Simulation
samples `θ` once per chain at the index case and reuses it for every
descendant via `parent_id` lookup. The per-individual draw is
`rand(build(θ))`.

The process describes the transmission alone; attach interventions, attributes
or an observation model with a [`ModelSpec`](@ref).
"""
function BranchingProcess(offspring::ClusterMixed, gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size, 1,
        NoTypeLabels())
end

function BranchingProcess(offspring::ClusterMixed;
        population_size::Union{Int, NoPopulation} = NoPopulation())
    BranchingProcess((Infectiousness(offspring),), population_size, 1, NoTypeLabels())
end

"""
    draw_offspring(rng, offspring::ClusterMixed, individual, state)

Draw offspring under a cluster-mixed specification. Samples `θ ~ mixing`
once per chain, caches it on the index case, and looks it up via
`parent_id` for every descendant so all members of a chain share `θ`.
"""
function draw_offspring(rng::AbstractRNG, offspring::ClusterMixed,
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

# ── Reproduction number and extinction probability ───────────────────
#
# Every case in a chain shares the θ drawn for its index case, so a chain is a
# single-type process with offspring law `build(θ)`. Chain-level quantities are
# therefore averages over `mixing` of the single-type results at fixed θ.

# Expectation of `f(θ)` under the mixing distribution. For a continuous law the
# integral is taken over the probability scale, `∫₀¹ f(quantile(mixing, u)) du`,
# which covers an unbounded support without truncating it; Gauss-Kronrod nodes
# never fall on the endpoints, so the quantile stays finite.
function _mixture_expectation(f, mixing::ContinuousUnivariateDistribution;
        atol::Real = 1e-10)
    value, _ = quadgk(u -> f(quantile(mixing, u)), 0, 1; atol, rtol = sqrt(eps()))
    return value
end
function _mixture_expectation(f, mixing::DiscreteNonParametric; atol::Real = 1e-10)
    return sum(p * f(θ) for (θ, p) in zip(support(mixing), probs(mixing)))
end

# Derivative of the offspring PGF, for Newton's method below.
_pgf_derivative(d::Poisson, s) = mean(d) * exp(mean(d) * (s - 1))
function _pgf_derivative(d::NegativeBinomial, s)
    r, p = params(d)
    return r * (1 - p) * p^r / (1 - (1 - p) * s)^(r + 1)
end
_pgf_derivative(d::Dirac, s) = d.value == 0 ? zero(s) : d.value * s^(d.value - 1)
function _pgf_derivative(d::DiscreteUnivariateDistribution, s)
    lo, hi = _series_range(d)
    return sum(x * pdf(d, x) * s^(x - 1) for x in max(lo, 1):hi; init = zero(s))
end

# Extinction probability for a fixed offspring law, by Newton's method on
# g(s) - s from s = 0. The PGF is convex, so the iterates rise monotonically to
# the smallest fixed point, and the rate stays at least linear with ratio 1/2
# however close the law is to critical. Plain fixed-point iteration slows to a
# stall there, and the quadrature over θ evaluates laws on both sides of R = 1.
function _extinction_at_fixed_law(d::DiscreteUnivariateDistribution; tol::Real,
        max_iter::Int)
    R = float(_law_mean(d))
    R <= 1 && return one(R)
    s = zero(R)
    for _ in 1:max_iter
        slope = _pgf_derivative(d, s) - 1
        slope < 0 || return s
        step = (_pgf(d, s) - s) / -slope
        s = min(s + step, one(s))
        abs(step) < tol && return s
    end
    @warn_unconverged_extinction(max_iter, "the reproduction number")
    return s
end

function reproduction_number(o::ClusterMixed)
    return _mixture_expectation(θ -> _law_mean(o.build(θ)), o.mixing)
end
reproduction_number(o::ClusterMixed{PoissonFamily}) = mean(o.mixing)

"""
    extinction_probability(o::ClusterMixed; tol=1e-10, max_iter=1000)

Probability that a chain started by a single index case dies out under
cluster-level heterogeneity. The chain's `θ` is drawn once from `mixing`.
The result averages the extinction probability of `build(θ)` over `mixing`.
At each `θ`, Newton's method finds the smallest fixed point of the offspring
PGF. When the mean of `build(θ)` is at most 1, the function returns exactly 1;
this assumes the offspring count varies.

`mixing` can be any continuous distribution, integrated by adaptive quadrature
on the probability scale, or a `DiscreteNonParametric`, summed over its support.
"""
function extinction_probability(o::ClusterMixed; tol::Real = 1e-10,
        max_iter::Int = 1000)
    return _mixture_expectation(
        θ -> _extinction_at_fixed_law(o.build(θ); tol, max_iter), o.mixing;
        atol = tol)
end
