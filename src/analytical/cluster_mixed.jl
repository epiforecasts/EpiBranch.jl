# ── Cluster-level heterogeneity ───────────────────────────────────────
# Offspring specification where the offspring distribution parameters
# vary across chains (clusters). For each chain a parameter θ is drawn
# from `mixing`; within a chain the offspring distribution is `build(θ)`.

"""
Stands for Poisson offspring in `ClusterMixed(Poisson, mixing)`, so that a
Gamma `mixing` distribution gets the exact chain size distribution
([`PoissonGammaChainSize`](@ref EpiBranch.PoissonGammaChainSize)).
"""
struct PoissonFamily end
(::PoissonFamily)(λ) = Poisson(λ)

"""
    ClusterMixed(build, mixing)

Offspring where the reproduction number varies from chain to chain rather
than from case to case, for example between settings or clusters: each chain
draws a value `θ` from `mixing`, and every case in that chain draws its
number of secondary cases from `build(θ)`. Use it in a
[`BranchingProcess`](@ref), or directly with `loglikelihood` and
[`chain_size_distribution`](@ref).

`build` is a function of `θ` returning an offspring distribution, or the
`Poisson` family itself. Poisson offspring with a Gamma-distributed rate has
an exact chain size distribution, which is used automatically; other
combinations are evaluated numerically by integrating over `mixing`.

# Examples

```julia
# Poisson offspring whose rate is Gamma distributed between chains
# (shape 2, mean 0.8): exact chain size distribution
o = ClusterMixed(Poisson, Gamma(2.0, 0.4))
loglikelihood(ChainSizes([1, 2, 1, 5]), o)

# negative binomial offspring (k = 0.5) with R varying between chains:
# evaluated numerically
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
    return print(io, "ClusterMixed(build=$(build_str), mixing=$(typeof(o.mixing)))")
end

"""
    ChainSizeMixture(build, mixing)

Chain size distribution when the reproduction number varies between chains
([`ClusterMixed`](@ref)) and no exact formula is available: the chain size
probabilities of `build(θ)`, averaged over `θ` drawn from `mixing`.
[`chain_size_distribution`](@ref) returns it.

The average is computed by numerical integration over the central 99.8% of
`mixing` (its 0.001 to 0.999 quantiles).
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
    lo = quantile(d.mixing, 1.0e-3)
    hi = quantile(d.mixing, 1 - 1.0e-3)
    integrand = θ -> pdf(chain_size_distribution(d.build(θ)), n) * pdf(d.mixing, θ)
    prob, _ = quadgk(integrand, lo, hi)
    return prob > 0.0 ? log(prob) : -Inf
end

Distributions.pdf(d::ChainSizeMixture, n::Integer) = exp(logpdf(d, n))

"""
    chain_size_distribution(o::ClusterMixed)

Chain size distribution when the reproduction number varies between chains:
exact for Poisson offspring with a Gamma-distributed rate
([`PoissonGammaChainSize`](@ref EpiBranch.PoissonGammaChainSize)), otherwise
computed numerically ([`ChainSizeMixture`](@ref)).
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

A branching process in which the reproduction number varies between chains:
each chain draws `θ` once, from the `mixing` distribution of the
[`ClusterMixed`](@ref) offspring, when its index case is created, and every
case in the chain draws its number of secondary cases from `build(θ)`.
`gt` is the generation time distribution (days).

Add interventions, population characteristics or an observation model with a
[`ModelSpec`](@ref).
"""
function BranchingProcess(
        offspring::ClusterMixed, gt;
        population_size::Union{Int, NoPopulation} = NoPopulation()
    )
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),), population_size, 1,
        NoTypeLabels()
    )
end

function BranchingProcess(
        offspring::ClusterMixed;
        population_size::Union{Int, NoPopulation} = NoPopulation()
    )
    return BranchingProcess((Infectiousness(offspring),), population_size, 1, NoTypeLabels())
end

"""
    draw_offspring(rng, offspring::ClusterMixed, individual, state)

Draw the number of secondary cases of one case when the reproduction number
varies between chains: `θ` is drawn once per chain, at the index case, and
every case in the chain draws from `build(θ)`.
"""
function draw_offspring(
        rng::AbstractRNG, offspring::ClusterMixed,
        individual, state::SimulationState
    )
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
function _mixture_expectation(
        f, mixing::ContinuousUnivariateDistribution;
        atol::Real = 1.0e-10
    )
    value, _ = quadgk(u -> f(quantile(mixing, u)), 0, 1; atol, rtol = sqrt(eps()))
    return value
end
function _mixture_expectation(f, mixing::DiscreteNonParametric; atol::Real = 1.0e-10)
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
function _extinction_at_fixed_law(
        d::DiscreteUnivariateDistribution; tol::Real,
        max_iter::Int
    )
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

Probability that a chain started by a single index case dies out when the
reproduction number varies between chains: the extinction probability of
`build(θ)`, averaged over `θ` drawn from `mixing`. Chains whose `θ` gives a
mean of at most 1 die out with certainty (assuming the number of secondary
cases is not fixed).

`mixing` can be any continuous distribution or a `DiscreteNonParametric`
distribution over a set of values.
"""
function extinction_probability(
        o::ClusterMixed; tol::Real = 1.0e-10,
        max_iter::Int = 1000
    )
    return _mixture_expectation(
        θ -> _extinction_at_fixed_law(o.build(θ); tol, max_iter), o.mixing;
        atol = tol
    )
end
