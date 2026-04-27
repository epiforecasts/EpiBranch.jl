# ── Real-time cluster-size likelihood ──────────────────────────────
# Replaces Endo et al. 2020's threshold rule for classifying clusters
# as concluded or ongoing with a continuous end-of-outbreak probability,
# optionally accounting for reporting delay.
#
# Each cluster contributes a mixture
#
#     L_i = π_i · P(X = x_i | s_i, R, k)
#         + (1 − π_i) · P(X ≥ x_i | s_i, R, k)
#
# where π_i = P(extinct by time of last report | τ_i, R, k, generation
# time, reporting delay). The threshold rule is the degenerate case
# π_i ∈ {0, 1}.

"""
Per-cluster real-time observation: an observed chain size, a number of
seed cases, and the time elapsed since the last reported case. Used by
the real-time cluster-size likelihood that replaces threshold-based
classification with a continuous end-of-outbreak probability.

Time units must match the generation-time and reporting-delay
distributions; days are conventional.
"""
struct RealTimeChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    tau::Vector{Float64}    # time since last reported case in each cluster

    function RealTimeChainSizes(data::AbstractVector{<:Integer},
            tau::AbstractVector{<:Real};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)))
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        length(tau) == length(data) ||
            throw(ArgumentError("tau must have the same length as data"))
        all(>=(1), data) || throw(ArgumentError("chain sizes must be ≥ 1"))
        all(>=(1), seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(>=(0), tau) || throw(ArgumentError("tau must be ≥ 0"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        new(convert(Vector{Int}, data),
            convert(Vector{Int}, seeds),
            convert(Vector{Float64}, tau))
    end
end

"""
    end_of_outbreak_probability(R, k, generation_time, reporting_delay; tau)

Probability that no further cases will be reported, given `tau` time has
already elapsed since the last reported case in a cluster with
reproduction number `R`, dispersion `k`, generation-time distribution
`generation_time`, and reporting-delay distribution `reporting_delay`.

Computed as `exp(-R * S(τ))` where `S(τ)` is the survival function of
the convolved generation-time + reporting-delay distribution. This is
a **single-most-recent-case approximation**: it uses only the time since
the last reported case and ignores residual hazard from older cases in
the same cluster. Validation by forward simulation shows the formula
agrees with the empirical extinction probability for `τ` larger than
about one generation interval; at smaller `τ` it overestimates
extinction because clusters with very recent reports tend to contain
more still-active cases. The exact formulation needs all case times
and is left for future work.
"""
function end_of_outbreak_probability(R, k, generation_time, reporting_delay;
        tau::Real)
    # Survival of the convolved (generation-time + reporting-delay)
    # distribution at time τ. The cluster has gone quiet for τ; we want
    # P(no further reports ever | already silent for τ). For a Poisson
    # offspring process this is exp(-R * S(τ)). The negative-binomial
    # extension uses the same hazard form because the offspring process
    # is still a Poisson with random rate when conditioned on the
    # rate-mixing variable.
    S = _convolved_survival(generation_time, reporting_delay, tau)
    return exp(-R * S)
end

"""
Survival function of the convolved generation-time + reporting-delay
distribution at time `t`. Uses Gauss-Kronrod quadrature.
"""
function _convolved_survival(g::Distribution, d::Distribution, t::Real)
    # P(G + D > t) = ∫ pdf(g, u) * ccdf(d, t - u) du
    integrand = u -> pdf(g, u) * ccdf(d, max(t - u, 0.0))
    lo, hi = 0.0, max(t, quantile(g, 0.999))
    val, _ = quadgk(integrand, lo, hi)
    return val + ccdf(g, hi)  # tail of g beyond hi contributes ccdf(d,0)=1
end

# When reporting delay is a point mass at zero (no delay), the survival
# is just ccdf(generation_time, t).
_convolved_survival(g::Distribution, d::Dirac, t::Real) = ccdf(g, max(t - d.value, 0.0))

"""
    loglikelihood(data::RealTimeChainSizes, model::BranchingProcess; reporting_delay)

Real-time cluster-size log-likelihood. Each cluster contributes a
mixture weighted by its end-of-outbreak probability:

    L_i = π_i · P(X = x_i | s_i)  +  (1 - π_i) · P(X ≥ x_i | s_i)

`reporting_delay` defaults to `Dirac(0.0)` (no delay).
"""
function loglikelihood(data::RealTimeChainSizes, model::BranchingProcess;
        reporting_delay::Distribution = Dirac(0.0))
    offspring = _single_type_offspring(model)
    model.generation_time isa Distribution || throw(ArgumentError(
        "real-time likelihood requires a Distribution generation time"))
    R = mean(offspring)
    k = offspring isa NegativeBinomial ? offspring.r : 1e6  # Poisson-like
    dist = chain_size_distribution(offspring)

    total = 0.0
    for i in eachindex(data.data)
        x = data.data[i]
        s = data.seeds[i]
        τ = data.tau[i]
        π = end_of_outbreak_probability(R, k,
            model.generation_time, reporting_delay; tau = τ)
        log_concluded = _chain_size_logpdf(dist, x, s)
        log_ongoing = _right_tail_logprob(dist, x, s)
        # Numerically stable mixture in log space.
        a = log(π) + log_concluded
        b = log1p(-π) + log_ongoing
        total += logsumexp((a, b))
    end
    return total
end
