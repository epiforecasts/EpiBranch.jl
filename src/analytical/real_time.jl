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
Per-cluster real-time observation. Carries cluster size, seed count,
and timing information: either just the time since the last reported
case (`tau`), or the full set of per-case ages (`case_ages`).

When per-case ages are supplied, the likelihood uses the exact
multi-case formula `π = exp(-R · Σ_i S(τ_i))`. With only `tau`,
it uses the single-most-recent-case approximation
`π ≈ exp(-R · S(τ))`, which agrees with the exact formula when older
cases have wound down (large `τ_i` for `i < x`) and biases towards
extinction when they haven't.

Time units must match the generation-time and reporting-delay
distributions; days are conventional.
"""
struct RealTimeChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    tau::Vector{Float64}
    case_ages::Union{Nothing, Vector{Vector{Float64}}}

    function RealTimeChainSizes(data::AbstractVector{<:Integer},
            tau::AbstractVector{<:Real};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)),
            case_ages::Union{Nothing, AbstractVector} = nothing)
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

        ca = if case_ages === nothing
            nothing
        else
            length(case_ages) == length(data) ||
                throw(ArgumentError("case_ages must have one entry per cluster"))
            for i in eachindex(case_ages)
                length(case_ages[i]) == data[i] || throw(ArgumentError(
                    "case_ages[$i] has $(length(case_ages[i])) ages but cluster size is $(data[i])"))
                all(>=(0), case_ages[i]) ||
                    throw(ArgumentError("case_ages must be ≥ 0"))
                isapprox(minimum(case_ages[i]), tau[i]; atol = 1e-8) ||
                    throw(ArgumentError(
                        "min(case_ages[$i]) = $(minimum(case_ages[i])) must equal tau[$i] = $(tau[i])"))
            end
            [convert(Vector{Float64}, c) for c in case_ages]
        end

        new(convert(Vector{Int}, data),
            convert(Vector{Int}, seeds),
            convert(Vector{Float64}, tau),
            ca)
    end
end

"""
    RealTimeChainSizes(case_ages; seeds = ones(Int, length(case_ages)))

Construct from per-case ages directly. Cluster sizes and `tau` are
inferred (`size = length(case_ages[i])`, `tau = minimum(case_ages[i])`).
Each inner vector lists the time-since-each-case at the snapshot for
one cluster.
"""
function RealTimeChainSizes(case_ages::AbstractVector{<:AbstractVector{<:Real}};
        seeds::AbstractVector{<:Integer} = ones(Int, length(case_ages)))
    sizes = [length(c) for c in case_ages]
    tau = [minimum(c) for c in case_ages]
    return RealTimeChainSizes(sizes, tau; seeds = seeds, case_ages = case_ages)
end

"""
    end_of_outbreak_probability(R, k, generation_time, reporting_delay; tau)

Probability that no further cases will be reported in a cluster, given
`tau` time has elapsed since the last reported case, under a NegBin
offspring distribution with mean `R` and dispersion `k`. With
`k → ∞` this reduces to the Poisson form `exp(-R · S(τ))` used by
Nishiura (2016).

This is a **single-most-recent-case approximation**: it uses only the
time since the last reported case and ignores residual hazard from
older cases in the same cluster. For a multi-case formula that uses
all case times, supply `case_ages` to `RealTimeChainSizes` and the
likelihood will use the per-case form
`∏_i (1 + S(τ_i) · R/k)^(-k)`.
"""
function end_of_outbreak_probability(R, k, generation_time, reporting_delay;
        tau::Real)
    S = _convolved_survival(generation_time, reporting_delay, tau)
    return _negbin_no_more(R, k, S)
end

"""
Per-case "no more reports" probability for a negative-binomial offspring
distribution: marginalise the per-case Poisson rate `λ` over the Gamma
mixing distribution, giving `(1 + S · R/k)^(-k)`. Reduces to
`exp(-R · S)` as `k → ∞` (Poisson limit).
"""
_negbin_no_more(R, k, S) = (1 + S * R / k)^(-k)

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
    loglikelihood(data::RealTimeChainSizes, model::BranchingProcess)

Real-time cluster-size log-likelihood with no reporting delay. Each
cluster contributes a mixture weighted by its end-of-outbreak
probability:

    L_i = π_i · P(X = x_i | s_i)  +  (1 - π_i) · P(X ≥ x_i | s_i)

For a non-trivial reporting delay, wrap the model in [`Reported`](@ref)
and call `loglikelihood(data, Reported(model, delay))`.
"""
function loglikelihood(data::RealTimeChainSizes, model::BranchingProcess)
    _real_time_loglik(data, model, Dirac(0.0))
end

"""
    loglikelihood(data::RealTimeChainSizes, model::Reported)

Real-time cluster-size log-likelihood with the reporting delay carried
by `model`. Equivalent to running the bare `BranchingProcess`
likelihood with `model.delay` folded into `S(τ)` via convolution.
"""
function loglikelihood(data::RealTimeChainSizes, m::Reported{<:BranchingProcess})
    _real_time_loglik(data, m.model, m.delay)
end

function _real_time_loglik(data::RealTimeChainSizes,
        model::BranchingProcess, delay::Distribution)
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
        π = if data.case_ages === nothing
            end_of_outbreak_probability(R, k,
                model.generation_time, delay; tau = data.tau[i])
        else
            _per_case_extinction_probability(R, k, model.generation_time,
                delay, data.case_ages[i])
        end
        log_concluded = _chain_size_logpdf(dist, x, s)
        log_ongoing = _right_tail_logprob(dist, x, s)
        # Numerically stable mixture in log space.
        a = log(π) + log_concluded
        b = log1p(-π) + log_ongoing
        total += logsumexp((a, b))
    end
    return total
end

"""
    _per_case_extinction_probability(R, k, generation_time, reporting_delay, ages)

Multi-case end-of-outbreak probability for one cluster under
NegBin(R, k) offspring. Each case contributes
`(1 + S(τ_i) · R/k)^(-k)`; the cluster's probability is the product.
Reduces to `exp(-R · Σ_i S(τ_i))` as `k → ∞` (Poisson limit).

Marginalises the per-case Poisson rate over its Gamma prior; does not
condition on the number of offspring of each case observed within the
cluster (which would require the transmission tree). The marginal
form is the natural input to a NegBin chain-size likelihood that does
not condition on ancestry either.
"""
function _per_case_extinction_probability(R, k, gt::Distribution,
        delay::Distribution, ages::AbstractVector{<:Real})
    log_p = 0.0
    for τ in ages
        S = _convolved_survival(gt, delay, τ)
        log_p += -k * log1p(S * R / k)
    end
    return exp(log_p)
end
