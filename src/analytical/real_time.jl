# ── Real-time cluster-size likelihood ──────────────────────────────
# When some clusters are still active at the reporting cutoff, each
# cluster contributes a mixture
#
#     L_i = π_i · P(X = x_i | s_i, R, k)
#         + (1 − π_i) · P(X ≥ x_i | s_i, R, k)
#
# where π_i = P(extinct by time of last report | τ_i, R, k, generation
# time, reporting delay). Per-cluster timing is supplied via a
# `Snapshot` on `Observed`. The threshold rule is the degenerate case
# π_i ∈ {0, 1}, encoded as Snapshot entries [Inf] (concluded) and []
# (ongoing).

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
        # τ = Inf collapses to S(∞) = 0 → contribution 0; safe via log1p(0).
        S = isfinite(τ) ? _convolved_survival(gt, delay, τ) : 0.0
        log_p += -k * log1p(S * R / k)
    end
    return exp(log_p)
end

"""
    loglikelihood(data::ChainSizes,
                  model::Observed{<:BranchingProcess, <:PerCaseObservation, <:Snapshot})

Cluster-size log-likelihood with per-cluster observation timing
supplied via [`Snapshot`](@ref). For each cluster:

- empty inner vector — right-tail only `log P(X ≥ x | s)` (binary
  "ongoing" claim, equivalent to the threshold rule's ongoing case).
- `[Inf]` — chain-PMF only `log P(X = x | s)` (concluded).
- one or more finite values — π-mixture with the per-case product
  `π = ∏_j (1 + S(τ_j)·ρ·R/k)^(-k)`.

`ρ = observation.detection_prob` enters via the chain-size PMF
(thinned to `ThinnedChainSize(base, ρ)` when ρ < 1) and via the
direct-offspring approximation `ρ·R` in `π`. See the no-snapshot
method docstring for caveats on the approximation at low ρ.
"""
function loglikelihood(data::ChainSizes,
        m::Observed{<:BranchingProcess, <:PerCaseObservation, <:Snapshot})
    length(m.snapshot) == length(data.data) || throw(ArgumentError(
        "snapshot has $(length(m.snapshot)) clusters but data has " *
        "$(length(data.data))"))
    process = m.process
    ρ = m.observation.detection_prob
    delay = m.observation.delay
    offspring = single_type_offspring(process)
    R = mean(offspring)
    k = offspring isa NegativeBinomial ? offspring.r : 1e6
    R_report = ρ * R
    dist = chain_size_distribution(m)  # ThinnedChainSize when ρ<1, else base.

    # Only the per-case π computation needs a Distribution generation
    # time. Pure all-concluded ([Inf]) and pure all-ongoing ([])
    # snapshots don't, so the GT requirement is checked lazily.
    needs_gt = any(any(isfinite, t) for t in m.snapshot.time_since)
    if needs_gt && !(process.generation_time isa Distribution)
        throw(ArgumentError(
            "Snapshot with finite τ requires a Distribution generation time"))
    end

    total = 0.0
    for i in eachindex(data.data)
        x = data.data[i]
        s = data.seeds[i]
        times = m.snapshot.time_since[i]
        log_concluded = _chain_size_logpdf(dist, x, s)
        log_ongoing = _right_tail_logprob(dist, x, s)
        if isempty(times)
            # Ongoing-only: right-tail likelihood.
            total += log_ongoing
        elseif all(!isfinite, times)
            # All-Inf: π → 1, chain-PMF only (no GT needed).
            total += log_concluded
        else
            π = _per_case_extinction_probability(R_report, k,
                process.generation_time, delay, times)
            a = log(π) + log_concluded
            b = log1p(-π) + log_ongoing
            total += logsumexp((a, b))
        end
    end
    return total
end
