# ── Internal likelihood implementations ──────────────────────────────
# These are called by the unified loglikelihood() interface in fitting.jl.

"""Chain size likelihood with imperfect observation."""
function _chain_size_ll_obs(data, offspring, obs_prob)
    dist = chain_size_distribution(offspring)
    max_obs = maximum(data)
    max_true = min(max_obs * 5, 10_000)

    log_p_true = [logpdf(dist, n) for n in 1:max_true]

    ll = 0.0
    for obs in data
        obs > max_true && return -Inf
        ll += logsumexp(log_p_true[n] + logpdf(Binomial(n, obs_prob), obs)
                        for n in obs:max_true)
    end
    return ll
end

"""Analytical chain length likelihood for Poisson offspring."""
function _chain_length_ll_poisson(data, offspring::Poisson)
    λ = mean(offspring)
    λ < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (λ < 1)"))
    return sum(log(1.0 - λ) + (n - 1) * log(λ) for n in data)
end

"""Analytical chain length likelihood for NegBin offspring."""
function _chain_length_ll_negbin(data, offspring::NegativeBinomial)
    k = offspring.r
    R = mean(offspring)
    R < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (R < 1)"))

    p = k / (k + R)
    max_len = maximum(data)

    q = zeros(max_len + 1)
    q[1] = (p / (1.0 - (1.0 - p) * 0.0))^k
    for n in 2:(max_len + 1)
        q[n] = (p / (1.0 - (1.0 - p) * q[n-1]))^k
    end

    ll = 0.0
    for len in data
        if len == 0
            ll += log(q[1])
        elseif len <= max_len
            prob = q[len + 1] - q[len]
            prob <= 0.0 && return -Inf
            ll += log(prob)
        else
            return -Inf
        end
    end
    return ll
end

# ── Utilities ────────────────────────────────────────────────────────

"""
    logsumexp(x)

Numerically stable log-sum-exp. Any iterable is accepted (generators
are collected first to allow two-pass computation).
"""
function logsumexp(x)
    v = x isa AbstractVector ? x : collect(Float64, x)
    isempty(v) && return -Inf
    mx = maximum(v)
    isinf(mx) && return mx
    return mx + log(sum(exp(xi - mx) for xi in v))
end
