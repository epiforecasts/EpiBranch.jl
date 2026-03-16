"""
    chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution)

Log-likelihood of observed chain sizes `data` under the analytical chain size
distribution implied by the given offspring distribution.

Supports Poisson and NegativeBinomial offspring distributions.
"""
function chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution)
    dist = chain_size_distribution(offspring)
    return sum(logpdf(dist, n) for n in data)
end

"""
    chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution,
                  obs_prob::Real)

Log-likelihood with imperfect observation. Each case is observed independently
with probability `obs_prob`. The observed chain size distribution is computed
via numerical convolution.

This approximation sums over possible true chain sizes.
"""
function chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution,
                       obs_prob::Real)
    0.0 < obs_prob <= 1.0 || throw(ArgumentError("obs_prob must be in (0, 1], got $obs_prob"))
    obs_prob == 1.0 && return chain_size_ll(data, offspring)

    dist = chain_size_distribution(offspring)
    max_obs = maximum(data)
    max_true = min(max_obs * 5, 10_000)

    # Precompute log-probabilities of true sizes
    log_p_true = [logpdf(dist, n) for n in 1:max_true]

    ll = 0.0
    for obs in data
        # P(observed = obs) = Σ_n P(true = n) * Binomial(n, obs_prob).pdf(obs)
        log_terms = Float64[]
        for n in obs:max_true
            lp = log_p_true[n] + logpdf(Binomial(n, obs_prob), obs)
            push!(log_terms, lp)
        end
        isempty(log_terms) && return -Inf
        ll += logsumexp(log_terms)
    end
    return ll
end

"""
    chain_length_ll(data::AbstractVector{<:Integer}, offspring::Distribution)

Log-likelihood of observed chain lengths under the analytical chain length
distribution. Chain length is the number of generations.

For Poisson(λ): P(length = n) = Geometric(1 - λ) for λ < 1.
For NegativeBinomial: uses simulation-based approximation via the
extinction probability.
"""
function chain_length_ll(data::AbstractVector{<:Integer}, offspring::Poisson)
    λ = mean(offspring)
    λ < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (λ < 1)"))
    # P(chain length ≥ n) = λ^(n-1), so P(length = n) = (1-λ) * λ^(n-1)
    # This is Geometric with success probability (1-λ), shifted by 1
    return sum(log(1.0 - λ) + (n - 1) * log(λ) for n in data)
end

function chain_length_ll(data::AbstractVector{<:Integer}, offspring::NegativeBinomial)
    k = offspring.r
    R = mean(offspring)
    R < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (R < 1)"))

    # For NegBin offspring, chain length follows a geometric-like distribution
    # P(length = n) ≈ (1 - q_ext_complement) where we use the PGF iteratively
    # Use numerical PGF iteration to compute P(extinct by generation n)
    p = k / (k + R)
    max_len = maximum(data)

    # q_n = P(extinct by generation n), computed via PGF iteration
    q = zeros(max_len + 1)
    q[1] = (p / (1.0 - (1.0 - p) * 0.0))^k  # P(0 offspring) = pgf(0)
    for n in 2:(max_len + 1)
        q[n] = (p / (1.0 - (1.0 - p) * q[n-1]))^k
    end

    # P(chain length = n) = q[n+1] - q[n] (probability of going extinct at generation n)
    # where q[1] = P(extinct at gen 0) = P(0 offspring)
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

"""
    logsumexp(x)

Numerically stable log-sum-exp.
"""
function logsumexp(x::AbstractVector{<:Real})
    isempty(x) && return -Inf
    mx = maximum(x)
    isinf(mx) && return mx
    return mx + log(sum(exp.(x .- mx)))
end
