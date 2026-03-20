# ── Analytical chain size likelihoods ─────────────────────────────────

"""
    chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution)

Log-likelihood of observed chain sizes `data` under the analytical chain size
distribution implied by the given offspring distribution.

The appropriate analytical distribution is selected based on the offspring type:
- `Poisson` → Borel
- `NegativeBinomial` → Lagrange-inversion formula (individual-level
  overdispersion, NOT outbreak-level Gamma-Borel)
"""
function chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution)
    dist = chain_size_distribution(offspring)
    return sum(logpdf(dist, n) for n in data)
end

"""
    chain_size_ll(data::AbstractVector{<:Integer}, offspring::Distribution,
                  obs_prob::Real)

Log-likelihood with imperfect observation. Each case is observed independently
with probability `obs_prob`. True chain sizes are marginalised over.
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
        obs > max_true && return -Inf
        ll += logsumexp(log_p_true[n] + logpdf(Binomial(n, obs_prob), obs)
                        for n in obs:max_true)
    end
    return ll
end

# ── Simulation-based likelihood ──────────────────────────────────────

"""
    chain_size_ll(data::AbstractVector{<:Integer}, model::TransmissionModel;
                  interventions=[], sim_opts=SimOpts(), n_sim=10_000,
                  rng=Random.default_rng())

Simulation-based log-likelihood of observed chain sizes under any transmission
model, optionally with interventions. An empirical chain size distribution
is built from simulated chains.

Likelihoods can be evaluated under interventions — something
not possible in the R packages because **epichains** and **ringbp** are separate codebases.
"""
function chain_size_ll(data::AbstractVector{<:Integer}, model::TransmissionModel;
                       interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                       attributes::Union{Function, Nothing}=nothing,
                       sim_opts::SimOpts=SimOpts(),
                       n_sim::Int=10_000,
                       rng::AbstractRNG=Random.default_rng())
    states = simulate_batch(model, n_sim; interventions, attributes, sim_opts, rng)

    sim_sizes = Int[]
    for state in states
        cs = chain_statistics(state)
        append!(sim_sizes, cs.size)
    end

    isempty(sim_sizes) && return -Inf

    max_size = max(maximum(data), maximum(sim_sizes))
    counts = zeros(Int, max_size)
    for s in sim_sizes
        counts[s] += 1
    end

    n_total = length(sim_sizes)
    n_unique = max_size

    ll = 0.0
    for obs in data
        if obs > max_size || obs < 1
            return -Inf
        end
        # Add-one (Laplace) smoothing to avoid log(0)
        prob = (counts[obs] + 1) / (n_total + n_unique)
        ll += log(prob)
    end
    return ll
end

"""
    chain_length_ll(data::AbstractVector{<:Integer}, model::TransmissionModel;
                    interventions=[], sim_opts=SimOpts(), n_sim=10_000,
                    rng=Random.default_rng())

Simulation-based log-likelihood of observed chain lengths under any
transmission model, optionally with interventions.
"""
function chain_length_ll(data::AbstractVector{<:Integer}, model::TransmissionModel;
                         interventions::Vector{<:AbstractIntervention}=AbstractIntervention[],
                         attributes::Union{Function, Nothing}=nothing,
                         sim_opts::SimOpts=SimOpts(),
                         n_sim::Int=10_000,
                         rng::AbstractRNG=Random.default_rng())
    states = simulate_batch(model, n_sim; interventions, attributes, sim_opts, rng)

    sim_lengths = Int[]
    for state in states
        cs = chain_statistics(state)
        append!(sim_lengths, cs.length)
    end

    isempty(sim_lengths) && return -Inf

    max_len = max(maximum(data), maximum(sim_lengths))
    counts = zeros(Int, max_len + 1)  # 0-indexed (length 0 is possible)
    for l in sim_lengths
        counts[l + 1] += 1
    end

    n_total = length(sim_lengths)
    n_unique = max_len + 1

    ll = 0.0
    for obs in data
        if obs < 0 || obs > max_len
            return -Inf
        end
        prob = (counts[obs + 1] + 1) / (n_total + n_unique)
        ll += log(prob)
    end
    return ll
end

# ── Analytical chain length likelihoods ──────────────────────────────

"""
    chain_length_ll(data::AbstractVector{<:Integer}, offspring::Poisson)

Analytical log-likelihood of chain lengths for Poisson offspring.
P(length = n) = (1-λ) · λ^(n-1) for subcritical λ < 1.
"""
function chain_length_ll(data::AbstractVector{<:Integer}, offspring::Poisson)
    λ = mean(offspring)
    λ < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (λ < 1)"))
    return sum(log(1.0 - λ) + (n - 1) * log(λ) for n in data)
end

"""
    chain_length_ll(data::AbstractVector{<:Integer}, offspring::NegativeBinomial)

Analytical log-likelihood of chain lengths for NegBin offspring.
PGF iteration is used to compute P(extinct by generation n).
"""
function chain_length_ll(data::AbstractVector{<:Integer}, offspring::NegativeBinomial)
    k = offspring.r
    R = mean(offspring)
    R < 1.0 || throw(ArgumentError("chain length distribution only defined for subcritical process (R < 1)"))

    p = k / (k + R)
    max_len = maximum(data)

    # q_n = P(extinct by generation n)
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
