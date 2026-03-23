# ── Unified loglikelihood interface ───────────────────────────────────

import Distributions: fit, loglikelihood

"""
    loglikelihood(data::OffspringCounts, offspring::Distribution)

Log-likelihood of observed secondary case counts under a given
offspring distribution.
"""
loglikelihood(data::OffspringCounts, offspring::Distribution) =
    sum(logpdf(offspring, x) for x in data.data)

"""
    loglikelihood(data::ChainSizes, offspring::Distribution)

Log-likelihood of observed chain sizes under the analytical chain size
distribution implied by the offspring distribution.

If `data` was constructed with `obs_prob < 1`, imperfect observation
is accounted for by marginalising over true chain sizes.
"""
function loglikelihood(data::ChainSizes, offspring::Distribution)
    data.obs_prob < 1.0 && return _chain_size_ll_obs(data.data, offspring, data.obs_prob)
    dist = chain_size_distribution(offspring)
    return sum(logpdf(dist, n) for n in data.data)
end

"""
    loglikelihood(data::ChainLengths, offspring::Distribution)

Analytical log-likelihood of observed chain lengths. Only defined for
subcritical processes (R < 1).
"""
loglikelihood(data::ChainLengths, offspring::Poisson) =
    _chain_length_ll_poisson(data.data, offspring)

loglikelihood(data::ChainLengths, offspring::NegativeBinomial) =
    _chain_length_ll_negbin(data.data, offspring)

"""
    loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainLengths, model::TransmissionModel; kwargs...)

Simulation-based log-likelihood under any transmission model, optionally
with interventions.
"""
function loglikelihood(data::ChainSizes, model::TransmissionModel;
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
    return _empirical_ll(data.data, sim_sizes, min_val=1)
end

function loglikelihood(data::ChainLengths, model::TransmissionModel;
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
    return _empirical_ll(data.data, sim_lengths, min_val=0)
end

# ── Unified fit interface (extends Distributions.fit) ────────────────

"""
    fit(::Type{Poisson}, data::OffspringCounts)

Maximum likelihood estimate of a Poisson offspring distribution.
The MLE is simply the sample mean.
"""
function fit(::Type{Poisson}, data::OffspringCounts)
    λ = mean(data.data)
    return Poisson(λ)
end

"""
    fit(::Type{NegativeBinomial}, data::OffspringCounts; k_range=(0.01, 100.0))

Maximum likelihood estimate of a Negative Binomial offspring distribution.
The mean `R` is the sample mean; dispersion `k` is found by bisection on
the score equation.
"""
function fit(::Type{NegativeBinomial}, data::OffspringCounts;
             k_range::Tuple{Float64, Float64}=(0.01, 100.0))
    d = data.data
    n = length(d)
    R = mean(d)
    R == 0.0 && return NegBin(0.0, 1.0)

    v = var(d)
    v <= R && return NegBin(R, 1e6)

    function score(k)
        s = n * (digamma(k) - log(k / (k + R)))
        val = -s
        for x in d
            val += digamma(x + k)
        end
        return val
    end

    k_lo, k_hi = k_range
    s_lo = score(k_lo)
    s_hi = score(k_hi)

    if s_lo * s_hi > 0
        return NegBin(R, s_lo > 0 ? k_hi : k_lo)
    end

    for _ in 1:100
        k_mid = (k_lo + k_hi) / 2.0
        s_mid = score(k_mid)
        abs(s_mid) < 1e-10 && break
        (k_hi - k_lo) < 1e-12 && break
        if s_mid * s_lo > 0
            k_lo = k_mid
            s_lo = s_mid
        else
            k_hi = k_mid
        end
    end

    return NegBin(R, (k_lo + k_hi) / 2.0)
end

"""
    fit(::Type{Poisson}, data::ChainSizes; R_range=(0.001, 0.999))

Maximum likelihood estimate of a Poisson offspring distribution from
observed chain sizes, using the Borel chain size distribution.
Only defined for subcritical R < 1.
"""
function fit(::Type{Poisson}, data::ChainSizes;
             R_range::Tuple{Float64, Float64}=(0.001, 0.999))
    neg_ll(R) = -loglikelihood(data, Poisson(R))
    R = _bisect_min(neg_ll, R_range...)
    return Poisson(R)
end

"""
    fit(::Type{NegativeBinomial}, data::ChainSizes;
        R_range=(0.001, 0.999), k_range=(0.01, 100.0))

Maximum likelihood estimate of a NegBin offspring distribution from
observed chain sizes, using the GammaBorel chain size distribution.
Only defined for subcritical R < 1.
"""
function fit(::Type{NegativeBinomial}, data::ChainSizes;
             R_range::Tuple{Float64, Float64}=(0.001, 0.999),
             k_range::Tuple{Float64, Float64}=(0.01, 100.0),
             n_grid::Int=50)
    best_ll = -Inf
    best_R, best_k = 0.5, 1.0

    Rs = range(R_range..., length=n_grid)
    ks = range(k_range..., length=n_grid)

    for R in Rs, k in ks
        ll = loglikelihood(data, NegBin(R, k))
        if ll > best_ll
            best_ll = ll
            best_R, best_k = R, k
        end
    end

    step_R = (Rs[2] - Rs[1]) * 2
    step_k = (ks[2] - ks[1]) * 2
    Rs2 = range(max(R_range[1], best_R - step_R),
                min(R_range[2], best_R + step_R), length=n_grid)
    ks2 = range(max(k_range[1], best_k - step_k),
                min(k_range[2], best_k + step_k), length=n_grid)

    for R in Rs2, k in ks2
        ll = loglikelihood(data, NegBin(R, k))
        if ll > best_ll
            best_ll = ll
            best_R, best_k = R, k
        end
    end

    return NegBin(best_R, best_k)
end

"""
    fit(::Type{Poisson}, data::ChainLengths; R_range=(0.001, 0.999))

Maximum likelihood estimate of a Poisson offspring distribution from
observed chain lengths.
"""
function fit(::Type{Poisson}, data::ChainLengths;
             R_range::Tuple{Float64, Float64}=(0.001, 0.999))
    neg_ll(R) = -loglikelihood(data, Poisson(R))
    R = _bisect_min(neg_ll, R_range...)
    return Poisson(R)
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Golden section search for minimum of a 1D function on [lo, hi]."""
function _bisect_min(f, lo::Float64, hi::Float64; tol::Float64=1e-8, maxiter::Int=100)
    φ = (sqrt(5.0) - 1.0) / 2.0
    c = hi - φ * (hi - lo)
    d = lo + φ * (hi - lo)
    for _ in 1:maxiter
        (hi - lo) < tol && break
        if f(c) < f(d)
            hi = d
        else
            lo = c
        end
        c = hi - φ * (hi - lo)
        d = lo + φ * (hi - lo)
    end
    return (lo + hi) / 2.0
end

"""Empirical log-likelihood with Laplace smoothing."""
function _empirical_ll(observed, simulated; min_val::Int=0)
    isempty(simulated) && return -Inf
    max_val = max(maximum(observed), maximum(simulated))
    counts = zeros(Int, max_val - min_val + 1)
    for s in simulated
        counts[s - min_val + 1] += 1
    end
    n_total = length(simulated)
    n_unique = length(counts)

    ll = 0.0
    for obs in observed
        idx = obs - min_val + 1
        if idx < 1 || idx > length(counts)
            return -Inf
        end
        prob = (counts[idx] + 1) / (n_total + n_unique)
        ll += log(prob)
    end
    return ll
end
