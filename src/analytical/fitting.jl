# ── Unified loglikelihood interface ───────────────────────────────────

import Distributions: fit, loglikelihood

"""
    loglikelihood(data::OffspringCounts, offspring::Distribution)

Log-likelihood of observed secondary case counts under a given
offspring distribution.
"""
function loglikelihood(data::OffspringCounts, offspring::Distribution)
    sum(logpdf(offspring, x) for x in data.data)
end

"""
    loglikelihood(data::ChainSizes, offspring::Distribution)

Log-likelihood of observed chain sizes under the analytical chain size
distribution implied by the offspring distribution.
"""
function loglikelihood(data::ChainSizes, offspring::Distribution)
    dist = chain_size_distribution(offspring)
    return sum(logpdf(dist, n) for n in data.data)
end

# AD-compatible methods: use shared _borel_logpdf / _gammaborel_logpdf
# which accept any numeric type for parameters (ForwardDiff Dual compatible)
function loglikelihood(data::ChainSizes, offspring::Poisson{T}) where {T}
    μ = min(mean(offspring), one(mean(offspring)))
    return sum(n -> _borel_logpdf(μ, n), data.data)
end

function loglikelihood(data::ChainSizes, offspring::NegativeBinomial{T}) where {T}
    return sum(n -> _gammaborel_logpdf(offspring.r, mean(offspring), n), data.data)
end

"""
    loglikelihood(data::ChainSizes, m::PartiallyObserved)

Log-likelihood of observed chain sizes under a partially observed
branching process. Routes through `chain_size_distribution(m)` so
nested wrappers compose without per-pair specialisations.
"""
function loglikelihood(data::ChainSizes, m::PartiallyObserved)
    d = chain_size_distribution(m)
    return sum(logpdf(d, n) for n in data.data)
end

"""
    loglikelihood(data::ChainLengths, offspring::Distribution)

Analytical log-likelihood of observed chain lengths. Only defined for
subcritical processes (R < 1).
"""
# AD-compatible chain length: Poisson — P(length=n) = (1-λ)λ^n for subcritical
function loglikelihood(data::ChainLengths, offspring::Poisson{T}) where {T}
    λ = mean(offspring)
    λ < 1 ||
        throw(ArgumentError("chain length distribution only defined for subcritical process (λ < 1)"))
    return sum(log(one(λ) - λ) + n * log(λ) for n in data.data)
end

# AD-compatible chain length: NegBin — PGF iteration
function loglikelihood(data::ChainLengths, offspring::NegativeBinomial{T}) where {T}
    _chain_length_ll_negbin(data.data, offspring)
end

"""
    loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainLengths, model::TransmissionModel; kwargs...)

Simulation-based log-likelihood under any transmission model, optionally
with interventions.
"""
function _sim_loglikelihood(observed, model, column::Symbol, min_val::Int;
        interventions, attributes, sim_opts, n_sim, rng)
    states = simulate_batch(model, n_sim; interventions, attributes, sim_opts, rng)
    sim_values = Int[]
    # Track which simulations hit the case cap (right-censored)
    censored = Bool[]
    for state in states
        cs = chain_statistics(state)
        vals = getproperty(cs, column)
        append!(sim_values, vals)
        hit_cap = !state.extinct && state.cumulative_cases >= sim_opts.max_cases
        append!(censored, fill(hit_cap, length(vals)))
    end
    return _empirical_ll(observed, sim_values; min_val, censored,
        cap = sim_opts.max_cases)
end

for (DT, col, mv) in [(:ChainSizes, :size, 1), (:ChainLengths, :length, 0)]
    @eval function loglikelihood(data::$DT, model::TransmissionModel;
            interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
            attributes::Union{Function, NoAttributes} = NoAttributes(),
            sim_opts::SimOpts = SimOpts(),
            n_sim::Int = 10_000,
            rng::AbstractRNG = Random.default_rng())
        # Fast path: use analytical likelihood when no interventions and the
        # offspring distribution has one. Falls through to simulation if the
        # analytical method throws (e.g. unsupported distribution type).
        if isempty(interventions) && model.offspring isa Distribution
            try
                return loglikelihood(data, model.offspring)
            catch e
                e isa MethodError || rethrow()
            end
        end
        _sim_loglikelihood(data.data, model, $(QuoteNode(col)), $mv;
            interventions, attributes, sim_opts, n_sim, rng)
    end
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
        k_range::Tuple{Float64, Float64} = (0.01, 100.0))
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
        R_range::Tuple{Float64, Float64} = (0.001, 0.999))
    neg_ll(R) = -loglikelihood(data, Poisson(R))
    R = _golden_section_min(neg_ll, R_range...)
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
        R_range::Tuple{Float64, Float64} = (0.001, 0.999),
        k_range::Tuple{Float64, Float64} = (0.01, 100.0),
        n_grid::Int = 50)
    _grid_search_negbin(data, R_range, k_range, n_grid)
end

"""
    fit(::Type{Poisson}, data::ChainLengths; R_range=(0.001, 0.999))

Maximum likelihood estimate of a Poisson offspring distribution from
observed chain lengths.
"""
function fit(::Type{Poisson}, data::ChainLengths;
        R_range::Tuple{Float64, Float64} = (0.001, 0.999))
    neg_ll(R) = -loglikelihood(data, Poisson(R))
    R = _golden_section_min(neg_ll, R_range...)
    return Poisson(R)
end

"""
    fit(::Type{NegativeBinomial}, data::ChainLengths;
        R_range=(0.001, 0.999), k_range=(0.01, 100.0))

Maximum likelihood estimate of a NegBin offspring distribution from
observed chain lengths, using the analytical chain length distribution.
Only defined for subcritical R < 1.
"""
function fit(::Type{NegativeBinomial}, data::ChainLengths;
        R_range::Tuple{Float64, Float64} = (0.001, 0.999),
        k_range::Tuple{Float64, Float64} = (0.01, 100.0),
        n_grid::Int = 50)
    _grid_search_negbin(data, R_range, k_range, n_grid)
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Two-pass grid search for NegBin(R, k) MLE given any data type with a loglikelihood method."""
function _grid_search_negbin(data, R_range, k_range, n_grid)
    best_ll = -Inf
    best_R, best_k = 0.5, 1.0

    Rs = range(R_range..., length = n_grid)
    ks = range(k_range..., length = n_grid)

    for R in Rs, k in ks

        ll = loglikelihood(data, NegBin(R, k))
        if ll > best_ll
            best_ll = ll
            best_R, best_k = R, k
        end
    end

    # Refine around best point
    step_R = (Rs[2] - Rs[1]) * 2
    step_k = (ks[2] - ks[1]) * 2
    Rs2 = range(max(R_range[1], best_R - step_R),
        min(R_range[2], best_R + step_R), length = n_grid)
    ks2 = range(max(k_range[1], best_k - step_k),
        min(k_range[2], best_k + step_k), length = n_grid)

    for R in Rs2, k in ks2

        ll = loglikelihood(data, NegBin(R, k))
        if ll > best_ll
            best_ll = ll
            best_R, best_k = R, k
        end
    end

    return NegBin(best_R, best_k)
end

"""Golden section search for minimum of a 1D function on [lo, hi]."""
function _golden_section_min(
        f, lo::Float64, hi::Float64; tol::Float64 = 1e-8, maxiter::Int = 100)
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

"""Empirical log-likelihood with Laplace smoothing and right-censoring.

When `censored` is provided, simulated values flagged as censored contribute
to P(size >= cap) rather than P(size = cap). Observed values at or above
`cap` are evaluated as P(size >= cap).
"""
function _empirical_ll(observed, simulated;
        min_val::Int = 0,
        censored::Vector{Bool} = Bool[],
        cap::Int = typemax(Int))
    isempty(simulated) && return -Inf
    n_total = length(simulated)

    # Count uncensored simulated values
    max_val = max(maximum(observed), maximum(simulated))
    counts = zeros(Int, max_val - min_val + 1)
    n_censored = 0
    for (i, s) in enumerate(simulated)
        if !isempty(censored) && censored[i]
            n_censored += 1
        else
            counts[s - min_val + 1] += 1
        end
    end
    n_unique = length(counts) + (n_censored > 0 ? 1 : 0)

    ll = 0.0
    for obs in observed
        if obs >= cap && n_censored > 0
            # Right-censored: P(size >= cap)
            prob = (n_censored + 1) / (n_total + n_unique)
        else
            idx = obs - min_val + 1
            if idx < 1 || idx > length(counts)
                return -Inf
            end
            prob = (counts[idx] + 1) / (n_total + n_unique)
        end
        ll += log(prob)
    end
    return ll
end
