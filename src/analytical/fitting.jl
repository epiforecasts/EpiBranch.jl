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
distribution implied by the offspring distribution. Handles multi-seed
and right-censored observations when `data.seeds` or `data.concluded`
are non-default (see [`ChainSizes`](@ref)).
"""
function loglikelihood(data::ChainSizes, offspring::Distribution)
    dist = chain_size_distribution(offspring)
    return _chain_size_loglik(dist, data)
end

# AD-compatible methods: use shared _borel_logpdf / _gammaborel_logpdf
# which accept any numeric type for parameters (ForwardDiff Dual compatible)
function loglikelihood(data::ChainSizes, offspring::Poisson{T}) where {T}
    μ = min(mean(offspring), one(mean(offspring)))
    _has_default_metadata(data) &&
        return sum(n -> _borel_logpdf(μ, n), data.data)
    return _chain_size_loglik(Borel(μ), data)
end

function loglikelihood(data::ChainSizes, offspring::NegativeBinomial{T}) where {T}
    _has_default_metadata(data) &&
        return sum(n -> _gammaborel_logpdf(offspring.r, mean(offspring), n), data.data)
    return _chain_size_loglik(GammaBorel(offspring.r, mean(offspring)), data)
end

"""True when every observation is single-seed and concluded."""
_has_default_metadata(data::ChainSizes) = all(==(1), data.seeds) && all(data.concluded)

"""
    _chain_size_loglik(dist, data::ChainSizes)

Per-observation chain-size log-likelihood. Concluded clusters use
`_chain_size_logpdf(dist, x, s)`; ongoing clusters use `log P(X ≥ x | s)`
via the right-tail helper.
"""
function _chain_size_loglik(dist, data::ChainSizes)
    first_val = _chain_size_logpdf(dist, data.data[1], data.seeds[1])
    total = zero(first_val)
    for i in eachindex(data.data)
        x = data.data[i]
        s = data.seeds[i]
        total += data.concluded[i] ?
                 _chain_size_logpdf(dist, x, s) :
                 _right_tail_logprob(dist, x, s)
    end
    return total
end

"""
    _right_tail_logprob(dist, x, s)

`log P(X ≥ x | s)` for a chain size distribution. Computed as
`log(1 - Σ_{m=s}^{x-1} pdf(m | s))`. Returns `0.0` when `x ≤ s`
(support starts at `s`). Used for ongoing outbreaks (Endo et al. 2020).
"""
function _right_tail_logprob(dist, x::Integer, s::Integer)
    x <= s && return 0.0
    tail = 1.0
    for m in s:(x - 1)
        tail -= exp(_chain_size_logpdf(dist, m, s))
    end
    tail > 0 ? log(tail) : -Inf
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

"""
    loglikelihood(data::ChainSizes, m::Surveilled{<:Any, <:PerCaseObservation}; kwargs...)

Log-likelihood under a per-case-observed model. With no interventions,
routes through the analytical `chain_size_distribution(m)`. When
interventions are supplied, simulates the wrapped process model,
applies per-case Binomial thinning, and compares against the observed
data via the empirical likelihood.
"""
function loglikelihood(data::ChainSizes,
        m::Surveilled{<:Any, <:PerCaseObservation};
        interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
        attributes::Union{Function, NoAttributes} = NoAttributes(),
        sim_opts::SimOpts = SimOpts(),
        n_sim::Int = 10_000,
        rng::AbstractRNG = Random.default_rng())
    if isempty(interventions)
        try
            d = chain_size_distribution(m)
            return _chain_size_loglik(d, data)
        catch e
            e isa MethodError || rethrow()
        end
    end
    p = m.observation.detection_prob
    states = simulate_batch(m.process, n_sim;
        interventions, attributes, sim_opts, rng)
    sim_values = Int[]
    censored = Bool[]
    for state in states
        cs = chain_statistics(state)
        for true_size in cs.size
            obs = rand(rng, Binomial(true_size, p))
            obs >= 1 || continue
            push!(sim_values, obs)
            hit_cap = !state.extinct &&
                      state.cumulative_cases >= sim_opts.max_cases
            push!(censored, hit_cap)
        end
    end
    return _empirical_ll(data.data, sim_values; min_val = 1, censored,
        cap = sim_opts.max_cases)
end

function loglikelihood(::ChainLengths,
        ::Surveilled{<:Any, <:PerCaseObservation}; kwargs...)
    throw(ArgumentError(
        "loglikelihood(ChainLengths, Surveilled{..., PerCaseObservation}) is not defined: per-case detection does not translate to a well-defined chain length distribution. Use ChainSizes or evaluate on the bare process model."))
end

for (DT, col, mv) in [(:ChainSizes, :size, 1), (:ChainLengths, :length, 0)]
    @eval function loglikelihood(data::$DT, model::TransmissionModel;
            interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
            attributes::Union{Function, NoAttributes} = NoAttributes(),
            sim_opts::SimOpts = SimOpts(),
            n_sim::Int = 10_000,
            rng::AbstractRNG = Random.default_rng())
        # Fast path: use analytical likelihood when no interventions and
        # the offspring specification has one. Falls through to simulation
        # if no analytical method is defined.
        if isempty(interventions) && hasproperty(model, :offspring)
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
#
# For raw offspring counts, `Distributions.fit(Poisson, x)` already gives
# the MLE; for NegBin offspring counts, plug `loglikelihood(OffspringCounts(x), NegBin(R, k))`
# into Optim.jl or Turing's `maximum_likelihood`. EpiBranch's `fit` is
# scoped to data types whose likelihoods are not in Distributions.jl:
# `ChainSizes` and `ChainLengths`.

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
        k_range::Tuple{Float64, Float64} = (0.01, 100.0))
    _coord_descent_negbin(data, R_range, k_range)
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
        k_range::Tuple{Float64, Float64} = (0.01, 100.0))
    _coord_descent_negbin(data, R_range, k_range)
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Coordinate descent on NegBin(R, k) MLE, using golden section on each axis."""
function _coord_descent_negbin(data, R_range, k_range;
        tol::Float64 = 1e-6, maxiter::Int = 50)
    R = (R_range[1] + R_range[2]) / 2.0
    k = sqrt(k_range[1] * k_range[2])
    R_prev, k_prev = R, k
    for iter in 1:maxiter
        R = _golden_section_min(r -> -loglikelihood(data, NegBin(r, k)), R_range...)
        k = _golden_section_min(κ -> -loglikelihood(data, NegBin(R, κ)), k_range...)
        if iter > 1 &&
           abs(R - R_prev) < tol &&
           abs(k - k_prev) / max(k, 1.0) < tol
            break
        end
        R_prev, k_prev = R, k
    end
    return NegBin(R, k)
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
