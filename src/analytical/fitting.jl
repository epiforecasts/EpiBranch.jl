# ── Unified loglikelihood interface ───────────────────────────────────

import Distributions: loglikelihood

"""
    loglikelihood(data::OffspringCounts, offspring::Distribution)

Log-likelihood of observed secondary case counts under a given
offspring distribution.
"""
function loglikelihood(data::OffspringCounts, offspring::Distribution)
    sum(logpdf(offspring, x) for x in data.data)
end

"""
    loglikelihood(data::ChainSizes, offspring::Distribution; prob_concluded = nothing)

Log-likelihood of observed chain sizes under the analytical chain size
distribution implied by the offspring distribution. Multi-seed clusters
are handled via the `seeds` field of [`ChainSizes`](@ref).

With `prob_concluded === nothing` (default), every cluster is treated as
concluded and the likelihood is the standard final-size sum
`Σ_i log P(X = x_i | seeds_i)`.

With `prob_concluded::AbstractVector` (length `length(data.data)`, values in
`[0, 1]`), cluster `i` contributes the real-time mixture

    L_i = π_i · P(X = x_i | seeds_i) + (1 − π_i) · P(X ≥ x_i | seeds_i)

where `π_i = prob_concluded[i]` is the probability that cluster `i` is
finished (observed size = final size). See `end_of_outbreak_probability` for a
principled `prob_concluded` based on the generation-time distribution.
"""
function loglikelihood(data::ChainSizes, offspring::Distribution;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing)
    dist = chain_size_distribution(offspring)
    return _chain_size_loglik(dist, data; prob_concluded)
end

# AD-compatible methods: use shared _borel_logpdf / _gammaborel_logpdf
# which accept any numeric type for parameters (ForwardDiff Dual compatible)
function loglikelihood(data::ChainSizes, offspring::Poisson{T};
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing) where {T}
    μ = mean(offspring)
    if prob_concluded === nothing && all(==(1), data.seeds)
        return sum(n -> _borel_logpdf(μ, n), data.data)
    end
    return _chain_size_loglik(Borel(μ), data; prob_concluded)
end

function loglikelihood(data::ChainSizes, offspring::NegativeBinomial{T};
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing) where {T}
    if prob_concluded === nothing && all(==(1), data.seeds)
        return sum(n -> _gammaborel_logpdf(offspring.r, mean(offspring), n), data.data)
    end
    return _chain_size_loglik(GammaBorel(offspring.r, mean(offspring)), data; prob_concluded)
end

"""
    _chain_size_loglik(dist, data::ChainSizes; prob_concluded = nothing)

Per-cluster chain-size log-likelihood. With `prob_concluded === nothing` every
cluster contributes its concluded PMF
`log P(X = x_i | seeds_i)`; with `prob_concluded::AbstractVector` the mixture
`π_i · P(X = x_i) + (1 − π_i) · P(X ≥ x_i)` is summed.
"""
function _chain_size_loglik(dist, data::ChainSizes;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing)
    if prob_concluded !== nothing && length(prob_concluded) != length(data.data)
        throw(ArgumentError(
            "prob_concluded must have the same length as data " *
            "($(length(data.data))); got $(length(prob_concluded))"))
    end
    first_val = _chain_size_logpdf(dist, data.data[1], data.seeds[1])
    total = zero(first_val)
    for i in eachindex(data.data)
        lc = _chain_size_logpdf(dist, data.data[i], data.seeds[i])
        if prob_concluded === nothing
            total += lc
            continue
        end
        π_i = prob_concluded[i]
        (0 <= π_i <= 1) ||
            throw(ArgumentError("prob_concluded[$i] = $(π_i) is not in [0, 1]"))
        if π_i >= one(π_i)
            total += lc
        elseif π_i <= zero(π_i)
            total += _chain_size_right_tail_logprob(dist, data.data[i], data.seeds[i])
        else
            lo = _chain_size_right_tail_logprob(dist, data.data[i], data.seeds[i])
            total += _logsumexp2(log(π_i) + lc, log1p(-π_i) + lo)
        end
    end
    return total
end

"""AD-compatible binary log-sum-exp."""
function _logsumexp2(a, b)
    m = max(a, b)
    isinf(m) && return m
    return m + log(exp(a - m) + exp(b - m))
end

"""
    loglikelihood(data::ChainLengths, offspring::Distribution)

Analytical log-likelihood of observed chain lengths. Only defined for
subcritical processes (R < 1).
"""
# AD-compatible chain length: Poisson — PGF iteration on G(s) = exp(λ(s − 1)).
function loglikelihood(data::ChainLengths, offspring::Poisson{T}) where {T}
    λ = mean(offspring)
    λ < 1 ||
        throw(ArgumentError("chain length distribution only defined for subcritical process (λ < 1)"))
    G(s) = exp(λ * (s - one(λ)))
    return _chain_length_ll_pgf(data.data, G, typeof(λ))
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
function _sim_loglikelihood(observed, process, column::Symbol, min_val::Int;
        interventions, attributes, progression, observation, sim_opts, n_sim, rng)
    states = _simulate_n(process, n_sim, sim_opts; interventions, attributes,
        progression, observation, rng)
    sim_values = Int[]
    # Track which simulations hit the case cap (right-censored)
    censored = Bool[]
    cap = _case_cap(sim_opts)
    for state in states
        cs = chain_statistics(state)
        vals = getproperty(cs, column)
        append!(sim_values, vals)
        hit_cap = !state.extinct && state.cumulative_cases >= cap
        append!(censored, fill(hit_cap, length(vals)))
    end
    return _empirical_ll(observed, sim_values; min_val, censored, cap)
end

# Observed chain sizes for one simulated state. Dispatch on the model's
# observation: with `NoObservation` every infected case counts;
# otherwise only the cases the observation marked `:reported`.
_sim_chain_sizes(state, ::NoObservation) = chain_statistics(state).size
function _sim_chain_sizes(state, ::ObservationModel)
    counts = Dict{Int, Int}()
    for ind in state.individuals
        (is_infected(ind) && get(ind.state, :reported, false)) || continue
        counts[ind.chain_id] = get(counts, ind.chain_id, 0) + 1
    end
    return collect(values(counts))
end

"""
    loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainSizes, spec::ModelSpec; kwargs...)

Log-likelihood of observed chain sizes. With no interventions and a
single-type offspring law, uses the analytical chain-size distribution
transformed by the observation ([`observe`](@ref)); otherwise simulates
and compares the reported chain sizes against the data. The interventions,
attributes and observation come from the process for a bare model, or from
the spec for a [`ModelSpec`](@ref).
"""
function loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    _chain_size_model_loglik(data, model, interventions(model), attributes(model),
        _progression(model), observation(model); kwargs...)
end
function loglikelihood(data::ChainSizes, spec::ModelSpec; kwargs...)
    _chain_size_model_loglik(data, spec.process, interventions(spec), attributes(spec),
        _progression(spec), observation(spec); kwargs...)
end

# Shared core: score `data` against `process` with the modelling layers passed
# explicitly, so a bare process supplies its own and a `ModelSpec` supplies the
# spec's.
function _chain_size_model_loglik(data::ChainSizes, process, ivs, attrs, prog, obs;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        n_sim::Int = 10_000,
        rng::AbstractRNG = Random.default_rng())
    # Chain size depends only on the offspring law (and the observation), so the
    # analytical form is exact whenever the offspring is the single-type one and
    # nothing thins transmission. Interventions thin it, hence the `ivs` check;
    # progression and attributes don't touch the offspring, so they need none.
    # Structured/depleting models have no single-type offspring, so
    # `single_type_offspring` throws and the score falls through to simulation.
    if isempty(ivs)
        try
            d = observe(chain_size_distribution(single_type_offspring(process)), obs)
            return _chain_size_loglik(d, data)
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
        end
    end
    sim_opts = SimOpts(; n_initial, max_cases, max_generations, max_time,
        stopping_rules)
    states = _simulate_n(process, n_sim, sim_opts;
        interventions = ivs, attributes = attrs,
        progression = prog, observation = obs, rng)
    sim_values = Int[]
    censored = Bool[]
    cap = _case_cap(sim_opts)
    for state in states
        hit_cap = !state.extinct && state.cumulative_cases >= cap
        for v in _sim_chain_sizes(state, obs)
            v >= 1 || continue
            push!(sim_values, v)
            push!(censored, hit_cap)
        end
    end
    return _empirical_ll(data.data, sim_values; min_val = 1, censored, cap)
end

"""
    loglikelihood(data::ChainLengths, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainLengths, spec::ModelSpec; kwargs...)

Log-likelihood of observed chain lengths. Only defined with no observation
(per-case detection does not give a well-defined chain length); uses the
analytical chain-length distribution with no interventions, otherwise a
simulation estimate. The interventions, attributes and observation come
from the process for a bare model, or from the spec for a [`ModelSpec`](@ref).
"""
function loglikelihood(data::ChainLengths, model::TransmissionModel; kwargs...)
    _chain_length_model_loglik(data, model, interventions(model), attributes(model),
        _progression(model), observation(model); kwargs...)
end
function loglikelihood(data::ChainLengths, spec::ModelSpec; kwargs...)
    _chain_length_model_loglik(data, spec.process, interventions(spec), attributes(spec),
        _progression(spec), observation(spec); kwargs...)
end

function _chain_length_model_loglik(data::ChainLengths, process, ivs, attrs, prog, obs;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        n_sim::Int = 10_000,
        rng::AbstractRNG = Random.default_rng())
    obs isa NoObservation || throw(ArgumentError(
        "loglikelihood(ChainLengths, model with $(typeof(obs))) " *
        "is not defined: per-case detection does not give a well-defined chain " *
        "length. Use ChainSizes, or a model with no observation."))
    # Analytical iff no interventions thin transmission, as for chain size above;
    # progression and attributes don't affect the offspring law.
    if isempty(ivs)
        try
            return loglikelihood(data, single_type_offspring(process))
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
        end
    end
    sim_opts = SimOpts(; n_initial, max_cases, max_generations, max_time,
        stopping_rules)
    return _sim_loglikelihood(data.data, process, :length, 0;
        interventions = ivs, attributes = attrs, progression = prog,
        observation = obs, sim_opts, n_sim, rng)
end

# ── Helpers ──────────────────────────────────────────────────────────

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
