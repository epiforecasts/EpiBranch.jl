# ── Unified loglikelihood interface ───────────────────────────────────

import Distributions: loglikelihood

"""
    loglikelihood(data::OffspringCounts, offspring::Distribution)

Log-likelihood of observed secondary case counts under a given
offspring distribution.
"""
function loglikelihood(data::OffspringCounts, offspring::Distribution)
    return sum(logpdf(offspring, x) for x in data.data)
end

"""
    loglikelihood(data::OffspringCounts, offspring::AbstractVector{<:Distribution})

Log-likelihood of observed secondary case counts under a per-case offspring
distribution, e.g. `NegBin.(exp.(X * β), k)` for a case-level covariate
matrix `X`. `offspring` must have one distribution per observation in
`data`.

For data that list only cases with at least one secondary case, pass
`truncated.(offspring, 1, Inf)`, or a scalar `truncated(dist, 1, Inf)` to the
single-distribution method.

# Examples

```julia
counts = OffspringCounts([0, 1, 4, 0, 2])
μ = [0.5, 0.8, 3.0, 0.5, 1.2]
loglikelihood(counts, NegBin.(μ, 0.5))
```
"""
function loglikelihood(data::OffspringCounts, offspring::AbstractVector{<:Distribution})
    length(offspring) == length(data.data) || throw(
        ArgumentError(
            "offspring must have the same length as data " *
                "($(length(data.data))); got $(length(offspring))"
        )
    )
    return sum(logpdf(d, x) for (d, x) in zip(offspring, data.data))
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

Data recorded only once a chain reaches a given size are evaluated by passing a
law [`observe`](@ref) has conditioned, as
`observe(chain_size_distribution(offspring), MinimumSize(k))`.
"""
function loglikelihood(
        data::ChainSizes, offspring::Distribution;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    dist = chain_size_distribution(offspring)
    return _chain_size_loglik(dist, data; prob_concluded)
end

# AD-compatible methods: use shared _borel_logpdf / _gammaborel_logpdf
# which accept any numeric type for parameters (ForwardDiff Dual compatible)
function loglikelihood(
        data::ChainSizes, offspring::Poisson{T};
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    ) where {T}
    μ = mean(offspring)
    if prob_concluded === nothing && all(==(1), data.seeds)
        return sum(n -> _borel_logpdf(μ, n), data.data)
    end
    return _chain_size_loglik(Borel(μ), data; prob_concluded)
end

function loglikelihood(
        data::ChainSizes, offspring::NegativeBinomial{T};
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    ) where {T}
    if prob_concluded === nothing && all(==(1), data.seeds)
        return sum(n -> _gammaborel_logpdf(offspring.r, mean(offspring), n), data.data)
    end
    return _chain_size_loglik(GammaBorel(offspring.r, mean(offspring)), data; prob_concluded)
end

"""
    loglikelihood(data::ChainSizes, d::IndexChainSize; prob_concluded = nothing)

Log-likelihood of observed chain sizes under an [`IndexChainSize`](@ref)
law, i.e. an index case with its own offspring distribution. `d` is already
the chain-size law, so this routes directly to [`_chain_size_loglik`](@ref)
rather than through `chain_size_distribution`.
"""
function loglikelihood(
        data::ChainSizes, d::IndexChainSize;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    return _chain_size_loglik(d, data; prob_concluded)
end

"""
    loglikelihood(data::ChainSizes, d::TruncatedChainSize; prob_concluded = nothing)

Log-likelihood of chain sizes recorded only at or above a minimum size, under
the law a [`MinimumSize`](@ref) observation produces. `d` is already the
chain-size law, so this routes directly to [`_chain_size_loglik`](@ref).
"""
function loglikelihood(
        data::ChainSizes, d::TruncatedChainSize;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    return _chain_size_loglik(d, data; prob_concluded)
end

"""
    _chain_size_loglik(dist, data::ChainSizes; prob_concluded = nothing)

Per-cluster chain-size log-likelihood. With `prob_concluded === nothing` every
cluster contributes its concluded PMF
`log P(X = x_i | seeds_i)`; with `prob_concluded::AbstractVector` the mixture
`π_i · P(X = x_i) + (1 − π_i) · P(X ≥ x_i)` is summed.
"""
function _chain_size_loglik(
        dist, data::ChainSizes;
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    if prob_concluded !== nothing && length(prob_concluded) != length(data.data)
        throw(
            ArgumentError(
                "prob_concluded must have the same length as data " *
                    "($(length(data.data))); got $(length(prob_concluded))"
            )
        )
    end
    first_val = _chain_size_logpdf(dist, data.data[1], data.seeds[1])
    total = zero(first_val)
    for i in eachindex(data.data)
        lc = _chain_size_logpdf(dist, data.data[i], data.seeds[i])
        term = if prob_concluded === nothing
            lc
        else
            π_i = prob_concluded[i]
            (0 <= π_i <= 1) ||
                throw(ArgumentError("prob_concluded[$i] = $(π_i) is not in [0, 1]"))
            if π_i >= one(π_i)
                lc
            elseif π_i <= zero(π_i)
                _chain_size_right_tail_logprob(dist, data.data[i], data.seeds[i])
            else
                lo = _chain_size_right_tail_logprob(dist, data.data[i], data.seeds[i])
                _logsumexp2(log(π_i) + lc, log1p(-π_i) + lo)
            end
        end
        total += term
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
    return _chain_length_ll_negbin(data.data, offspring)
end

"""
    loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainLengths, model::TransmissionModel; kwargs...)

Simulation-based log-likelihood under any transmission model, optionally
with interventions.
"""
function _sim_loglikelihood(
        observed, process, column::Symbol, min_val::Int;
        interventions, attributes, progression, observation, recorder,
        sim_opts, n_sim, rng
    )
    states = _simulate_n(
        process, n_sim, sim_opts; interventions, attributes,
        progression, observation, recorder, rng
    )
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

# Chain sizes counted from each case's `:reported` flag, written by
# `Reporting` during the progression pass, by `apply_observation!`
# afterwards for an observation model, or both.
function _reported_chain_sizes(state)
    counts = Dict{Int, Int}()
    for ind in state.individuals
        (is_infected(ind) && get(ind.state, :reported, false)) || continue
        counts[ind.chain_id] = get(counts, ind.chain_id, 0) + 1
    end
    return collect(values(counts))
end

# Observed chain sizes for one simulated state. With `NoObservation` and no
# `Reporting` step in `progression`, detection is not thinned at all, so
# every infected case counts; otherwise only `:reported` cases do.
function _sim_chain_sizes(state, obs::MinimumSize, progression)
    sizes = any(t -> t isa Reporting, progression) ?
        _reported_chain_sizes(state) : chain_statistics(state).size
    return filter(>=(obs.min_size), sizes)
end
function _sim_chain_sizes(state, obs, progression)
    if obs isa NoObservation && !any(t -> t isa Reporting, progression)
        return chain_statistics(state).size
    end
    return _reported_chain_sizes(state)
end

# The smallest size an observation could have recorded, which the empirical
# likelihood bins from.
_observed_min_size(::ObservationModel) = 1
_observed_min_size(o::MinimumSize) = o.min_size

"""
    loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    loglikelihood(data::ChainSizes, spec::ModelSpec; kwargs...)

Log-likelihood of observed chain sizes. With no interventions and a
single-type offspring law, uses the analytical chain-size distribution
thinned by a `Reporting` step's detection probability and transformed by the
observation ([`observe`](@ref)); otherwise simulates and compares the
reported chain sizes against the data, honouring `data.seeds` and
`prob_concluded` as described below. The interventions, progression,
attributes and observation come from the process for a bare model, or from
the spec for a [`ModelSpec`](@ref).

On the simulation path, clusters are drawn grouped by seed count: for each
distinct value in `data.seeds`, `n_sim` simulations are run with that many
index cases, and each simulation's chains are summed to a single cluster
size (as [`chain_size_distribution`](@ref)'s `rand` does), so a multi-seed
cluster is compared against simulated clusters with the same number of
seeds rather than against single-seed ones. `prob_concluded`, when given, is
the real-time per-cluster mixture `π_i · P(X = x_i) + (1 − π_i) · P(X ≥ x_i)`
taken against the simulated clusters' empirical distribution.
"""
function loglikelihood(data::ChainSizes, model::TransmissionModel; kwargs...)
    return _chain_size_model_loglik(
        data, model, interventions(model), attributes(model),
        _progression(model), observation(model), recorder(model); kwargs...
    )
end
function loglikelihood(data::ChainSizes, spec::ModelSpec; kwargs...)
    return _chain_size_model_loglik(
        data, spec.process, interventions(spec), attributes(spec),
        _progression(spec), observation(spec), recorder(spec); kwargs...
    )
end

# Shared core: evaluate `data` against `process` with the modelling layers
# passed explicitly, so a bare process supplies its own and a `ModelSpec`
# supplies the spec's.
function _chain_size_model_loglik(
        data::ChainSizes, process, ivs, attrs, prog, obs, rec;
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        n_sim::Int = 10_000,
        rng::AbstractRNG = Random.default_rng(),
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    if prob_concluded !== nothing && length(prob_concluded) != length(data.data)
        throw(
            ArgumentError(
                "prob_concluded must have the same length as data " *
                    "($(length(data.data))); got $(length(prob_concluded))"
            )
        )
    end
    # Chain size depends only on the offspring law (and the observation), so the
    # analytical form is exact whenever the offspring is the single-type one and
    # nothing thins transmission. Interventions thin it, hence the `ivs` check;
    # progression and attributes don't touch the offspring, so they need none.
    # Structured/depleting models have no single-type offspring, so
    # `single_type_offspring` throws and the evaluation falls through to simulation.
    if isempty(ivs)
        try
            d = observe(
                _fold_reporting(
                    prog, chain_size_distribution(single_type_offspring(process)), obs
                ),
                obs
            )
            return _chain_size_loglik(d, data; prob_concluded)
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
        end
    end
    # No closed form: simulate clusters grouped by seed count, one batch of
    # `n_sim` runs per distinct value in `data.seeds`, so a k-seed cluster is
    # compared against simulated clusters that also start from k index cases
    # rather than against single-seed ones.
    total = 0.0
    for k in sort(unique(data.seeds))
        group = findall(==(k), data.seeds)
        sim_opts = SimOpts(;
            n_initial = k, max_cases, max_generations, max_time, stopping_rules
        )
        states = _simulate_n(
            process, n_sim, sim_opts;
            interventions = ivs, attributes = attrs,
            progression = prog, observation = obs, recorder = rec, rng
        )
        cap = _case_cap(sim_opts)
        sim_values = Int[]
        censored = Bool[]
        for state in states
            # A multi-seed cluster is one simulation: its chains (one per
            # seed) are summed to the cluster's total observed size, rather
            # than reported as separate single-seed sizes.
            vals = _sim_chain_sizes(state, obs, prog)
            isempty(vals) && continue
            push!(sim_values, sum(vals))
            push!(censored, !state.extinct && state.cumulative_cases >= cap)
        end
        group_prob_concluded = prob_concluded === nothing ? nothing : prob_concluded[group]
        total += _empirical_ll(
            data.data[group], sim_values;
            min_val = _observed_min_size(obs), censored, cap,
            prob_concluded = group_prob_concluded
        )
    end
    return total
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
    return _chain_length_model_loglik(
        data, model, interventions(model), attributes(model),
        _progression(model), observation(model), recorder(model); kwargs...
    )
end
function loglikelihood(data::ChainLengths, spec::ModelSpec; kwargs...)
    return _chain_length_model_loglik(
        data, spec.process, interventions(spec), attributes(spec),
        _progression(spec), observation(spec), recorder(spec); kwargs...
    )
end

function _chain_length_model_loglik(
        data::ChainLengths, process, ivs, attrs, prog, obs, rec;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        n_sim::Int = 10_000,
        rng::AbstractRNG = Random.default_rng()
    )
    obs isa NoObservation || throw(
        ArgumentError(
            "loglikelihood(ChainLengths, model with $(typeof(obs))) " *
                "is not defined: per-case detection does not give a well-defined chain " *
                "length. Use ChainSizes, or a model with no observation."
        )
    )
    # Analytical iff no interventions thin transmission, as for chain size above;
    # progression and attributes don't affect the offspring law.
    if isempty(ivs)
        try
            return loglikelihood(data, single_type_offspring(process))
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
        end
    end
    sim_opts = SimOpts(;
        n_initial, max_cases, max_generations, max_time,
        stopping_rules
    )
    return _sim_loglikelihood(
        data.data, process, :length, 0;
        interventions = ivs, attributes = attrs, progression = prog,
        observation = obs, recorder = rec, sim_opts, n_sim, rng
    )
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Empirical log-likelihood with Laplace smoothing and right-censoring.

When `censored` is provided, simulated values flagged as censored contribute
to P(size >= cap) rather than P(size = cap). Observed values at or above
`cap` are evaluated as P(size >= cap).

`prob_concluded`, when given (one value per observation, in `[0, 1]`), mixes
each observation's point mass with its right tail under the same simulated
empirical law, exactly as the analytical [`_chain_size_loglik`](@ref)
mixture does: `π_i · P(X = x_i) + (1 − π_i) · P(X ≥ x_i)`.
"""
function _empirical_ll(
        observed, simulated;
        min_val::Int = 0,
        censored::Vector{Bool} = Bool[],
        cap::Int = typemax(Int),
        prob_concluded::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
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
    denom = n_total + n_unique

    # Right-censored draws are size ≥ cap, so they belong to the tail of
    # every bin at or below the cap.
    log_right_tail(idx) = log(
        (
            sum(@view counts[idx:end]) + (length(counts) - idx + 1) +
                (n_censored > 0 ? n_censored + 1 : 0)
        ) / denom
    )

    ll = 0.0
    for (i, x) in enumerate(observed)
        at_cap = x >= cap && n_censored > 0
        if at_cap
            lc = log((n_censored + 1) / denom)
        else
            idx = x - min_val + 1
            (idx < 1 || idx > length(counts)) && return -Inf
            lc = log((counts[idx] + 1) / denom)
        end
        if prob_concluded === nothing
            ll += lc
            continue
        end
        π_i = prob_concluded[i]
        (0 <= π_i <= 1) ||
            throw(ArgumentError("prob_concluded[$i] = $(π_i) is not in [0, 1]"))
        if π_i >= one(π_i)
            ll += lc
        elseif at_cap
            ll += lc
        else
            lo = log_right_tail(x - min_val + 1)
            ll += π_i <= zero(π_i) ? lo : _logsumexp2(log(π_i) + lc, log1p(-π_i) + lo)
        end
    end
    return ll
end
