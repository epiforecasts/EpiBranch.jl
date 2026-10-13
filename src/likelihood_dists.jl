# ── Distribution wrappers for direct Turing use ─────────────────────
#
# Internal wrapper types that turn any model with a defined
# `loglikelihood(::ChainSizes/ChainLengths/OffspringCounts, model)`
# method into something usable with Turing's `~`. Constructed via
# `chain_size_distribution`, `chain_length_distribution`, and
# `offspring_distribution`; not exported on their own. `logpdf` routes
# back to the existing `loglikelihood` methods so analytical fast
# paths and simulation fallbacks are unchanged. `rand(d, n)` simulates
# from the underlying model and projects to the relevant data type.

struct _ChainSizeLaw{M, S, P, K} <:
    Distributions.Distribution{Multivariate, Discrete}
    model::M
    seeds::S
    prob_concluded::P
    kwargs::K
end

function _chain_sizes(data, seeds)
    seeds === nothing && return ChainSizes(data)
    return ChainSizes(data; seeds)
end

function Distributions.logpdf(
        d::_ChainSizeLaw, data::AbstractVector{<:Integer}
    )
    cs = _chain_sizes(data, d.seeds)
    # `loglikelihood(::ChainSizes, model)` itself picks the analytical or
    # simulation-based route and, either way, now honours `prob_concluded`
    # (the simulation-based route from its empirical tail).
    return loglikelihood(cs, d.model; prob_concluded = d.prob_concluded, d.kwargs...)
end

function Distributions.loglikelihood(
        d::_ChainSizeLaw, data::AbstractVector{<:Integer}
    )
    return logpdf(d, data)
end

# `::Int` rather than `::Integer` to disambiguate against
# `rand(::AbstractRNG, ::Sampleable{Multivariate}, ::Int64)`.
function Base.rand(rng::AbstractRNG, d::_ChainSizeLaw, n::Int)
    states = simulate(_underlying_model(d), n; d.kwargs..., rng)
    # Each simulation is one cluster of `n_initial` seeds; its
    # observed size is the total infected across all seed chains, so a
    # multi-seed cluster is summed rather than having every chain but the
    # first silently dropped. `seeds`/`prob_concluded` describe the grouping of
    # *observed* data and have no role in generating new draws.
    return [sum(chain_statistics(s).size) for s in states]
end

_underlying_model(d::_ChainSizeLaw) = d.model

struct _ChainLengthLaw{M, K} <:
    Distributions.Distribution{Multivariate, Discrete}
    model::M
    kwargs::K
end

function Distributions.logpdf(
        d::_ChainLengthLaw, data::AbstractVector{<:Integer}
    )
    return loglikelihood(ChainLengths(data), d.model; d.kwargs...)
end

function Distributions.loglikelihood(
        d::_ChainLengthLaw, data::AbstractVector{<:Integer}
    )
    return logpdf(d, data)
end

function Base.rand(rng::AbstractRNG, d::_ChainLengthLaw, n::Int)
    states = simulate(d.model, n; d.kwargs..., rng)
    # Cluster length is the deepest generation reached across the
    # cluster's seed chains, so multi-seed clusters take the maximum
    # rather than dropping all but the first chain.
    return [maximum(chain_statistics(s).length) for s in states]
end

# ── Public constructors ─────────────────────────────────────────────
#
# Extend the existing `chain_size_distribution` so that — with no
# kwargs and an offspring that has a closed-form distribution — it
# still returns the analytical `Borel` / `GammaBorel`. With `seeds`,
# `prob_concluded`, or anything else passed through, it returns the wrapper.
# `chain_length_distribution` and `offspring_distribution` are new
# entry points.

"""
    chain_size_distribution(model; seeds=nothing, prob_concluded=nothing, kwargs...)
    chain_size_distribution(spec::ModelSpec; seeds=nothing, prob_concluded=nothing, kwargs...)

Distribution over observed chain (cluster) sizes under `model`, the primary
entry point for Bayesian inference on chain-size data with Turing's `~`:

```julia
@model function fit(sizes)
    R ~ Gamma(2, 1)
    sizes ~ chain_size_distribution(BranchingProcess(NegBin(R, 0.5)))
end
```

With no keyword arguments, no interventions and a single-type offspring law,
the analytical chain-size distribution (`Borel` / `GammaBorel`) is returned
directly; otherwise a wrapper that evaluates via `loglikelihood`. Keyword
arguments:

- `seeds`: per-cluster number of index cases, for multi-seed clusters. With
  interventions, clusters are simulated grouped by seed count, one batch of
  runs per distinct value in `seeds`.
- `prob_concluded`: per-cluster probability the cluster is finished (its
  observed size is its final size), for real-time data with ongoing clusters —
  see `end_of_outbreak_probability`. With interventions, the mixture is taken
  against the simulated clusters' empirical P(X ≥ x) rather than an
  analytical tail.
- `n_sim`, …: forwarded to the underlying simulation-based `loglikelihood`
  when the analytical fast path does not apply.
"""
function chain_size_distribution(
        model::TransmissionModel;
        seeds = nothing, prob_concluded = nothing, kwargs...
    )
    # The analytical fast path is only valid when nothing perturbs the
    # bare offspring law. A model that carries interventions must route
    # through the simulation-based wrapper, even with no explicit kwargs,
    # or its interventions would be silently dropped. The model's
    # observation is applied analytically via `observe`.
    if seeds === nothing && prob_concluded === nothing && isempty(kwargs) &&
            isempty(interventions(model))
        base = chain_size_distribution(single_type_offspring(model))
        # Only the `Reporting`-reading step can fail here (e.g. a callable
        # per-individual probability); fall through to the simulation-based
        # wrapper rather than crash the primary `~` entry point.
        # `_chain_size_model_loglik` does the same. A failure from `observe`
        # itself (e.g. a non-scalar `PerCaseObservation`) is a user error,
        # and should propagate rather than be swallowed here.
        folded = try
            _fold_reporting(_progression(model), base, observation(model))
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
            nothing
        end
        folded !== nothing && return observe(folded, observation(model))
    end
    return _ChainSizeLaw(model, seeds, prob_concluded, NamedTuple(kwargs))
end

function chain_size_distribution(
        spec::ModelSpec; seeds = nothing,
        prob_concluded = nothing, kwargs...
    )
    if seeds === nothing && prob_concluded === nothing && isempty(kwargs) &&
            isempty(interventions(spec))
        base = chain_size_distribution(single_type_offspring(spec.process))
        folded = try
            _fold_reporting(_progression(spec), base, observation(spec))
        catch e
            (e isa MethodError || e isa ArgumentError) || rethrow()
            nothing
        end
        folded !== nothing && return observe(folded, observation(spec))
    end
    return _ChainSizeLaw(spec, seeds, prob_concluded, NamedTuple(kwargs))
end

"""
    chain_length_distribution(model; kwargs...)

Distribution over observed chain lengths under `model`. Use with
Turing's `~`:

```julia
@model function fit(data)
    R ~ Beta(1, 1)
    data ~ chain_length_distribution(BranchingProcess(Poisson(R)))
end
```

`kwargs` are forwarded to the underlying
`loglikelihood(::ChainLengths, model)` call (e.g. `n_sim`,
`interventions` for the simulation-based path).
"""
function chain_length_distribution(model::TransmissionModel; kwargs...)
    return _ChainLengthLaw(model, NamedTuple(kwargs))
end

function chain_length_distribution(spec::ModelSpec; kwargs...)
    return _ChainLengthLaw(spec, NamedTuple(kwargs))
end

"""
    offspring_distribution(model)
    offspring_distribution(spec::ModelSpec)

Per-case offspring distribution of `model`. For a `BranchingProcess`
this is the same `Distribution` you passed in as `offspring`.

```julia
@model function fit(data)
    R ~ Beta(1, 1)
    data ~ offspring_distribution(BranchingProcess(Poisson(R)))
end
```

For a `ModelSpec`, or a process that stores its own interventions, each
intervention's [`EpiBranch.analytic_offspring_effect`](@ref) is applied to
the offspring distribution first. If an intervention has no such method, this
throws an error that names the simulation-based alternative.
"""
function offspring_distribution(model::TransmissionModel)
    return _offspring_through_interventions(model, _per_case_offspring(model))
end

function offspring_distribution(spec::ModelSpec)
    return _offspring_through_interventions(spec, _per_case_offspring(spec.process))
end

function _per_case_offspring(model::TransmissionModel)
    off = single_type_offspring(model)
    # `single_type_offspring` may return a spec that has a chain-size law
    # but no per-individual offspring `Distribution` (e.g. `ClusterMixed`,
    # whose θ is shared within a chain, so the counts are not iid). Refuse
    # clearly rather than handing back something `logpdf(·, count)` can't use.
    off isa Distribution || throw(
        ArgumentError(
            "offspring_distribution needs a per-case offspring Distribution, but " *
                "$(typeof(model)) carries a $(typeof(off)) offspring spec with no single " *
                "per-individual law (e.g. ClusterMixed shares θ within a chain). Use " *
                "chain_size_distribution for such models."
        )
    )
    return off
end
