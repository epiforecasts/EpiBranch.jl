# ── Distribution wrappers for direct Turing use ─────────────────────
#
# Internal wrapper types that turn any model with a defined
# `loglikelihood(::ChainSizes/ChainLengths/OffspringCounts, model)`
# method into something usable with Turing's `~`. Constructed via
# `chain_size_distribution`, `chain_length_distribution`, and
# `offspring_distribution`; not exported on their own. `logpdf` routes
# back to the existing `loglikelihood` methods so analytical fast
# paths and simulation fallbacks are unchanged. `rand(d, n)` simulates
# from the underlying model, applies the same reporting and observation
# `logpdf` scores against, and projects to the relevant data type, so a
# prior or posterior predictive check draws from what the likelihood
# evaluates rather than from the unobserved simulation. Only the keywords
# `simulate` itself accepts are forwarded to it; a likelihood-only keyword
# such as `n_sim` is stored for `logpdf` but dropped here.

# Keywords a stored `kwargs` may carry for the simulation-based `logpdf`
# fallback (such as `n_sim`) but that `simulate` itself does not accept.
_simulate_kwargs(kwargs::NamedTuple, accepted) =
    NamedTuple(k => v for (k, v) in pairs(kwargs) if k in accepted)

const _SIMULATE_ONCE_KWARGS = (
    :n_initial, :initial_cases, :max_cases, :max_generations, :max_time,
    :stopping_rules, :condition, :max_attempts,
)
const _SIMULATE_BATCH_KWARGS = (
    :n_initial, :initial_cases, :max_cases, :max_generations, :max_time,
    :stopping_rules, :parallel,
)

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
    model = _underlying_model(d)
    obs = observation(model)
    prog = _progression(model)
    sim_kwargs = _simulate_kwargs(d.kwargs, _SIMULATE_ONCE_KWARGS)
    # `seeds`/`prob_concluded` describe the grouping of *observed* data and
    # have no role in generating new draws.
    return [_rand_chain_size(rng, model, sim_kwargs, obs, prog) for _ in 1:n]
end

# One cluster's observed size: the same reported-case total that
# `_sim_chain_sizes` sums across the cluster's seed chains and feeds to the
# simulation-based `logpdf` (a multi-seed cluster counts as one draw rather
# than one per seed). A cluster with no reported case has no observed size
# under `logpdf` either, so it is dropped and redrawn rather than counted as
# zero.
function _rand_chain_size(rng, model, sim_kwargs, obs, prog)
    while true
        state = simulate(model; sim_kwargs..., rng)
        vals = _sim_chain_sizes(state, obs, prog)
        isempty(vals) || return sum(vals)
    end
    return
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
    sim_kwargs = _simulate_kwargs(d.kwargs, _SIMULATE_BATCH_KWARGS)
    states = simulate(d.model, n; sim_kwargs..., rng)
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

Distribution of observed chain sizes (the number of cases in a cluster) under
`model`, including its interventions and observation model. Use it to fit a
model to cluster-size data, for example in Turing.jl, where `sizes ~ ...`
says the observed sizes follow this distribution:

```julia
@model function chain_model(sizes)
    R ~ Gamma(2, 1)
    sizes ~ chain_size_distribution(BranchingProcess(NegBin(R, 0.5)))
end
```

Without interventions, and with a single-type offspring distribution, this is
the exact chain-size distribution (`Borel` or `GammaBorel`, adjusted for the
observation model). It stays exact with `prob_concluded`, and with `seeds`
unless the observation model also under-reports: there is no closed form for
several index cases under under-reporting, so that one combination is
estimated from simulated outbreaks and its value moves from one call to the
next. With interventions, or with a multi-type or multi-window model (no
single-type offspring law to fall back on), the probability of the data is
estimated from simulated outbreaks instead. Keyword arguments:

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
    # bare offspring law, and the law itself has a closed form. A model
    # with interventions must route through the simulation-based
    # wrapper, even with no explicit kwargs, or its interventions would be
    # silently dropped; a multi-type or multi-window model has no
    # single-type law to fall back on in the first place. The model's
    # observation is applied analytically via `observe`.
    if seeds === nothing && prob_concluded === nothing && isempty(kwargs) &&
            isempty(interventions(model)) && has_single_type_law(model)
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
            isempty(interventions(spec)) && has_single_type_law(spec)
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

Distribution of observed chain lengths (generations of onward transmission
in a cluster, 0 for a single case) under `model`. Use it to fit a model to
chain-length data, for example in Turing.jl:

```julia
@model function chain_model(lengths)
    R ~ Beta(1, 1)
    lengths ~ chain_length_distribution(BranchingProcess(Poisson(R)))
end
```

Keywords `n_sim`, `n_initial`, `max_cases`, `max_generations`, `max_time`,
`stopping_rules` and `rng` control the simulation used when no exact formula
exists. Interventions come from the model; add them with a
[`ModelSpec`](@ref). See [`ChainLengths`](@ref) for the length convention.
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

Offspring distribution of `model`: the distribution of the number of secondary
cases per case. For a `BranchingProcess` it is the distribution you gave as
`offspring`. Use it to fit a model to counts of secondary cases, for example
in Turing.jl:

```julia
@model function offspring_model(counts)
    R ~ Beta(1, 1)
    counts ~ offspring_distribution(BranchingProcess(Poisson(R)))
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
