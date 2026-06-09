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
    pi::P
    kwargs::K
end

function _chain_sizes(data, seeds)
    seeds === nothing && return ChainSizes(data)
    return ChainSizes(data; seeds)
end

function Distributions.logpdf(
        d::_ChainSizeLaw, data::AbstractVector{<:Integer})
    cs = _chain_sizes(data, d.seeds)
    # The pi mixture is only defined against the analytical chain-size
    # distribution. If we were handed a transmission model, route pi
    # through its offspring so the analytical multi-seed/pi method
    # picks it up instead of the simulation-based one (which doesn't
    # accept pi).
    if d.pi === nothing
        return loglikelihood(cs, d.model; d.kwargs...)
    end
    target = d.model isa TransmissionModel ? single_type_offspring(d.model) : d.model
    return loglikelihood(cs, target; pi = d.pi, d.kwargs...)
end

function Distributions.loglikelihood(
        d::_ChainSizeLaw, data::AbstractVector{<:Integer})
    return logpdf(d, data)
end

# `::Int` rather than `::Integer` to disambiguate against
# `rand(::AbstractRNG, ::Sampleable{Multivariate}, ::Int64)`.
function Base.rand(rng::AbstractRNG, d::_ChainSizeLaw, n::Int)
    states = simulate(_underlying_model(d), n; d.kwargs..., rng)
    # Each simulation is one cluster of `sim_opts.n_initial` seeds; its
    # observed size is the total infected across all seed chains, so a
    # multi-seed cluster is summed rather than having every chain but the
    # first silently dropped. `seeds`/`pi` describe the grouping of
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
        d::_ChainLengthLaw, data::AbstractVector{<:Integer})
    return loglikelihood(ChainLengths(data), d.model; d.kwargs...)
end

function Distributions.loglikelihood(
        d::_ChainLengthLaw, data::AbstractVector{<:Integer})
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
# `pi`, or anything else passed through, it returns the wrapper.
# `chain_length_distribution` and `offspring_distribution` are new
# entry points.

function chain_size_distribution(model::TransmissionModel;
        seeds = nothing, pi = nothing, kwargs...)
    if seeds === nothing && pi === nothing && isempty(kwargs)
        return chain_size_distribution(single_type_offspring(model))
    end
    return _ChainSizeLaw(model, seeds, pi, NamedTuple(kwargs))
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

"""
    offspring_distribution(model)

Per-case offspring distribution of `model`. For a `BranchingProcess`
this is the same `Distribution` you passed in as `offspring`.

```julia
@model function fit(data)
    R ~ Beta(1, 1)
    data ~ offspring_distribution(BranchingProcess(Poisson(R)))
end
```
"""
function offspring_distribution(model::TransmissionModel)
    return single_type_offspring(model)
end
