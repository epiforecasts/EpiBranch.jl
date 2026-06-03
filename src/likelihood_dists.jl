# ── Distribution wrappers for direct Turing use ─────────────────────
#
# These wrap a model so `data ~ wrapper(model)` works in Turing without
# needing `Turing.@addlogprob! loglikelihood(...)`. Each wrapper delegates
# to the existing `loglikelihood(data_wrapper, model)` methods, so the
# analytical fast paths and simulation fallbacks are unchanged.

"""
    ChainSizeLikelihood(model; seeds=nothing, pi=nothing, kwargs...)

Distribution wrapper that turns any model with a defined
`loglikelihood(::ChainSizes, model; pi)` into something usable with
Turing's `~`:

```julia
@model function fit(data)
    R ~ Beta(1, 1)
    data ~ ChainSizeLikelihood(Poisson(R))
end
```

`seeds` and `pi`, if supplied, are forwarded to `ChainSizes` and to the
underlying `loglikelihood` call respectively. Extra `kwargs` are
forwarded to `loglikelihood` (e.g. `n_sim`, `interventions` for the
simulation-based path).
"""
struct ChainSizeLikelihood{M, S, P, K} <:
       Distributions.Distribution{Multivariate, Discrete}
    model::M
    seeds::S
    pi::P
    kwargs::K
end

function ChainSizeLikelihood(model; seeds = nothing, pi = nothing, kwargs...)
    ChainSizeLikelihood(model, seeds, pi, NamedTuple(kwargs))
end

function _chain_sizes(data, seeds)
    seeds === nothing && return ChainSizes(data)
    return ChainSizes(data; seeds)
end

function Distributions.logpdf(
        d::ChainSizeLikelihood, data::AbstractVector{<:Integer})
    if d.pi === nothing
        return loglikelihood(_chain_sizes(data, d.seeds), d.model; d.kwargs...)
    end
    return loglikelihood(_chain_sizes(data, d.seeds), d.model; pi = d.pi, d.kwargs...)
end

function Distributions.loglikelihood(
        d::ChainSizeLikelihood, data::AbstractVector{<:Integer})
    return logpdf(d, data)
end

"""
    ChainLengthLikelihood(model; kwargs...)

Distribution wrapper around `loglikelihood(::ChainLengths, model)`. See
[`ChainSizeLikelihood`](@ref) for usage; `kwargs` are forwarded to the
underlying `loglikelihood` call.
"""
struct ChainLengthLikelihood{M, K} <:
       Distributions.Distribution{Multivariate, Discrete}
    model::M
    kwargs::K
end

function ChainLengthLikelihood(model; kwargs...)
    ChainLengthLikelihood(model, NamedTuple(kwargs))
end

function Distributions.logpdf(
        d::ChainLengthLikelihood, data::AbstractVector{<:Integer})
    return loglikelihood(ChainLengths(data), d.model; d.kwargs...)
end

function Distributions.loglikelihood(
        d::ChainLengthLikelihood, data::AbstractVector{<:Integer})
    return logpdf(d, data)
end

"""
    OffspringCountLikelihood(offspring)

Distribution wrapper around `loglikelihood(::OffspringCounts,
offspring)`. The argument is a `Distribution` (typically `Poisson` or
`NegativeBinomial`) — the same thing the underlying method takes.
"""
struct OffspringCountLikelihood{D <: Distribution} <:
       Distributions.Distribution{Multivariate, Discrete}
    offspring::D
end

function Distributions.logpdf(
        d::OffspringCountLikelihood, data::AbstractVector{<:Integer})
    return loglikelihood(OffspringCounts(data), d.offspring)
end

function Distributions.loglikelihood(
        d::OffspringCountLikelihood, data::AbstractVector{<:Integer})
    return logpdf(d, data)
end
