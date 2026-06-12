# ── Data wrapper types for unified likelihood/fitting interface ────────

"""
Observed secondary case counts -- the number of individuals each case
infected.  Used with `loglikelihood` and `fit`.

# Examples

```julia
data = OffspringCounts([0, 1, 2, 0, 3, 1, 0])
loglikelihood(data, NegBin(0.8, 0.5))
```
"""
struct OffspringCounts
    data::Vector{Int}
    function OffspringCounts(data::AbstractVector{<:Integer})
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        all(x -> x >= 0, data) || throw(ArgumentError("counts must be non-negative"))
        new(convert(Vector{Int}, data))
    end
end

"""
Observed transmission chain sizes (total number of cases per chain).
Used with `loglikelihood` and `fit`.

Fields:

- `data::Vector{Int}` — observed cluster sizes.
- `seeds::Vector{Int}` — number of independent index cases per cluster
  (default `1`).

By default every cluster is treated as concluded (final-size
likelihood). For real-time data with still-active clusters, pass a
per-cluster `pi` vector of "is finished" probabilities to
`loglikelihood`; see the `pi` kwarg on `loglikelihood(::ChainSizes, ::Distribution)`.

# Examples

```julia
# Standard case: all single-seed.
data = ChainSizes([1, 1, 3, 1, 5])

# Multi-seed clusters.
data = ChainSizes([3, 5, 10, 2]; seeds = [1, 2, 1, 1])
```
"""
struct ChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    function ChainSizes(data::AbstractVector{<:Integer};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)))
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        all(x -> x >= 1, data) || throw(ArgumentError("chain sizes must be ≥ 1"))
        all(s -> s >= 1, seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        new(convert(Vector{Int}, data), convert(Vector{Int}, seeds))
    end
end

"""
Observed transmission chain lengths (number of generations).
Used with `loglikelihood` and `fit`.

# Examples

```julia
data = ChainLengths([0, 1, 0, 2, 1])
loglikelihood(data, Poisson(0.5))
```
"""
struct ChainLengths
    data::Vector{Int}
    function ChainLengths(data::AbstractVector{<:Integer})
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        all(x -> x >= 0, data) || throw(ArgumentError("chain lengths must be ≥ 0"))
        new(convert(Vector{Int}, data))
    end
end
