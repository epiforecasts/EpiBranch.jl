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
        return new(convert(Vector{Int}, data))
    end
end

"""
Observed transmission chain sizes (total number of cases per chain).
Used with `loglikelihood` and `fit`.

Fields:

- `data::Vector{Int}` — observed cluster sizes.
- `seeds::Vector{Int}` — number of independent index cases per cluster
  (default `1`).
- `min_size::Int` — smallest cluster size the data collection could have
  recorded (default `1`, i.e. no truncation).

By default every cluster is treated as concluded (final-size
likelihood). For real-time data with still-active clusters, pass a
per-cluster `prob_concluded` vector of "is finished" probabilities to
`loglikelihood`; see the `prob_concluded` kwarg on
`loglikelihood(::ChainSizes, ::Distribution)`.

With `min_size > 1`, clusters only enter the data once they reach that
size (for example, only groups of two or more cases are recorded). The
likelihood then conditions on `N ≥ min_size`: `P(N = n | N ≥ min_size) =
P(N = n) / P(N ≥ min_size)`.

# Examples

```julia
# Standard case: all single-seed.
data = ChainSizes([1, 1, 3, 1, 5])

# Multi-seed clusters.
data = ChainSizes([3, 5, 10, 2]; seeds = [1, 2, 1, 1])

# Only clusters of two or more cases are recorded.
data = ChainSizes([2, 3, 5, 2]; min_size = 2)
```
"""
struct ChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    min_size::Int
    function ChainSizes(
            data::AbstractVector{<:Integer};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)),
            min_size::Integer = 1
        )
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        min_size >= 1 || throw(ArgumentError("min_size must be ≥ 1, got $min_size"))
        all(x -> x >= min_size, data) ||
            throw(ArgumentError("chain sizes must be ≥ min_size ($min_size)"))
        all(s -> s >= 1, seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        return new(convert(Vector{Int}, data), convert(Vector{Int}, seeds), Int(min_size))
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
        return new(convert(Vector{Int}, data))
    end
end
