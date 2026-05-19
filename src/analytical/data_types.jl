# ── Data wrapper types for unified likelihood/fitting interface ────────

"""
Observed secondary case counts -- the number of individuals each case
infected.  Used with `loglikelihood` and `fit`.

# Examples

```julia
data = OffspringCounts([0, 1, 2, 0, 3, 1, 0])
fit(data, NegativeBinomial)
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
- `pi::Vector{Float64}` — per-cluster probability that the cluster is
  finished (i.e. that the observed size is the final size). Default
  `1.0` for every cluster, which is the final-size likelihood. Values
  in `[0, 1]`; intermediate values activate the real-time mixture
  `π · P(X = x | seeds) + (1 − π) · P(X ≥ x | seeds)`.

# Examples

```julia
# Standard case: all single-seed, all concluded.
data = ChainSizes([1, 1, 3, 1, 5])

# Multi-seed clusters, still all concluded.
data = ChainSizes([3, 5, 10, 2]; seeds = [1, 2, 1, 1])

# Real-time mixture: per-cluster finished-probabilities (e.g. from
# `end_of_outbreak_probability` applied to time since the most recent case).
data = ChainSizes([1, 1, 1766, 3]; seeds = [1, 1, 17, 3],
                  pi = [1.0, 1.0, 0.0, 0.99])
```

See also [`loglikelihood(::ChainSizes, ::Distribution)`](@ref) for the
mixture-aware likelihood, and `end_of_outbreak_probability` for a principled `pi`
choice based on the generation-time distribution.
"""
struct ChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    pi::Vector{Float64}
    function ChainSizes(data::AbstractVector{<:Integer};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)),
            pi::AbstractVector{<:Real} = ones(Float64, length(data)))
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        length(pi) == length(data) ||
            throw(ArgumentError("pi must have the same length as data"))
        all(x -> x >= 1, data) || throw(ArgumentError("chain sizes must be ≥ 1"))
        all(s -> s >= 1, seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(p -> 0.0 <= p <= 1.0, pi) ||
            throw(ArgumentError("pi must lie in [0, 1]"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        new(convert(Vector{Int}, data),
            convert(Vector{Int}, seeds),
            convert(Vector{Float64}, pi))
    end
end

"""
Observed transmission chain lengths (number of generations).
Used with `loglikelihood` and `fit`.

# Examples

```julia
data = ChainLengths([0, 1, 0, 2, 1])
fit(data, Poisson)
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
