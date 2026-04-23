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

Optional per-observation metadata supports multi-seed and right-censored
clusters, as in Endo, Abbott, Kucharski & Funk (2020,
[doi:10.12688/wellcomeopenres.15842.3](https://doi.org/10.12688/wellcomeopenres.15842.3)):

- `seeds::Vector{Int}` — number of independent index cases per cluster
  (default `1`).
- `concluded::Vector{Bool}` — `true` if the cluster has finished
  transmitting (P(X = x | s)); `false` for ongoing clusters whose
  likelihood becomes the right-tail P(X ≥ x | s). Default `true`.

# Examples

```julia
# Standard case: all single-seed, all concluded.
data = ChainSizes([1, 1, 3, 1, 5])

# Mixed data: some clusters ongoing, some with multiple imports.
data = ChainSizes([3, 5, 10, 2];
    seeds = [1, 2, 1, 1],
    concluded = [true, true, false, true])
```
"""
struct ChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    concluded::Vector{Bool}
    function ChainSizes(data::AbstractVector{<:Integer};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data)),
            concluded::AbstractVector{Bool} = trues(length(data)))
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        length(concluded) == length(data) ||
            throw(ArgumentError("concluded must have the same length as data"))
        all(x -> x >= 1, data) || throw(ArgumentError("chain sizes must be ≥ 1"))
        all(s -> s >= 1, seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        new(convert(Vector{Int}, data),
            convert(Vector{Int}, seeds),
            convert(Vector{Bool}, concluded))
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
