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

# Examples

```julia
data = ChainSizes([1, 1, 3, 1, 5])
fit(data, NegativeBinomial)
```
"""
struct ChainSizes
    data::Vector{Int}
    obs_prob::Float64
    function ChainSizes(data::AbstractVector{<:Integer}; obs_prob::Float64 = 1.0)
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        all(x -> x >= 1, data) || throw(ArgumentError("chain sizes must be ≥ 1"))
        0.0 < obs_prob <= 1.0 || throw(ArgumentError("obs_prob must be in (0, 1]"))
        new(convert(Vector{Int}, data), obs_prob)
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
