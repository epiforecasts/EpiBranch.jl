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
    OffspringCounts(infector, infectee; unlinked = 0)

Build offspring counts from a table of infector–infectee pairs:
`infector[i]` transmitted to `infectee[i]`. Only confirmed transmissions
belong here: from [`contacts`](@ref), that means rows filtered to
`infected == true`, since `contacts` also reports exposure events that did
not result in infection. Every case appearing in either vector gets one
count, the number of times it appears in `infector` (zero for a case
identified only as an infectee). `unlinked` adds that many extra cases
with no identified transmission link at all, each with an offspring count
of zero.

# Examples

```julia
# 1 infected 2 and 3; 2 infected 4; two further cases have no known links.
data = OffspringCounts([1, 1, 2], [2, 3, 4]; unlinked = 2)
```
"""
function OffspringCounts(
        infector::AbstractVector{<:Integer}, infectee::AbstractVector{<:Integer};
        unlinked::Integer = 0
    )
    length(infector) == length(infectee) ||
        throw(ArgumentError("infector and infectee must have the same length"))
    unlinked >= 0 || throw(ArgumentError("unlinked must be non-negative"))
    any(infector .== infectee) &&
        throw(ArgumentError("a case cannot be its own infector"))
    allunique(zip(infector, infectee)) ||
        throw(ArgumentError("infector-infectee pairs must be unique"))
    allunique(infectee) ||
        throw(ArgumentError("a case cannot have more than one infector"))

    infector_of = Dict(infectee[i] => infector[i] for i in eachindex(infectee))
    for start in keys(infector_of)
        visited = Set{eltype(infector)}()
        current = start
        while haskey(infector_of, current)
            current in visited &&
                throw(ArgumentError("infector-infectee pairs must not form a transmission cycle"))
            push!(visited, current)
            current = infector_of[current]
        end
    end

    counts = Dict{eltype(infector), Int}(c => 0 for c in union(infector, infectee))
    for i in infector
        counts[i] += 1
    end
    return OffspringCounts(vcat(collect(values(counts)), zeros(Int, unlinked)))
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
per-cluster `prob_concluded` vector of "is finished" probabilities to
`loglikelihood`; see the `prob_concluded` kwarg on
`loglikelihood(::ChainSizes, ::Distribution)`.

Data recorded only once a cluster reaches a given size (for example, only
groups of two or more cases) are scored against a [`MinimumSize`](@ref)
observation, which conditions the likelihood on `N ≥ min_size`.

# Examples

```julia
# Standard case: all single-seed.
data = ChainSizes([1, 1, 3, 1, 5])

# Multi-seed clusters.
data = ChainSizes([3, 5, 10, 2]; seeds = [1, 2, 1, 1])

# Only clusters of two or more cases are recorded.
data = ChainSizes([2, 3, 5, 2])
loglikelihood(data, observe(chain_size_distribution(off), MinimumSize(2)))
```
"""
struct ChainSizes
    data::Vector{Int}
    seeds::Vector{Int}
    function ChainSizes(
            data::AbstractVector{<:Integer};
            seeds::AbstractVector{<:Integer} = ones(Int, length(data))
        )
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        length(seeds) == length(data) ||
            throw(ArgumentError("seeds must have the same length as data"))
        all(s -> s >= 1, seeds) || throw(ArgumentError("seeds must be ≥ 1"))
        all(i -> data[i] >= seeds[i], eachindex(data)) ||
            throw(ArgumentError("chain size must be ≥ number of seeds"))
        return new(convert(Vector{Int}, data), convert(Vector{Int}, seeds))
    end
end

"""
    ChainSizes(; membership, singletons = 0)

Build chain sizes from a vector of cluster memberships, the inverse of
grouping by `chain_id` in [`linelist`](@ref): cases sharing a label in
`membership` belong to the same chain, and the size recorded for that
chain is how many cases share it. `singletons` adds that many extra
chains of size 1, for cases identified as having no cluster at all.

`membership` is keyword-only: it has the same `AbstractVector{<:Integer}`
shape as the already-tallied sizes taken by `ChainSizes(data; seeds)`
above, and Julia dispatches on positional argument types rather than on
which keyword is supplied, so a positional `membership` vector of
integers would be ambiguous with `data`.

# Examples

```julia
# Chain 1 has 2 cases, chain 2 has 1, chain 3 has 3; two further cases
# were not linked to any cluster.
data = ChainSizes(; membership = [1, 1, 2, 3, 3, 3], singletons = 2)
```
"""
function ChainSizes(; membership::AbstractVector{<:Integer}, singletons::Integer = 0)
    singletons >= 0 || throw(ArgumentError("singletons must be non-negative"))
    counts = Dict{eltype(membership), Int}()
    for m in membership
        counts[m] = get(counts, m, 0) + 1
    end
    sizes = vcat(collect(values(counts)), ones(Int, singletons))
    return ChainSizes(sizes)
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
