# ── Data wrapper types for unified likelihood/fitting interface ────────

"""
    OffspringCounts(data)

Observed numbers of secondary cases: how many people each case infected, one
count per case. Pass it to `loglikelihood` with an offspring distribution,
such as `NegBin(R, k)`, to estimate R and the dispersion k.

# Examples

```julia
data = OffspringCounts([0, 1, 2, 0, 3, 1, 0, 0, 5, 0])
loglikelihood(data, NegBin(0.8, 0.5))

# maximum-likelihood estimate of R for fixed k = 0.5, over a grid
Rs = 0.05:0.05:3.0
R_hat = Rs[argmax([loglikelihood(data, NegBin(R, 0.5)) for R in Rs])]
```

For R and k together, maximise `loglikelihood` with an optimiser such as
Optim.jl, or fit in Turing.jl with [`offspring_distribution`](@ref).
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

Offspring counts from a list of who infected whom: `infector[i]` infected
`infectee[i]`. List only confirmed transmissions; from [`contacts`](@ref),
keep the rows with `infected == true`, since `contacts` also lists exposures
that did not lead to infection. Each case in either vector gets one count,
the number of people it infected (zero for a case that appears only as an
infectee). `unlinked` adds that many cases with no known transmission link,
each counted as infecting nobody.

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
    ChainSizes(data; seeds = ones(Int, length(data)))

Observed chain sizes: the total number of cases in each transmission chain
(cluster). Pass it to `loglikelihood` with an offspring distribution or a
model, or fit in Turing.jl with [`chain_size_distribution`](@ref).

- `data`: the size of each cluster.
- `seeds`: the number of index cases (separate introductions) in each
  cluster, 1 by default.

Every cluster is assumed to be over, so its size is its final size. For
real-time data where some clusters may still grow, pass `prob_concluded` (the
probability each cluster is over) to `loglikelihood`; see
[`end_of_outbreak_probability`](@ref). For data recorded only once a cluster
reaches a given size (for example only clusters of two or more cases), use a
[`MinimumSize`](@ref) observation model.

# Examples

```julia
# one index case per cluster
data = ChainSizes([1, 1, 3, 1, 5])
loglikelihood(data, NegBin(0.8, 0.5))

# clusters with several introductions
data = ChainSizes([3, 5, 10, 2]; seeds = [1, 2, 1, 1])

# only clusters of two or more cases are recorded
data = ChainSizes([2, 3, 5, 2])
loglikelihood(data, observe(chain_size_distribution(NegBin(0.8, 0.5)), MinimumSize(2)))
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

Chain sizes from the cluster each case belongs to, such as the `chain_id`
column of a [`linelist`](@ref): cases with the same label in `membership` are
in the same chain, and each chain's size is the number of cases sharing its
label. `singletons` adds that many chains of one case, for cases not linked
to any cluster.

`membership` must be given by name, `ChainSizes(; membership = ...)`, since
`ChainSizes(x)` reads `x` as the sizes themselves.

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
    ChainLengths(data)

Observed chain lengths: the number of generations of onward transmission in
each chain, 0 for a chain of a single case. Pass it to `loglikelihood` with an
offspring distribution or a model, or fit in Turing.jl with
[`chain_length_distribution`](@ref).

!!! note "Chain length is one less than in epichains"
    epichains' `chain_length` counts generations including the index case, so
    a single-case chain has length 1 there and 0 here. Subtract 1 from
    epichains chain lengths before using them here.

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
