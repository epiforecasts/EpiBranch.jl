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
    ChainSizeClasses(lower, upper, counts)

Observed chain sizes recorded only in size classes: `counts[i]` chains had a
size between `lower[i]` and `upper[i]` cases, inclusive. Use `Inf` as `upper`
for the top class to mean "or more". Passing this to `loglikelihood` with an
offspring distribution or any chain-size law gives the exact likelihood of
the classes,

    Σᵢ counts[i] log P(lower[i] ≤ X ≤ upper[i])

instead of the likelihood of exact sizes invented to stand in for them.

# Examples

```julia
# 120 isolated cases, 40 chains of 2-9 cases, 5 chains of 10 or more
data = ChainSizeClasses([1, 2, 10], [1, 9, Inf], [120, 40, 5])
loglikelihood(data, NegBin(0.5, 0.3))
```
"""
struct ChainSizeClasses
    lower::Vector{Int}
    upper::Vector{Float64}
    counts::Vector{Int}
    function ChainSizeClasses(
            lower::AbstractVector{<:Integer}, upper::AbstractVector{<:Real},
            counts::AbstractVector{<:Integer}
        )
        isempty(lower) && throw(ArgumentError("lower must be non-empty"))
        length(upper) == length(lower) ||
            throw(ArgumentError("upper must have the same length as lower"))
        length(counts) == length(lower) ||
            throw(ArgumentError("counts must have the same length as lower"))
        all(x -> x >= 1, lower) || throw(ArgumentError("lower bounds must be ≥ 1"))
        all(x -> x >= 0, counts) || throw(ArgumentError("counts must be non-negative"))
        all(l <= u for (l, u) in zip(lower, upper)) ||
            throw(ArgumentError("each upper bound must be ≥ its lower bound"))
        all(u -> isinf(u) || isinteger(u), upper) ||
            throw(ArgumentError("upper bounds must be integers or Inf"))
        order = sortperm(lower)
        all(
            lower[order[i + 1]] > upper[order[i]] for i in 1:(length(order) - 1)
        ) || throw(ArgumentError("classes must not overlap"))
        return new(
            convert(Vector{Int}, lower), convert(Vector{Float64}, upper),
            convert(Vector{Int}, counts)
        )
    end
end

"""
    ChainSizeClasses(data)

One class per distinct value in `data`: a compact form of exact chain sizes,
with the same likelihood as `ChainSizes(data)` (one seed per chain) but
evaluating each distinct size once rather than each chain individually. This
is faster for large datasets with many repeated sizes.

# Examples

```julia
data = ChainSizeClasses([1, 1, 3, 1, 5])
loglikelihood(data, NegBin(0.8, 0.5))
```
"""
function ChainSizeClasses(data::AbstractVector{<:Integer})
    isempty(data) && throw(ArgumentError("data must be non-empty"))
    all(x -> x >= 1, data) || throw(ArgumentError("chain sizes must be ≥ 1"))
    sizes = sort(unique(data))
    counts = [count(==(s), data) for s in sizes]
    return ChainSizeClasses(sizes, sizes, counts)
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

"""
    IncidenceCounts(data; seeds = 1)

Observed incidence: the number of new cases in each successive time step
(day, week, or whatever unit `data` is recorded in) of a single outbreak,
starting one step after the `seeds` index cases. Pass it to `loglikelihood`
with an offspring distribution and a generation-time distribution, or with a
model that already has its own, to fit a branching process to a case or
admission time series without simulating it.

- `data`: the incidence count for each step, in calendar order. For a
  multi-type process ([`MultiTypeOffspring`](@ref EpiBranch.MultiTypeOffspring)),
  a vector holds the total over types, and a matrix with one row per step and
  one column per type holds the counts by type.
- `seeds`: the number of index cases introduced at step 0, 1 by default. For
  a multi-type process, a vector with the number of each type.

# Examples

```julia
data = IncidenceCounts([1, 2, 4, 5, 9, 12, 18])
loglikelihood(data, NegBin(1.3, 0.5), LogNormal(1.6, 0.5))

# children and adults, one column each, started by one adult
by_type = IncidenceCounts([0 1; 1 1; 1 3; 2 4]; seeds = [0, 1])
```
"""
struct IncidenceCounts{D <: VecOrMat{Int}, S <: Union{Int, Vector{Int}}}
    data::D
    seeds::S
    function IncidenceCounts(
            data::AbstractVecOrMat{<:Integer};
            seeds::Union{Integer, AbstractVector{<:Integer}} = 1
        )
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        all(x -> x >= 0, data) || throw(ArgumentError("incidence counts must be non-negative"))
        if seeds isa Integer
            seeds >= 1 || throw(ArgumentError("seeds must be ≥ 1"))
            data isa AbstractMatrix && throw(
                ArgumentError(
                    "counts by type (a matrix) need `seeds` by type, a vector with one " *
                        "entry per column"
                )
            )
            seeds = convert(Int, seeds)
        else
            all(>=(0), seeds) && sum(seeds) >= 1 ||
                throw(ArgumentError("seeds by type must be ≥ 0 and sum to at least 1"))
            data isa AbstractMatrix && size(data, 2) != length(seeds) && throw(
                ArgumentError(
                    "counts by type have $(size(data, 2)) columns but `seeds` has " *
                        "$(length(seeds)) entries; give one per type"
                )
            )
            seeds = convert(Vector{Int}, seeds)
        end
        d = data isa AbstractMatrix ? convert(Matrix{Int}, data) : convert(Vector{Int}, data)
        return new{typeof(d), typeof(seeds)}(d, seeds)
    end
end

"""
    SignalSeries(signal, density, data)

One observed series, scored against a [`Signal`](@ref): for each step a
[`SignalData`](@ref) covers, `density(signal(state, interval))` gives the
distribution `data` is scored against at that step. `density` is a function
from a signal's value to any `Distribution`, such as `v -> NegBin(v, k)`;
EpiBranch ships no measurement density of its own, because measurement
choices belong to the data, not the package.

`data` holds one observed value per step.

# Examples

```julia
SignalSeries(TransitionCount(:onset), v -> Poisson(v), reported_onsets)
```
"""
struct SignalSeries{S <: Signal, F}
    signal::S
    density::F
    data::Vector{Float64}
    function SignalSeries(signal::S, density::F, data::AbstractVector{<:Real}) where {S <: Signal, F}
        isempty(data) && throw(ArgumentError("data must be non-empty"))
        return new{S, F}(signal, density, convert(Vector{Float64}, data))
    end
end

"""
    SignalData(steps, series)

Several [`SignalSeries`](@ref) observed over the same outbreak, each scored
against the same simulated path (see `loglikelihood`). `steps` gives the
boundaries of the reporting steps, in calendar time, such as `0:7:70` for ten
weekly steps from day 0; every series' `data` holds one value per step
(`length(steps) - 1`).

Pass it to `loglikelihood` with a model: several series observed together
need one simulated path scored against all of them, because separate `~`
statements in Turing would simulate a fresh outbreak for each. Unlike
[`IncidenceCounts`](@ref) and the chain-size and chain-length data, there is
no closed form here yet, only simulation.

# Examples

```julia
steps = 0.0:7.0:70.0
data = SignalData(
    steps,
    [
        SignalSeries(TransitionCount(:onset), v -> Poisson(v), reported_onsets),
        SignalSeries(Prevalence(:hospitalised), v -> Poisson(v), occupied_beds),
    ]
)
loglikelihood(data, model)
```
"""
struct SignalData{T <: Real}
    steps::Vector{T}
    series::Vector{SignalSeries}
    function SignalData(steps::AbstractVector{<:Real}, series::AbstractVector{<:SignalSeries})
        length(steps) >= 2 || throw(ArgumentError("steps must have at least two boundaries"))
        issorted(steps) || throw(ArgumentError("steps must be increasing"))
        isempty(series) && throw(ArgumentError("series must be non-empty"))
        n = length(steps) - 1
        for s in series
            length(s.data) == n || throw(
                ArgumentError(
                    "every series' data must have one value per step ($n steps " *
                        "from `steps`); got $(length(s.data))"
                )
            )
        end
        return new{eltype(steps)}(collect(steps), collect(series))
    end
end
