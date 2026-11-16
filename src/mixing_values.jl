# ── Mixing ───────────────────────────────────────────────────────────
#
# How the groups of a population meet, given without the group sizes. Each
# engine derives what it needs from the mixing value and the population: the
# offspring means of a branching process, or the rate per pair of a finite
# population.

"""
    Mixing

Parent type of the ways to give how groups mix: [`FrequencyDependent`](@ref)
and [`DensityDependent`](@ref). A [`BranchingProcess`](@ref) and a
[`MixingProcess`](@ref) take one, with the [`Population`](@ref), and each
derives what it needs:

| | Branching process (offspring mean of a type-`j` case's type-`i` cases) | Finite population (rate per pair) |
|---|---|---|
| `FrequencyDependent(c)` | `c[j,i] * q * D`, no sizes needed | `c[j,i] * q / N_i` |
| `DensityDependent(r)` | `r[j,i] * N_i * q * D`, needs the sizes | `r[j,i] * q` |

for `q` the probability a contact transmits, `D` the infectious period and
`N_i` the size of group `i`. Group sizes therefore enter only where the model
uses them: under frequency dependence for depletion alone, and under density
dependence also for the reproduction number.
"""
abstract type Mixing end

"""Placeholder meaning no mixing value is given."""
struct NoMixing end

"""
    FrequencyDependent(contacts; groups = NoTypeLabels(), labels = NoTypeLabels())
    FrequencyDependent(contacts::StratifiedContacts)

Mixing given as contacts per person per day, as a contact survey gives it:
`contacts[j, i]` is the mean number of group-`i` contacts a group-`j` person
makes per day, whatever the size of group `i`. Contacts per person are the
same in a larger population, so a person's rate of contact with each member
of group `i` falls as `contacts[j, i] / N_i`. See [`Mixing`](@ref) for what
each engine derives from it.

`groups` gives each group's value of the `mixing_by` attributes of a
[`MixingProcess`](@ref), one tuple per group, and `labels` names the types of
a [`BranchingProcess`](@ref). A [`StratifiedContacts`](@ref) supplies both.

The matrix is used as given and is not made reciprocal
(`contacts[j, i] * N[j] == contacts[i, j] * N[i]`).

# Examples
```julia
mixing = FrequencyDependent([10.0 1.0; 4.0 2.0])  # row = respondent, column = contact
BranchingProcess(mixing, 0.05, 5.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5))
```
"""
struct FrequencyDependent{G, L} <: Mixing
    matrix::Matrix{Float64}
    groups::G
    labels::L
    function FrequencyDependent(contacts::AbstractMatrix{<:Real}, groups, labels)
        return new{typeof(groups), typeof(labels)}(
            _checked_mixing(contacts, groups, labels, "contacts")..., labels
        )
    end
end

"""
    DensityDependent(rates; groups = NoTypeLabels(), labels = NoTypeLabels())

Mixing given as a contact rate per pair of people: `rates[j, i]` is the rate
per day at which one group-`j` person contacts one given group-`i` person, so
they meet more often in a larger group. See [`Mixing`](@ref) for what each
engine derives from it. A branching process needs the group sizes for it even
without depletion, because the number of people one case contacts is the rate
times the size of the group.

`groups` and `labels` are as for [`FrequencyDependent`](@ref).

# Examples
```julia
mixing = DensityDependent([0.002 0.0004; 0.0004 0.0002])
BranchingProcess(mixing, 0.05, 5.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5);
    population = Population([1500, 3500]))
```
"""
struct DensityDependent{G, L} <: Mixing
    matrix::Matrix{Float64}
    groups::G
    labels::L
    function DensityDependent(rates::AbstractMatrix{<:Real}, groups, labels)
        return new{typeof(groups), typeof(labels)}(
            _checked_mixing(rates, groups, labels, "rates")..., labels
        )
    end
end

function FrequencyDependent(
        contacts::AbstractMatrix{<:Real}; groups = NoTypeLabels(), labels = NoTypeLabels()
    )
    return FrequencyDependent(contacts, groups, labels)
end
function DensityDependent(
        rates::AbstractMatrix{<:Real}; groups = NoTypeLabels(), labels = NoTypeLabels()
    )
    return DensityDependent(rates, groups, labels)
end

function _checked_mixing(matrix, groups, labels, name)
    n = size(matrix, 1)
    size(matrix, 2) == n ||
        throw(ArgumentError("$name must be square, got $(size(matrix))"))
    all(x -> isfinite(x) && x >= 0, matrix) ||
        throw(ArgumentError("$name must be finite and not negative"))
    for (what, x) in (("groups", groups), ("labels", labels))
        x isa NoTypeLabels && continue
        length(x) == n || throw(ArgumentError("$what has $(length(x)) entries for $n groups"))
    end
    groups isa NoTypeLabels || allunique(groups) ||
        throw(ArgumentError("groups are not unique"))
    return Matrix{Float64}(matrix), groups
end

function Base.show(io::IO, m::Mixing)
    return print(io, nameof(typeof(m)), "(", size(m.matrix, 1), " groups)")
end

# The group tuples a mixing value carries, or none.
_mixing_groups(::NoMixing) = NoTypeLabels()
_mixing_groups(m::Mixing) = m.groups isa NoTypeLabels ? NoTypeLabels() : Tuple[g isa Tuple ? g : (g,) for g in m.groups]

# The probability of transmission per infectee group, as a vector.
function _per_group_q(q::Real, n::Int)
    0 <= q <= 1 || throw(ArgumentError("q must be between 0 and 1, got $q"))
    return fill(Float64(q), n)
end
function _per_group_q(q::AbstractVector{<:Real}, n::Int)
    length(q) == n || throw(ArgumentError("q has $(length(q)) entries for $n groups"))
    all(x -> 0 <= x <= 1, q) || throw(ArgumentError("q must be between 0 and 1"))
    return Float64.(q)
end

# The sizes of the groups, for a mixing value that needs them.
_group_sizes(p::GroupedPopulation, n::Int, what) = p.sizes
_group_sizes(p::SinglePopulation, n::Int, what) = n == 1 ? [p.size] : _no_group_sizes(what)
_group_sizes(::NoPopulation, n::Int, what) = _no_group_sizes(what)
function _no_group_sizes(what)
    throw(
        ArgumentError(
            "$what needs the size of each group: give `population = Population(sizes)`"
        )
    )
end

"""
    _next_generation(mixing, q, D, population) -> Matrix

The next-generation matrix of a branching process: entry `[i, j]` is the mean
number of type-`i` cases one type-`j` case infects. Frequency dependence needs
no sizes; density dependence scales row `i` by the size of group `i`.
"""
function _next_generation(m::FrequencyDependent, q, D::Real, population)
    n = size(m.matrix, 1)
    return permutedims(m.matrix) .* _per_group_q(q, n) .* D
end
function _next_generation(m::DensityDependent, q, D::Real, population)
    n = size(m.matrix, 1)
    sizes = _group_sizes(
        population, n, "a density-dependent BranchingProcess, whose offspring mean follows the group sizes,"
    )
    return permutedims(m.matrix) .* sizes .* _per_group_q(q, n) .* D
end

# The rate at which one infective of group `j` contacts one given member of
# group `i`, with the probability of transmission folded in: entry `[j, i]`.
function _pair_rates(m::FrequencyDependent, q, population)
    n = size(m.matrix, 1)
    sizes = _group_sizes(
        population, n, "a frequency-dependent MixingProcess, whose rate per pair falls with the group size,"
    )
    all(>(0), sizes) || throw(ArgumentError("group sizes must be positive for a frequency-dependent rate"))
    return m.matrix .* permutedims(_per_group_q(q, n) ./ sizes)
end
function _pair_rates(m::DensityDependent, q, population)
    n = size(m.matrix, 1)
    return m.matrix .* permutedims(_per_group_q(q, n))
end
