# ── Multi-type offspring from an offspring matrix ────────────────────

"""
    MultiTypeOffspring(offspring_matrix, dist_fn)

The offspring distribution of a multi-type branching process (for example
children and adults, or health-care workers and the community), built from a
next-generation matrix `M`: `M[i, j]` is the mean number of type-`i` cases
infected by one type-`j` case: column `j` holds a type-`j` case's offspring,
and its sum is that type's reproduction number.

!!! warning "Contact matrices are often the other way round"
    A contact matrix from a survey frequently has the respondent as the row
    and the contact as the column (as with socialmixr's `contact_matrix`),
    so a person's number of contacts is a row sum. Transpose such a matrix
    (`permutedims`) before passing it as `M`; the multi-type tutorial works
    through an example. Passing it as given runs without error but gives the
    wrong type-specific reproduction numbers.

A type-`j` case draws how many people it infects in total, then splits them
at random between types in proportion to column `j`. `dist_fn` gives that
total count law in one of two ways:

- A function of the column sum `R_j`: a type-`j` case draws from
  `dist_fn(R_j)`. The mean number infected is the mean of `dist_fn(R_j)`,
  which equals `R_j` only if the distribution has mean `R_j`. A function such
  as `R -> Poisson(θ * R)` scales the whole matrix by `θ`.
- A vector of one distribution per type: a type-`j` case draws from
  `dist_fn[j]` directly, independent of `R_j`. Use this when the offspring
  law needs to differ between types in more than its mean, for example one
  heavy-tailed and one not, or two laws with the same mean but different
  dispersion.

!!! warning
    A function `dist_fn` must take a mean if the column sums are meant to be
    the reproduction numbers. Use `R -> NegBin(R, k)` (mean `R`, dispersion
    `k`). `R -> NegativeBinomial(R, 0.3)` from Distributions.jl reads `R` as a
    number of successes, not a mean: with it, a matrix whose dominant
    eigenvalue is 1.045 gives a process with reproduction number 2.44.
    [`reproduction_number`](@ref) reports the reproduction number the model
    actually has.

`BranchingProcess(M, dist_fn, generation_time)` builds one of these;
[`reproduction_number`](@ref) and [`extinction_probability`](@ref) use the
same matrix and distribution.
"""
struct MultiTypeOffspring{
        M <: AbstractMatrix{<:Real}, F, R <: AbstractVector{<:Real},
        A <: AbstractMatrix{<:Real},
    }
    offspring_matrix::M
    dist_fn::F
    R_by_type::R
    alloc_probs::A
end

function MultiTypeOffspring(offspring_matrix::AbstractMatrix{<:Real}, dist_fn)
    n = size(offspring_matrix, 1)
    size(offspring_matrix, 2) == n || throw(
        ArgumentError(
            "offspring_matrix must be square, got $(size(offspring_matrix))"
        )
    )
    dist_fn isa AbstractVector && length(dist_fn) != n && throw(
        ArgumentError(
            "dist_fn has $(length(dist_fn)) laws for $n types"
        )
    )

    R_by_type = vec(sum(offspring_matrix, dims = 1))
    alloc_probs = similar(offspring_matrix, float(eltype(offspring_matrix)))
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end
    return MultiTypeOffspring(offspring_matrix, dist_fn, R_by_type, alloc_probs)
end

# The total-count law for a type-`j` case with column sum `R`: `dist_fn(R)`
# for a function, or `dist_fn[j]` for one law per type. Dispatch on the type
# of `dist_fn` rather than branch inside one method: a third way of giving
# the law then needs only a new method here, not an edit to this one.
_type_law(dist_fn, R, ::Integer) = dist_fn(R)
_type_law(dist_fn::AbstractVector, ::Real, j::Integer) = dist_fn[j]

_n_types(o::MultiTypeOffspring) = size(o.offspring_matrix, 1)

_offspring_label(o::MultiTypeOffspring) = "MultiTypeOffspring($(_n_types(o)) types)"

# Multi-type offspring in a single-window process reaches the multi-type
# analytics directly. Everything else goes through the single-type accessor,
# which also raises the error for a model with several windows.
function _analytic_offspring(m::BranchingProcess)
    length(m.infectiousness) == 1 || return single_type_offspring(m)
    return _analytic_offspring(m, m.infectiousness[1].offspring)
end
_analytic_offspring(m::BranchingProcess, ::Any) = single_type_offspring(m)
_analytic_offspring(::BranchingProcess, o::MultiTypeOffspring) = o

function _single_type(::MultiTypeOffspring)
    throw(
        ArgumentError(
            "This function only works with single-type models; for a multi-type model " *
                "use reproduction_number or extinction_probability"
        )
    )
end

"""
    BranchingProcess(offspring_matrix, dist_fn, generation_time; kwargs...)

A multi-type branching process from a next-generation matrix:
`offspring_matrix[i, j]` is the mean number of type-`i` cases infected by one
type-`j` case. `dist_fn` gives each type of infector its offspring
distribution, either as a function of the column sum (its reproduction
number) or as one law per type, and `generation_time` is the generation time
in days. See [`MultiTypeOffspring`](@ref EpiBranch.MultiTypeOffspring) for how
secondary cases are split between types.

# Examples
```julia
M = [1.2 0.4; 0.3 0.9]
BranchingProcess(M, R -> NegBin(R, 0.5), Gamma(2.0, 3.0))
```
"""
function BranchingProcess(
        offspring_matrix::Matrix{Float64},
        dist_fn,
        gt;
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels()
    )
    offspring = MultiTypeOffspring(offspring_matrix, dist_fn)
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),), population_size,
        _n_types(offspring), type_labels
    )
end

# The next-generation matrix for `BranchingProcess(contacts, q, D, ...)`
# below: `contacts[j, i]`, the daily contacts a type-`j` respondent makes
# with type `i` (row = respondent, as a survey matrix is laid out), is
# transposed so that columns are infectors, then scaled by the transmission
# probability `q` (per infectee type when given as a vector, because
# susceptibility belongs to the row) and the infectious period `D`.
function _contact_offspring_matrix(contacts::AbstractMatrix{<:Real}, q, D)
    n = size(contacts, 1)
    size(contacts, 2) == n || throw(
        ArgumentError("contacts must be square, got $(size(contacts))")
    )
    q isa AbstractVector && length(q) != n && throw(
        ArgumentError("q has $(length(q)) entries for $n types, expected $n")
    )
    return Matrix{Float64}(permutedims(contacts) .* q .* D)
end

"""
    BranchingProcess(contacts, q, D, dist_fn, gt; kwargs...)

A multi-type branching process built directly from a contact matrix, rather
than a next-generation matrix: the common case for a contact matrix from a
social contact survey, laid out respondent by contact (the `contact_matrix()`
function from socialmixr in R, a summary from cntmosaic in Python, or a
published matrix), together with a transmission probability and an
infectious period.

`contacts[j, i]` is the mean number of type-`i` contacts a type-`j`
respondent makes per day, the layout such a survey matrix comes in. This
builds the next-generation matrix `permutedims(contacts) .* q .* D`,
transposing for the caller: a next-generation matrix has columns for
infectors and rows for infectees, the other way round from a contact
matrix, whose rows are the respondent.

`q` is the probability that a contact between an infectious and a
susceptible person results in transmission: a single number applied to
every contact, or a vector of one probability per infectee type, for a
per-type susceptibility. `D` is the mean infectious period in days. `dist_fn`
and `gt` are as in the next-generation-matrix form of [`BranchingProcess`](@ref):
the offspring law for a type's column sum, and the generation time.

# Examples
```julia
using EpiBranch, Distributions

contacts = [10.0 1.0; 4.0 2.0]  # row = respondent, column = contact
q, D = 0.05, 5.0
model = BranchingProcess(
    contacts, q, D, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5);
    type_labels = ["child", "adult"],
)
```
"""
function BranchingProcess(
        contacts::AbstractMatrix{<:Real}, q::Union{Real, AbstractVector{<:Real}},
        D::Real, dist_fn, gt;
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels()
    )
    M = _contact_offspring_matrix(contacts, q, D)
    return BranchingProcess(M, dist_fn, gt; population_size, type_labels)
end

"""
    draw_offspring(rng, offspring::MultiTypeOffspring, individual, state)

Draw the number of secondary cases of each type for a case of type `j`: a
total from its offspring law (see [`MultiTypeOffspring`](@ref)), split at
random between types in proportion to column `j` of the next-generation
matrix.
"""
function draw_offspring(
        rng::AbstractRNG, offspring::MultiTypeOffspring,
        individual, state::SimulationState
    )
    n = _n_types(offspring)
    pt = individual_type(individual)
    R = offspring.R_by_type[pt]
    # A sink type (all-zero offspring column) produces no offspring. Return
    # before calling `dist_fn`: the documented `R -> NegBin(R, k)` form throws at
    # R = 0 because `NegBin` requires R > 0.
    R <= 0 && return zeros(Int, n)
    total = rand(rng, _type_law(offspring.dist_fn, R, pt))
    total == 0 && return zeros(Int, n)
    return rand(rng, Multinomial(total, offspring.alloc_probs[:, pt]))
end
