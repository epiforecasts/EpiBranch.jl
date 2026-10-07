# ── Multi-type offspring from an offspring matrix ────────────────────

"""
    MultiTypeOffspring(offspring_matrix, dist_fn)

The offspring distribution of a multi-type branching process (for example
children and adults, or health-care workers and the community), built from a
next-generation matrix `M`: `M[i, j]` is the mean number of type-`i` cases
infected by one type-`j` case.

A type-`j` case draws how many people it infects in total from
`dist_fn(R_j)`, where `R_j` is the sum of column `j`, then splits them at
random between types in proportion to that column. The mean number infected
is the mean of `dist_fn(R_j)`, which equals `R_j` only if the distribution
has mean `R_j`. A function such as `R -> Poisson(θ * R)` scales the whole
matrix by `θ`.

!!! warning
    `dist_fn` must take a mean if the column sums are meant to be the
    reproduction numbers. Use `R -> NegBin(R, k)` (mean `R`, dispersion `k`).
    `R -> NegativeBinomial(R, 0.3)` from Distributions.jl reads `R` as a
    number of failures, not a mean: with it, a matrix whose dominant
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

    R_by_type = vec(sum(offspring_matrix, dims = 1))
    alloc_probs = similar(offspring_matrix, float(eltype(offspring_matrix)))
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end
    return MultiTypeOffspring(offspring_matrix, dist_fn, R_by_type, alloc_probs)
end

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
type-`j` case. `dist_fn` turns the total for each type of infector (the column
sum, its reproduction number) into an offspring distribution, and
`generation_time` is the generation time in days. See
[`MultiTypeOffspring`](@ref EpiBranch.MultiTypeOffspring) for how secondary
cases are split between types.

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

"""
    draw_offspring(rng, offspring::MultiTypeOffspring, individual, state)

Draw the number of secondary cases of each type for a case of type `j`: a
total from `dist_fn(R_j)`, split at random between types in proportion to
column `j` of the next-generation matrix.
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
    total = rand(rng, offspring.dist_fn(R))
    total == 0 && return zeros(Int, n)
    return rand(rng, Multinomial(total, offspring.alloc_probs[:, pt]))
end
