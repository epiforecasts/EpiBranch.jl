# ── Multi-type offspring from an offspring matrix ────────────────────

"""
    MultiTypeOffspring(offspring_matrix, dist_fn)

Offspring specification of a multi-type branching process built from an
offspring matrix. `offspring_matrix[i, j]` is the expected number of type-`i`
offspring from a type-`j` parent. A type-`j` parent draws its total number of
offspring from `dist_fn(R_j)`, where `R_j` is the sum of column `j`, and
allocates them across types multinomially in proportion to that column.

A [`BranchingProcess`](@ref) built with
`BranchingProcess(offspring_matrix, dist_fn, generation_time)` stores one, so
the offspring draw and the multi-type analytics
([`reproduction_number`](@ref), [`extinction_probability`](@ref)) use the same
matrix and distribution family.
"""
struct MultiTypeOffspring{M <: AbstractMatrix{<:Real}, F, R <: AbstractVector{<:Real},
    A <: AbstractMatrix{<:Real}}
    offspring_matrix::M
    dist_fn::F
    R_by_type::R
    alloc_probs::A
end

function MultiTypeOffspring(offspring_matrix::AbstractMatrix{<:Real}, dist_fn)
    n = size(offspring_matrix, 1)
    size(offspring_matrix, 2) == n || throw(ArgumentError(
        "offspring_matrix must be square, got $(size(offspring_matrix))"))

    R_by_type = vec(sum(offspring_matrix, dims = 1))
    alloc_probs = similar(offspring_matrix, float(eltype(offspring_matrix)))
    for j in 1:n
        s = R_by_type[j]
        alloc_probs[:, j] = s > 0 ? offspring_matrix[:, j] ./ s : fill(1.0 / n, n)
    end
    _check_offspring_means(R_by_type, dist_fn)
    return MultiTypeOffspring(offspring_matrix, dist_fn, R_by_type, alloc_probs)
end

# Only the column proportions of the matrix reach the draw; the size of each
# type's offspring comes from the mean of `dist_fn(R_j)`. The two usually agree,
# and when they do not it is the distribution that decides, so a matrix written
# down as the reproduction numbers between types then means something else. The
# common way to get there is `Distributions.NegativeBinomial(R, k)`, whose first
# argument is a number of failures rather than a mean (use `NegBin(R, k)`), so
# say so at construction instead of leaving a silently rescaled R.
function _check_offspring_means(R_by_type, dist_fn)
    for (j, R) in enumerate(R_by_type)
        R > 0 || continue
        m = mean(dist_fn(R))
        isapprox(m, R; rtol = 1e-6) && continue
        @warn "The offspring matrix gives type $j a column sum of $R, but " *
              "`dist_fn($R)` has mean $m, which is what both the simulation and " *
              "the analytical helpers use. If you meant a negative binomial with " *
              "mean R and dispersion k, use `NegBin(R, k)`." maxlog=1
        return nothing
    end
    return nothing
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
    throw(ArgumentError(
        "This function only works with single-type models; for a multi-type model " *
        "use reproduction_number or extinction_probability"))
end

"""
    BranchingProcess(offspring_matrix, dist_fn, generation_time; kwargs...)

Construct a multi-type branching process from an offspring matrix.
`offspring_matrix[i, j]` is the expected number of type-`i` offspring from a
type-`j` parent. `dist_fn` maps each type's R to an offspring distribution.
"""
function BranchingProcess(offspring_matrix::Matrix{Float64},
        dist_fn::Function,
        gt::Union{Distribution, Function};
        population_size::Union{Int, NoPopulation} = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels())
    offspring = MultiTypeOffspring(offspring_matrix, dist_fn)
    BranchingProcess((Infectiousness(offspring; kernel = gt),), population_size,
        _n_types(offspring), type_labels)
end

"""
    draw_offspring(rng, offspring::MultiTypeOffspring, individual, state)

Draw offspring counts per type for a parent of type `j` under an offspring
matrix: a total from `dist_fn(R_j)`, split multinomially across types.
"""
function draw_offspring(rng::AbstractRNG, offspring::MultiTypeOffspring,
        individual, state::SimulationState)
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
