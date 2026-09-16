# ── Multi-type analytics ─────────────────────────────────────────────
#
# Threshold and extinction results for a process built from an offspring
# matrix. A type-j parent draws a total count from G_j = dist_fn(R_j) and splits it
# multinomially with proportions a_ij = M[i, j] / R_j. Its vector PGF is
# therefore f_j(s) = G_j(Σ_i a_ij s_i). This agrees with a product of
# independent per-type PGFs only when G_j is Poisson.

# Total-count law of a type-`j` parent. A sink type (zero column) has no
# offspring, so its law is `Dirac(0)` and `dist_fn`, which may reject R = 0, is
# never called for it.
function _total_count_law(o::MultiTypeOffspring, j::Integer)
    R = o.R_by_type[j]
    return R > 0 ? o.dist_fn(R) : Dirac(0)
end

_total_count_laws(o::MultiTypeOffspring) = [_total_count_law(o, j) for j in 1:_n_types(o)]

# Mean of a total-count law. `mean` covers the standard families, but a law built
# on another, such as a truncated Poisson, can have no method for it, and the
# generic fallback iterates the distribution and fails. Sum the series there, as
# `_pgf` does for the same laws.
function _law_mean(d::DiscreteUnivariateDistribution)
    # `Statistics.mean` accepts any iterable, so asking whether a method applies
    # says nothing; the distribution-specific method either exists or the
    # iterable one is reached and fails.
    try
        return mean(d)
    catch err
        err isa MethodError || rethrow()
    end
    lo, hi = _series_range(d)
    return sum(x * pdf(d, x) for x in lo:hi)
end

# The range of counts a truncated series has to cover. `maximum` gives it for a
# bounded law. For an unbounded one, ask the law how much mass sits above each
# count until what is left is negligible. Subtracting the masses one at a time
# instead would leave a residual of the order of the number of terms times
# `eps`, which for a few dozen terms sits above any useful tolerance, so the
# walk would run to its cap; and `quantile` is no use either, because on a
# truncated law it clamps to the bounds and an integer law clamped to an
# infinite bound throws.
function _series_range(d::DiscreteUnivariateDistribution, tail::Real = 1e-14,
        cap::Int = 1_000_000)
    lo = round(Int, minimum(d))
    hi = maximum(d)
    isfinite(hi) && return lo, round(Int, hi)
    x = lo
    while ccdf(d, x) > tail
        x += 1
        if x - lo >= cap
            @warn "The offspring law has more than $tail of its mass above " *
                  "$cap counts, so the series stops there and the reproduction " *
                  "number and extinction probability are understated." maxlog=1
            break
        end
    end
    return lo, x
end

# Mean matrix of the process: entry `[i, j]` is the expected number of type-`i`
# offspring of a type-`j` parent. The entries use the means of the total-count
# laws, so the matrix matches what the simulator draws even if the mean of
# `dist_fn(R_j)` differs from the column sum `R_j`.
function _mean_matrix(o::MultiTypeOffspring, laws = _total_count_laws(o))
    means = [_law_mean(law) for law in laws]
    T = float(promote_type(eltype(o.alloc_probs), map(typeof, means)...))
    return T[o.alloc_probs[i, j] * means[j] for i in 1:_n_types(o), j in 1:_n_types(o)]
end

# Spectral radius of a non-negative matrix.
_spectral_radius(A::AbstractMatrix{<:LinearAlgebra.BlasReal}) = maximum(abs, eigvals(A))

# Generic element types (e.g. dual numbers under ForwardDiff) have no `eigvals`,
# so this method uses power iteration on `A + I`. The shift makes the dominant
# eigenvalue of a non-negative matrix the unique one of largest modulus, and the
# unit diagonal keeps every iterate strictly positive, so the iteration converges
# even when `A` is reducible.
#
# The stopping test compares values only, so for dual numbers the derivative
# parts of the iterates may still be far from their limits when it passes. The
# method therefore iterates on `A + I` and its transpose together and returns
# the quotient uᵀAv / uᵀv of the right (`v`) and left (`u`) eigenvectors. This
# quotient is stationary in `u` and `v` at the eigenvectors, so once `u` and `v`
# have converged in value its first derivative is uᵀ(dA)v / uᵀv, whatever their
# derivative parts are. Convergence slows as the top two eigenvalues approach
# each other, and the method warns if it stops at `max_iter` before converging,
# since the value and derivative are then unreliable.
# When `u` and `v` are nearly orthogonal, as for a defective dominant
# eigenvalue, the quotient is unstable and the method returns the growth of the
# largest entry of `v` instead.
function _spectral_radius(A::AbstractMatrix{<:Real}; tol::Real = 1e-12,
        max_iter::Int = 100_000)
    n = size(A, 1)
    B = [A[i, j] + (i == j) for i in 1:n, j in 1:n]
    Bt = permutedims(B)
    v = ones(eltype(B), n)
    u = ones(eltype(B), n)
    λ = one(eltype(B))
    converged = false
    for _ in 1:max_iter
        w = B * v
        λ = maximum(w)
        v_new = w ./ λ
        z = Bt * u
        u_new = z ./ maximum(z)
        converged = maximum(abs, v_new .- v) <= tol &&
                    maximum(abs, u_new .- u) <= tol
        v, u = v_new, u_new
        converged && break
    end
    converged ||
        @warn "Power iteration for the spectral radius stopped after $max_iter " *
              "iterations without converging, which happens when the two largest " *
              "eigenvalues are nearly equal. Its value and derivative may be " *
              "inaccurate." maxlog=1
    uv = sum(u .* v)
    uv > sqrt(tol * sum(abs2, u) * sum(abs2, v)) || return λ - 1
    return sum(u .* (A * v)) / uv
end

# Probability generating functions of total-count laws: closed forms for
# Poisson, negative binomial and Dirac, and a truncated series for any other
# discrete distribution.
_pgf(d::Poisson, s) = exp(mean(d) * (s - 1))
function _pgf(d::NegativeBinomial, s)
    r, p = params(d)
    return (p / (1 - (1 - p) * s))^r
end
_pgf(d::Dirac, s) = s^d.value
function _pgf(d::DiscreteUnivariateDistribution, s)
    lo, hi = _series_range(d)
    return sum(pdf(d, x) * s^x for x in lo:hi)
end

"""
    reproduction_number(model)
    reproduction_number(offspring)

Reproduction number of a branching process, computed from its offspring
specification.

For a single-type model this is the mean of the offspring distribution. For a
multi-type model built from an offspring matrix it is R*, the dominant
eigenvalue (spectral radius) of the mean next-generation matrix, whose
`[i, j]` entry is the expected number of type-`i` offspring from a type-`j`
parent. An outbreak can grow with positive probability only if the
reproduction number exceeds 1.

For a matrix whose types cannot all infect one another, R* above 1 says that
some group of types can grow, and not that a case of any given type can: an
index case of a type that never reaches such a group still dies out for
certain. [`extinction_probability`](@ref) answers that per type.

# Examples

```julia
M = [1.5 0.3;
     0.3 1.0]
reproduction_number(BranchingProcess(M, R -> NegBin(R, 0.5), Exponential(5.0)))
reproduction_number(BranchingProcess(NegBin(2.5, 0.16)))  # 2.5
```
"""
reproduction_number(d::DiscreteUnivariateDistribution) = mean(d)
reproduction_number(o::MultiTypeOffspring) = _spectral_radius(_mean_matrix(o))
function reproduction_number(model::Union{TransmissionModel, ModelSpec})
    return reproduction_number(_analytic_offspring(model))
end

"""
    extinction_probability(o::MultiTypeOffspring; tol=1e-10, max_iter=1000)

Extinction probability of a multi-type branching process, one entry per type:
element `j` is the probability that an outbreak started by a single type-`j`
case dies out.

It is the smallest fixed point in `[0, 1]` of the vector PGF,
`q_j = G_j(Σ_i a_ij q_i)`, where `G_j` is the PGF of `dist_fn(R_j)` and
`a_ij = M[i, j] / R_j` the proportions in which a type-`j` parent's offspring
are split across types. Fixed-point iteration from zero converges to it.

A type-`j` outbreak can grow only if type-`j` cases lead, through some chain
of transmission, to a group of types that infect each other with a
reproduction number above 1. Types without such a chain, and every type when
[`reproduction_number`](@ref) is at most 1, get exactly 1. Both of those take
the offspring count to vary: a deterministic law, such as `Dirac(1)` at R = 1,
gives every case exactly one offspring and so never dies out, whether it is a
class of its own or one the other types feed, and 1 is the wrong answer for it.

Iteration that has not converged by `max_iter` warns, which happens when the
reproduction number is close to 1.

To use it on a model built with
`BranchingProcess(offspring_matrix, dist_fn, generation_time)`, call
`extinction_probability(model)`.
"""
function extinction_probability(o::MultiTypeOffspring; tol::Real = 1e-10,
        max_iter::Int = 1000)
    n = _n_types(o)
    laws = _total_count_laws(o)
    M = _mean_matrix(o, laws)
    T = eltype(M)
    can_grow = _reaches_supercritical_class(M)
    any(can_grow) || return ones(T, n)

    q = T[can_grow[j] ? 0 : 1 for j in 1:n]
    for _ in 1:max_iter
        q_new = T[can_grow[j] ?
                  _pgf(laws[j], sum(o.alloc_probs[i, j] * q[i] for i in 1:n)) : 1
                  for j in 1:n]
        maximum(abs.(q_new .- q)) < tol && return q_new
        q = q_new
    end
    @warn_unconverged_extinction(max_iter, "the reproduction number")
    return q
end

# Whether each type can lead to a communicating class of types whose mean
# matrix restricted to the class has spectral radius above 1. A type-`j` parent
# has type-`i` offspring when `M[i, j] > 0`. Only such types have extinction
# probability below 1.
function _reaches_supercritical_class(M::AbstractMatrix)
    n = size(M, 1)
    # reach[i, j]: a type-`j` case has type-`i` descendants (or i == j).
    reach = [i == j || M[i, j] > 0 for i in 1:n, j in 1:n]
    for k in 1:n, j in 1:n, i in 1:n
        reach[i, j] = reach[i, j] || (reach[i, k] && reach[k, j])
    end
    supercritical = falses(n)
    for i in 1:n
        class = [j for j in 1:n if reach[i, j] && reach[j, i]]
        supercritical[i] = i == first(class) ?
                           _spectral_radius(M[class, class]) > 1 :
                           supercritical[first(class)]
    end
    return [any(supercritical[i] && reach[i, j] for i in 1:n) for j in 1:n]
end
