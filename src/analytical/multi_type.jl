# ── Multi-type analytics ─────────────────────────────────────────────
#
# Threshold and extinction results for a process built from an offspring
# matrix. A type-j parent draws a total count from G_j = dist_fn(R_j) and splits it
# multinomially with proportions a_ij = M[i, j] / R_j. Its vector PGF is
# therefore f_j(s) = G_j(Σ_i a_ij s_i). This agrees with a product of
# independent per-type PGFs only when G_j is Poisson.

# Total-count law of a type-`j` parent. A sink type (zero column) has no
# offspring. Its law is `Dirac(0)`, so `dist_fn`, which may reject R = 0, is
# never called with R = 0.
function _total_count_law(o::MultiTypeOffspring, j::Integer)
    R = o.R_by_type[j]
    return R > 0 ? o.dist_fn(R) : Dirac(0)
end

_total_count_laws(o::MultiTypeOffspring) = [_total_count_law(o, j) for j in 1:_n_types(o)]

# Mean matrix of the process: entry `[i, j]` is the expected number of type-`i`
# offspring of a type-`j` parent. The entries use the means of the total-count
# laws, so the matrix matches what the simulator draws even if the mean of
# `dist_fn(R_j)` differs from the column sum `R_j`.
function _mean_matrix(o::MultiTypeOffspring, laws = _total_count_laws(o))
    means = [mean(law) for law in laws]
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
# quotient is stationary in `u` and `v` at the eigenvectors, so its first
# derivative is uᵀ(dA)v / uᵀv whatever the derivative parts of `u` and `v` are.
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
    hi = isfinite(maximum(d)) ? maximum(d) : quantile(d, 1 - 1e-14)
    return sum(pdf(d, x) * s^x for x in minimum(d):round(Int, hi))
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
are split across types. Fixed-point iteration from zero converges to it. When
[`reproduction_number`](@ref) is at most 1, every entry is 1.

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
    _spectral_radius(M) <= 1 && return ones(T, n)

    q = zeros(T, n)
    for _ in 1:max_iter
        q_new = T[_pgf(laws[j], sum(o.alloc_probs[i, j] * q[i] for i in 1:n))
                  for j in 1:n]
        maximum(abs.(q_new .- q)) < tol && return q_new
        q = q_new
    end
    return q
end
