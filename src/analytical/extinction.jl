"""
    extinction_probability(R::Real, k::Real; tol=1e-10, max_iter=1000)

Compute the extinction probability of a branching process with
Negative Binomial offspring distribution parameterised by mean `R`
and dispersion `k`.

Uses fixed-point iteration on the probability generating function.
For R ≤ 1, returns 1.0 (certain extinction).
"""
function extinction_probability(R::Real, k::Real; tol::Real=1e-10, max_iter::Int=1000)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))

    R <= 1.0 && return 1.0

    # PGF of NegBin(k, p) is (p / (1 - (1-p)*s))^k
    # where p = k/(k+R). Fixed point: q = pgf(q).
    p = k / (k + R)

    # Start from a value close to 0
    q = 0.5
    for _ in 1:max_iter
        q_new = (p / (1.0 - (1.0 - p) * q))^k
        abs(q_new - q) < tol && return q_new
        q = q_new
    end

    return q
end

"""
    extinction_probability(d::Distribution; tol=1e-10, max_iter=1000)

Compute extinction probability for any discrete offspring distribution
via fixed-point iteration on the PGF.

For Poisson(λ): uses the PGF exp(λ(s-1)).
For NegativeBinomial: extracts R and k and uses the closed-form PGF.
"""
function extinction_probability(d::Poisson; tol::Real=1e-10, max_iter::Int=1000)
    λ = mean(d)
    λ <= 1.0 && return 1.0

    q = 0.5
    for _ in 1:max_iter
        q_new = exp(λ * (q - 1.0))
        abs(q_new - q) < tol && return q_new
        q = q_new
    end
    return q
end

function extinction_probability(d::NegativeBinomial; tol::Real=1e-10, max_iter::Int=1000)
    k = d.r
    R = mean(d)
    return extinction_probability(R, k; tol, max_iter)
end

"""
    epidemic_probability(R::Real, k::Real; kwargs...)

Probability that a single introduction leads to a major epidemic.
Complement of extinction probability.
"""
epidemic_probability(R::Real, k::Real; kwargs...) =
    1.0 - extinction_probability(R, k; kwargs...)

"""
    epidemic_probability(d::Distribution; kwargs...)

Probability of a major epidemic for a given offspring distribution.
"""
epidemic_probability(d::Distribution; kwargs...) =
    1.0 - extinction_probability(d; kwargs...)

# ── BranchingProcess dispatch ────────────────────────────────────────

"""
    extinction_probability(model::BranchingProcess; kwargs...)

Extinction probability for a single-type branching process, extracted
from the model's offspring distribution.
"""
function extinction_probability(model::BranchingProcess; kwargs...)
    model.offspring isa Distribution || throw(ArgumentError(
        "Analytical extinction probability only available for single-type models"))
    return extinction_probability(model.offspring; kwargs...)
end

"""
    epidemic_probability(model::BranchingProcess; kwargs...)

Epidemic probability for a single-type branching process.
"""
epidemic_probability(model::BranchingProcess; kwargs...) =
    1.0 - extinction_probability(model; kwargs...)
