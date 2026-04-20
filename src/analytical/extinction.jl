"""
    extinction_probability(R::Real, k::Real; tol=1e-10, max_iter=1000)

Compute the extinction probability of a branching process with
Negative Binomial offspring distribution parameterised by mean `R`
and dispersion `k`.

Fixed-point iteration on the probability generating function is used.
For R ≤ 1, returns 1.0 (certain extinction).
"""
function extinction_probability(R::Real, k::Real; tol::Real = 1e-10, max_iter::Int = 1000)
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

For Poisson(λ): the PGF exp(λ(s-1)) is used.
For NegativeBinomial: R and k are extracted and the closed-form PGF is applied.
"""
function extinction_probability(d::Poisson; tol::Real = 1e-10, max_iter::Int = 1000)
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

function extinction_probability(d::NegativeBinomial; tol::Real = 1e-10, max_iter::Int = 1000)
    k = d.r
    R = mean(d)
    return extinction_probability(R, k; tol, max_iter)
end

"""
    epidemic_probability(R::Real, k::Real; kwargs...)

Probability that a single introduction leads to a major epidemic.
Complement of extinction probability.
"""
function epidemic_probability(R::Real, k::Real; kwargs...)
    1.0 - extinction_probability(R, k; kwargs...)
end

"""
    epidemic_probability(d::Distribution; kwargs...)

Probability of a major epidemic for a given offspring distribution.
"""
function epidemic_probability(d::Distribution; kwargs...)
    1.0 - extinction_probability(d; kwargs...)
end

# ── BranchingProcess dispatch ────────────────────────────────────────

"""
    extinction_probability(model::TransmissionModel; kwargs...)

Extinction probability for a single-type transmission model, extracted
from the model's offspring specification via `_single_type_offspring`.
Works for `BranchingProcess` and wrappers that delegate that accessor
(e.g. `PartiallyObserved`).
"""
function extinction_probability(model::TransmissionModel; kwargs...)
    return extinction_probability(_single_type_offspring(model); kwargs...)
end

"""
    epidemic_probability(model::TransmissionModel; kwargs...)

Epidemic probability for a single-type transmission model.
"""
function epidemic_probability(model::TransmissionModel; kwargs...)
    1.0 - extinction_probability(model; kwargs...)
end

# ── Containment probability (analytical) ─────────────────────────────

"""
    probability_contain(R, k; n_initial=1, ind_control=0.0, pop_control=0.0)

Probability that an outbreak is contained (goes extinct), accounting for
individual-level and population-level control measures and multiple initial
infections.

- `ind_control`: probability each case is individually controlled (removed
  before transmitting), e.g. through case isolation
- `pop_control`: population-level reduction in R, e.g. through social
  distancing. Effective R becomes `(1 - pop_control) * R`
- `n_initial`: number of initial independent introductions

The containment probability for a single introduction is:

    q = ind_control + (1 - ind_control) * pgf(q)

where `pgf` is the PGF of the offspring distribution with effective R.
For `n_initial` independent introductions, the probability is `q^n_initial`.
"""
function probability_contain(R::Real, k::Real;
        n_initial::Int = 1,
        ind_control::Real = 0.0,
        pop_control::Real = 0.0,
        tol::Real = 1e-10, max_iter::Int = 1000)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    0.0 <= ind_control <= 1.0 || throw(ArgumentError("ind_control must be in [0, 1]"))
    0.0 <= pop_control <= 1.0 || throw(ArgumentError("pop_control must be in [0, 1]"))
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))

    R_eff = (1.0 - pop_control) * R
    R_eff <= 1.0 && return 1.0

    p = k / (k + R_eff)

    # Fixed-point iteration: q = ind_control + (1-ind_control) * pgf(q)
    q = 0.5
    for _ in 1:max_iter
        pgf_q = (p / (1.0 - (1.0 - p) * q))^k
        q_new = ind_control + (1.0 - ind_control) * pgf_q
        abs(q_new - q) < tol && return q_new^n_initial
        q = q_new
    end

    return q^n_initial
end

"""
    probability_contain(d::Distribution; n_initial=1, ind_control=0.0, pop_control=0.0)

Containment probability for a given offspring distribution.
"""
function probability_contain(d::NegativeBinomial; kwargs...)
    return probability_contain(mean(d), d.r; kwargs...)
end

function probability_contain(d::Poisson; n_initial::Int = 1,
        ind_control::Real = 0.0, pop_control::Real = 0.0, kwargs...)
    # Poisson is NegBin with k→∞; use large k
    return probability_contain(mean(d), 1e6; n_initial, ind_control, pop_control, kwargs...)
end

"""
    probability_contain(model::TransmissionModel; kwargs...)

Containment probability for a single-type transmission model. Delegates
through `_single_type_offspring`, so wrappers such as `PartiallyObserved`
work too.
"""
function probability_contain(model::TransmissionModel; kwargs...)
    return probability_contain(_single_type_offspring(model); kwargs...)
end
