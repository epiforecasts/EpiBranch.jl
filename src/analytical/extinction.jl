# Fixed-point iteration converges slowly near a reproduction number of 1.
# At R = 1.005, the default 1000 iterations end about 7e-5 from the answer;
# the gap grows as R approaches 1. Warn when `max_iter` is reached to identify
# results that are still unconverged.
#
# A macro gives each iteration its own warning call site and `maxlog` count.
# A shared function would let one
# unconverged multi-type call silence every later single-type one in the session.
# The expansion drops the line numbers of this definition, which attributes its
# code to the call site, where coverage tools look for it.
macro warn_unconverged_extinction(max_iter, cause)
    return esc(
        Base.remove_linenums!(
            quote
                @warn "Fixed-point iteration for the extinction probability stopped " *
                    "after $($max_iter) iterations without converging, which happens " *
                    "when $($cause) is close to 1. The result may be inaccurate; " *
                    "raise `max_iter`." maxlog = 1
            end
        )
    )
end

"""
    extinction_probability(R::Real, k::Real; tol=1e-10, max_iter=1000)

Probability that transmission from a single introduced case dies out without
a major outbreak, when each case infects a negative binomial number of others
with mean `R` and dispersion `k` (`NegBin(R, k)`; smaller `k` means more
superspreading). It is 1 when `R ≤ 1`. Interventions are not included; for
those, see [`probability_contain`](@ref) or simulate and use
[`containment_probability`](@ref).

The result is the smallest solution of `q = G(q)`, where `G` is the
probability generating function of the offspring distribution, found by
iteration to tolerance `tol`. Convergence is slow when `R` is close to 1: a
warning is given if `max_iter` iterations are reached.

# Examples

```julia
extinction_probability(2.5, 0.16)   # strong superspreading: about 0.79
extinction_probability(2.5, 100.0)  # little superspreading: about 0.11
```
"""
function extinction_probability(R::Real, k::Real; tol::Real = 1.0e-10, max_iter::Int = 1000)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))

    R <= 1.0 && return 1.0

    offspring = NegativeBinomial(k, k / (k + R))

    q = 0.5
    for _ in 1:max_iter
        q_new = _pgf(offspring, q)
        abs(q_new - q) < tol && return q_new
        q = q_new
    end

    @warn_unconverged_extinction(max_iter, "the reproduction number")
    return q
end

"""
    extinction_probability(d::Poisson; tol=1e-10, max_iter=1000)
    extinction_probability(d::NegativeBinomial; tol=1e-10, max_iter=1000)

Probability that transmission from a single introduced case dies out without
a major outbreak, for a Poisson or negative binomial offspring distribution
`d`, such as `NegBin(2.5, 0.16)`. Computed as for
`extinction_probability(R, k)`.
"""
function extinction_probability(d::Poisson; tol::Real = 1.0e-10, max_iter::Int = 1000)
    mean(d) <= 1.0 && return 1.0

    q = 0.5
    for _ in 1:max_iter
        q_new = _pgf(d, q)
        abs(q_new - q) < tol && return q_new
        q = q_new
    end
    @warn_unconverged_extinction(max_iter, "the reproduction number")
    return q
end

function extinction_probability(d::NegativeBinomial; tol::Real = 1.0e-10, max_iter::Int = 1000)
    k = d.r
    R = mean(d)
    return extinction_probability(R, k; tol, max_iter)
end

"""
    epidemic_probability(R::Real, k::Real; kwargs...)

Probability that a single introduced case leads to a major epidemic, with
`NegBin(R, k)` offspring: one minus [`extinction_probability`](@ref).
"""
function epidemic_probability(R::Real, k::Real; kwargs...)
    return 1.0 - extinction_probability(R, k; kwargs...)
end

"""
    epidemic_probability(offspring; kwargs...)
    epidemic_probability(model; kwargs...)

Probability that a single introduced case leads to a major epidemic: one
minus [`extinction_probability`](@ref), for an offspring distribution or a
model. For a multi-type model there is one value per type of index case.

Given a model, only its offspring distribution is used: interventions,
population characteristics and the observation model are ignored. Use
[`probability_contain`](@ref) for simple control measures, or simulate and
use [`containment_probability`](@ref).
"""
function epidemic_probability(offspring; kwargs...)
    return 1.0 .- extinction_probability(offspring; kwargs...)
end

# ── BranchingProcess dispatch ────────────────────────────────────────

"""
    extinction_probability(model::TransmissionModel; kwargs...)

Probability that transmission from a single introduced case dies out, from
the model's offspring distribution. Interventions, population characteristics
and the observation model are ignored; use [`probability_contain`](@ref) for
simple control measures, or simulate and use
[`containment_probability`](@ref). For a single-type model the
result is one number; for a multi-type model built from an offspring matrix it
has one value per type of index case.
"""
function extinction_probability(model::Union{TransmissionModel, ModelSpec}; kwargs...)
    return extinction_probability(_analytic_offspring(model); kwargs...)
end

# ── Containment probability (analytical) ─────────────────────────────

"""
    probability_contain(R, k; n_initial=1, ind_control=0.0, pop_control=0.0)

Closed-form probability that an outbreak is contained (dies out), with
`NegBin(R, k)` offspring, under two simple kinds of control and with several
introductions. It is the closed-form counterpart of the simulation estimate
[`containment_probability`](@ref), and reduces to
[`extinction_probability`](@ref) with no control and one introduction.

- `ind_control`: the probability that each case is controlled individually,
  for example isolated, before it infects anyone.
- `pop_control`: the proportional reduction in R from population-wide
  measures such as social distancing; the effective R is
  `(1 - pop_control) * R`.
- `n_initial`: the number of independent introductions.

For one introduction, the containment probability `q` solves

    q = ind_control + (1 - ind_control) * G(q)

where `G` is the probability generating function of the offspring
distribution with the effective R. For `n_initial` introductions it is
`q^n_initial`.

This is a port of `probability_contain` (and the `probability_extinct`
equation it builds on) in the R package superspreading (Lambert et al.,
https://github.com/epiverse-trace/superspreading, MIT).

# Examples

```julia
# isolating half of cases before they transmit, three introductions
probability_contain(2.5, 0.16; ind_control = 0.5, n_initial = 3)
```
"""
function probability_contain(
        R::Real, k::Real;
        n_initial::Int = 1,
        ind_control::Real = 0.0,
        pop_control::Real = 0.0,
        tol::Real = 1.0e-10, max_iter::Int = 1000
    )
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    0.0 <= ind_control <= 1.0 || throw(ArgumentError("ind_control must be in [0, 1]"))
    0.0 <= pop_control <= 1.0 || throw(ArgumentError("pop_control must be in [0, 1]"))
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))

    R_eff = (1.0 - pop_control) * R
    R_eff <= 1.0 && return 1.0

    offspring = NegativeBinomial(k, k / (k + R_eff))

    # Fixed-point iteration: q = ind_control + (1-ind_control) * pgf(q)
    q = 0.5
    for _ in 1:max_iter
        q_new = ind_control + (1.0 - ind_control) * _pgf(offspring, q)
        abs(q_new - q) < tol && return q_new^n_initial
        q = q_new
    end

    # This iteration's rate at the fixed point is `(1 - ind_control)` times the
    # effective reproduction number, which includes `pop_control`. Convergence
    # slows when this product approaches 1.
    @warn_unconverged_extinction(
        max_iter,
        "the effective reproduction number times one minus `ind_control`"
    )
    return q^n_initial
end

"""
    probability_contain(d::Distribution; n_initial=1, ind_control=0.0, pop_control=0.0)

Closed-form containment probability for a Poisson or negative binomial
offspring distribution, with the same keywords as
`probability_contain(R, k)`. Poisson offspring is treated as negative
binomial with very large `k`.
"""
function probability_contain(d::NegativeBinomial; kwargs...)
    return probability_contain(mean(d), d.r; kwargs...)
end

function probability_contain(
        d::Poisson; n_initial::Int = 1,
        ind_control::Real = 0.0, pop_control::Real = 0.0, kwargs...
    )
    # Poisson is NegBin with k→∞; use large k
    return probability_contain(mean(d), 1.0e6; n_initial, ind_control, pop_control, kwargs...)
end

"""
    probability_contain(model::TransmissionModel; kwargs...)

Closed-form containment probability for a single-type model, from its
offspring distribution, with the same keywords as `probability_contain(R, k)`.
The model's own interventions are not included; express control through
`ind_control` and `pop_control`.
"""
function probability_contain(model::Union{TransmissionModel, ModelSpec}; kwargs...)
    return probability_contain(single_type_offspring(model); kwargs...)
end
