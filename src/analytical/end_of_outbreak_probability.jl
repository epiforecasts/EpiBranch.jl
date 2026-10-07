# ── End-of-outbreak probability ─────────────────────────────────────
#
# π(τ; R, k, generation_time) is the probability that no further cases
# occur in a cluster, given `τ` time has elapsed since the most recent
# observed case. Used as a principled per-cluster `is_finished` weight
# in the real-time chain-size mixture likelihood (see
# `loglikelihood(::ChainSizes, ::Distribution)`).
#
# Closed form for full reporting (`ρ = 1`):
#
#   π(τ; R, k, G) = ((k + R · (1 − S(τ))) / (k + R))^k
#
# with `S(τ) = ccdf(generation_time, τ)`. Reduces to the offspring
# zero-probability `G(0) = (k/(k+R))^k` at `τ = 0` and tends to 1 as
# `τ → ∞`. Reference: Thompson, Morgan & Jansen, *Phil Trans B* 2019,
# in the per-case Markov-property reduction at the most recent
# observed case.
#
# Under-reporting (`ρ < 1`) needs the full Volterra recursion for the
# per-case η(τ) and is not implemented here.

"""
    end_of_outbreak_probability(R, k, generation_time::Distribution, τ::Real)

Probability that a cluster is over, so no further cases will occur, given
that its most recent case was `τ` days ago. Assumes `NegBin(R, k)` offspring,
a generation time distribution `generation_time` (days) and that every case
is reported. Following Thompson, Morgan & Jansen (2019, Phil Trans B), it is

    ((k + R · (1 − S(τ))) / (k + R))^k

where `S(τ)` is the probability that a generation time exceeds `τ`. It rises
from the probability of a case infecting nobody at `τ = 0` towards 1 as `τ`
grows.

Use it as `prob_concluded` in `loglikelihood(::ChainSizes, ...)` or
[`chain_size_distribution`](@ref) for real-time cluster data, where some
clusters may still be growing.

# Examples

```julia
# 14 days since the last case, generation time with mean 6 days
end_of_outbreak_probability(0.8, 0.5, Gamma(2.0, 3.0), 14.0)
```
"""
function end_of_outbreak_probability(R::Real, k::Real, generation_time::Distribution, τ::Real)
    isinf(τ) && return one(float(R))
    τ <= zero(τ) && return (k / (k + R))^k
    S = ccdf(generation_time, τ)
    return ((k + R * (one(S) - S)) / (k + R))^k
end

"""
    end_of_outbreak_probability(offspring::NegativeBinomial, generation_time, τ)
    end_of_outbreak_probability(offspring::Poisson, generation_time, τ)

End-of-outbreak probability with the offspring distribution given directly,
such as `NegBin(0.8, 0.5)` or `Poisson(0.8)`. For Poisson offspring (no
superspreading) it is `exp(−R · S(τ))`.
"""
function end_of_outbreak_probability(
        offspring::NegativeBinomial, generation_time::Distribution,
        τ::Real
    )
    return end_of_outbreak_probability(mean(offspring), offspring.r, generation_time, τ)
end

function end_of_outbreak_probability(offspring::Poisson, generation_time::Distribution, τ::Real)
    R = mean(offspring)
    isinf(τ) && return one(float(R))
    τ <= zero(τ) && return exp(-R)
    S = ccdf(generation_time, τ)
    return exp(-R * S)
end

"""
    end_of_outbreak_probability(model::BranchingProcess, τ)

End-of-outbreak probability using the offspring and generation time
distributions of a `BranchingProcess`, `τ` days after the most recent case.

This assumes every case is reported. Under-reporting (a model with a
[`PerCaseObservation`](@ref)) needs the recursion of Thompson, Morgan &
Jansen (2019), which is not implemented, and gives an error; use a model
without an observation model for the full-reporting value.
"""
function end_of_outbreak_probability(model::Union{BranchingProcess, ModelSpec}, τ::Real)
    # Refuse under per-case under-reporting rather than silently using
    # the bare offspring: makes the missing `ρ < 1` case discoverable.
    _eoo_assert_full_reporting(observation(model))
    return end_of_outbreak_probability(
        single_type_offspring(model), _single_kernel(model), τ
    )
end

_eoo_assert_full_reporting(::NoObservation) = nothing
function _eoo_assert_full_reporting(::PerCaseObservation)
    throw(
        ArgumentError(
            "end_of_outbreak_probability under per-case under-reporting (ρ < 1) " *
                "is not implemented. The closed form here assumes full reporting; " *
                "the ρ < 1 case needs the Volterra recursion of Thompson, Morgan & " *
                "Jansen (2019). Evaluate on a model with no observation to compute " *
                "the ρ = 1 value."
        )
    )
end

"""
    end_of_outbreak_probability(R, k, gt, τs::AbstractVector)

End-of-outbreak probability for each of several clusters, given the days
since each cluster's most recent case. Returns one value per cluster, ready
to pass as `prob_concluded` to `loglikelihood(::ChainSizes, ...)`.
"""
function end_of_outbreak_probability(
        R::Real, k::Real, generation_time::Distribution,
        τs::AbstractVector{<:Real}
    )
    return [end_of_outbreak_probability(R, k, generation_time, τ) for τ in τs]
end
