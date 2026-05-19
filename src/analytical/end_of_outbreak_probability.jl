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

Probability that a cluster has finished, given the most recent case
was observed `τ` time units ago, under NegBin(R, k) offspring and
full reporting. Returns a value in `[0, 1]`.
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

Convenience overloads that read `R` (and `k` when applicable) directly
from a `Distributions` offspring object. The Poisson form uses the
`k → ∞` limit `π(τ) = exp(−R · S(τ))`.
"""
function end_of_outbreak_probability(
        offspring::NegativeBinomial, generation_time::Distribution,
        τ::Real)
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
    end_of_outbreak_probability(model::TransmissionModel, τ)

Convenience that reads the offspring and generation-time distributions
from a `BranchingProcess` (or any `TransmissionModel` exposing those
fields).
"""
function end_of_outbreak_probability(model::TransmissionModel, τ::Real)
    return end_of_outbreak_probability(single_type_offspring(model), model.generation_time, τ)
end

"""
    end_of_outbreak_probability.(R, k, gt, τs::AbstractVector)

Element-wise broadcast for a vector of τ values, returning a `Vector`
of the same length. Useful for populating the `pi` field of
[`ChainSizes`](@ref).
"""
function end_of_outbreak_probability(R::Real, k::Real, generation_time::Distribution,
        τs::AbstractVector{<:Real})
    return [end_of_outbreak_probability(R, k, generation_time, τ) for τ in τs]
end
