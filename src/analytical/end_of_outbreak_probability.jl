# ── End-of-outbreak probability ─────────────────────────────────────
#
# π(τ; R, k, generation_time) is the probability that a single case causes
# no further cases, given `τ` time has elapsed since it occurred. Used as a
# per-case `is_finished` weight in the real-time chain-size mixture
# likelihood (see `loglikelihood(::ChainSizes, ::Distribution)`); exact for a
# whole cluster only when every other case in it can no longer transmit
# (for example a cluster of one). The vector method below combines the
# per-case values, each raised to a power that reflects how many secondary
# cases it is already known to have caused, into the probability for the
# whole cluster.
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

Probability that a single case causes no further cases, given that it was
`τ` days ago. Assumes `NegBin(R, k)` offspring, a generation time
distribution `generation_time` (days) and that every case is reported.
Following Thompson, Morgan & Jansen (2019, Phil Trans B), it is

    ((k + R · (1 − S(τ))) / (k + R))^k

where `S(τ)` is the probability that a generation time exceeds `τ`. It rises
from the probability of a case infecting nobody at `τ = 0` towards 1 as `τ`
grows.

This is a per-case probability. It is the probability that a whole cluster is
over only when the other cases in the cluster can no longer transmit either,
for example a cluster of one case, or one where `τ` is measured from its
earliest rather than its most recent case. For a cluster of several cases
still capable of transmitting, pass every case's own `τ` to the vector method
below instead, which combines them properly; using this method with only the
most recent case's `τ` overstates the probability that the cluster is over.

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
    end_of_outbreak_probability(spec::ModelSpec, τ)

End-of-outbreak probability using the offspring and generation time
distributions of a `BranchingProcess`, `τ` days after the most recent case.

This assumes every case is reported. Under-reporting (a [`Reporting`](@ref)
step with a probability below 1, or any deprecated
[`PerCaseObservation`](@ref)) needs the recursion of Thompson, Morgan & Jansen
(2019), which is not implemented, and gives an error; use a model that reports
every case for the full-reporting value.

For a `ModelSpec`, or a process that stores its own interventions, each
intervention's [`EpiBranch.analytic_offspring_effect`](@ref) is applied to
the offspring distribution first. If an intervention has no such method, this
throws an error that names the simulation-based alternative.
"""
function end_of_outbreak_probability(model::Union{BranchingProcess, ModelSpec}, τ::Real)
    # Refuse under per-case under-reporting rather than silently using
    # the bare offspring: makes the missing `ρ < 1` case discoverable.
    _eoo_assert_full_reporting(observation(model), _progression(model))
    off = _offspring_through_interventions(model, single_type_offspring(model))
    return end_of_outbreak_probability(off, _single_kernel(model), τ)
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

function _eoo_assert_full_reporting(obs, progression)
    _eoo_assert_full_reporting(obs)
    ρ = _reporting_probability(progression)
    (ρ === nothing || ρ >= 1.0) && return nothing
    throw(
        ArgumentError(
            "end_of_outbreak_probability under per-case under-reporting (ρ < 1) " *
                "is not implemented. The closed form here assumes full reporting; " *
                "the ρ < 1 case needs the Volterra recursion of Thompson, Morgan & " *
                "Jansen (2019). Evaluate on a model with a `Reporting` step at " *
                "probability 1, or none, to compute the ρ = 1 value."
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

"""
    end_of_outbreak_probability(R, k, gt, τs::AbstractVector, secondary_cases::AbstractVector{<:Integer})
    end_of_outbreak_probability(offspring::NegativeBinomial, gt, τs::AbstractVector, secondary_cases::AbstractVector{<:Integer})

Probability that a whole cluster is over, given the time since onset of every
one of its cases (`τs`) and the number of secondary cases each is already
known to have caused (`secondary_cases`). A cluster is over only once none of
its cases transmits again: this is the product, over its cases, of
[`end_of_outbreak_probability`](@ref)`(R, k, gt, τs[i])` raised to the power
`(k + secondary_cases[i]) / k`. Under `NegBin(R, k)` offspring, a case already
known to have caused more secondary cases is more likely to cause further
ones too, so it contributes less to the cluster being over than a case with
the same `τ` and no known onward transmission.

Pass `zeros(Int, length(τs))` for `secondary_cases` when the transmission
tree within the cluster is unknown; this reduces to the plain product of the
per-case values,
`prod(end_of_outbreak_probability(R, k, gt, τ) for τ in τs)`.

# Examples

```julia
offspring = NegBin(0.95, 0.18)
gt = Gamma(2.7, 5.7)             # mean about 15 days
onsets = [0.0, 3.0, 5.0, 6.0]    # four cases
t = 10.0
τs = t .- onsets
secondary_cases = [2, 1, 0, 0]   # case 1 infected two, case 2 one
end_of_outbreak_probability(offspring, gt, τs, secondary_cases)
```
"""
function end_of_outbreak_probability(
        R::Real, k::Real, generation_time::Distribution,
        τs::AbstractVector{<:Real}, secondary_cases::AbstractVector{<:Integer}
    )
    length(secondary_cases) == length(τs) || throw(
        ArgumentError(
            "secondary_cases must have the same length as τs " *
                "($(length(τs))); got $(length(secondary_cases))"
        )
    )
    all(>=(0), secondary_cases) ||
        throw(ArgumentError("secondary_cases must be non-negative"))
    return prod(
        end_of_outbreak_probability(R, k, generation_time, τ)^((k + a) / k)
            for (τ, a) in zip(τs, secondary_cases)
    )
end

function end_of_outbreak_probability(
        offspring::NegativeBinomial, generation_time::Distribution,
        τs::AbstractVector{<:Real}, secondary_cases::AbstractVector{<:Integer}
    )
    return end_of_outbreak_probability(
        mean(offspring), offspring.r, generation_time, τs, secondary_cases
    )
end
