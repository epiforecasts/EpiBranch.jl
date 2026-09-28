"""
    expected_ve(estimator, mode, efficacy, cumulative_hazard; vaccine_fraction = 0.5)

The value an estimator of vaccine efficacy converges to in a large trial where
every participant faces the same force of infection from outside the trial,
protection is in place from enrolment, and everyone is followed to the same
end. `cumulative_hazard` is the force of infection integrated over follow-up,
Λ = ∫λ(t)dt, faced by an unvaccinated participant; `mode` is `LeakyMode()` or
`AllOrNothingMode()`.

| | `LeakyMode()` | `AllOrNothingMode()` |
|---|---|---|
| [`RiskRatio`](@ref) | 1 − (1 − e^{−(1−φ)Λ}) / (1 − e^{−Λ}) | θ |
| [`CoxHazardRatio`](@ref) | φ | root of the expected Cox score |

Under a leaky vaccine the hazard ratio is constant, so the Cox estimate is
unbiased, while the ratio of attack rates tends to 1 as Λ grows. Under an
all-or-nothing vaccine the ratio of attack rates is exact, while the hazard
ratio falls over follow-up as unprotected vaccinees are infected, and the Cox
estimate lies above θ; the Cox limit depends on `vaccine_fraction`, the
proportion of participants in the vaccine arm, and is computed by quadrature
(Smith, Rodrigues & Fine 1984; Halloran, Haber & Longini 1992).

# Examples

```jldoctest; setup = :(using EpiTrial, EpiBranch)
julia> round(expected_ve(RiskRatio(), LeakyMode(), 0.6, 1.0), digits = 3)
0.478
```
"""
function expected_ve end

function expected_ve(::RiskRatio, ::LeakyMode, efficacy, cumulative_hazard;
        vaccine_fraction = 0.5)
    Λ = cumulative_hazard
    return 1 - (1 - exp(-(1 - efficacy) * Λ)) / (1 - exp(-Λ))
end

function expected_ve(::RiskRatio, ::AllOrNothingMode, efficacy, cumulative_hazard;
        vaccine_fraction = 0.5)
    return float(efficacy)
end

function expected_ve(::CoxHazardRatio, ::LeakyMode, efficacy, cumulative_hazard;
        vaccine_fraction = 0.5)
    return float(efficacy)
end

function expected_ve(::CoxHazardRatio, ::AllOrNothingMode, efficacy, cumulative_hazard;
        vaccine_fraction = 0.5)
    θ, Λ, p = efficacy, cumulative_hazard, vaccine_fraction
    (θ == 0 || θ == 1) && return float(θ)
    # With u the cumulative hazard so far, a control participant is still
    # uninfected with probability e^{-u} and a vaccinee with θ + (1-θ)e^{-u};
    # events occur at densities e^{-u} and (1-θ)e^{-u}. The Cox estimate
    # converges to the root in β of the expected score.
    function score(β)
        integrand(u) = begin
            s0, s1 = exp(-u), θ + (1 - θ) * exp(-u)
            f0, f1 = exp(-u), (1 - θ) * exp(-u)
            w = p * s1 * exp(β) / (p * s1 * exp(β) + (1 - p) * s0)
            p * f1 - w * (p * f1 + (1 - p) * f0)
        end
        # An absolute tolerance, because near the root the integral is close to
        # zero and a relative one is never met.
        return first(quadgk(integrand, 0, Λ; atol = 1e-12))
    end
    # The score decreases in β; bracket the root and bisect.
    lo, hi = log(1 - θ) - 10, 0.0
    for _ in 1:60
        mid = (lo + hi) / 2
        score(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return 1 - exp((lo + hi) / 2)
end
