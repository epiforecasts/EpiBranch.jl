"""
    NegBin(R, k)

Convenience constructor for a Negative Binomial offspring distribution
parameterised by mean reproduction number `R` and dispersion parameter `k`.

A `NegativeBinomial` from Distributions.jl is returned, with mean `R` and
variance `R + R²/k`.
"""
function NegBin(R::Real, k::Real)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    p = k / (k + R)
    return NegativeBinomial(k, p)
end

"""
    ringbp_generation_time(; presymptomatic_fraction=0.3, omega=2.0)

Return a function suitable for the `generation_time` field of a
`BranchingProcess`, implementing ringbp's incubation-linked generation
time model.

An incubation period (Float64) is accepted by the returned function, and a
truncated normal approximation to the skew-normal SN(ξ, ω, α) is produced,
where ξ = incubation period, and α is chosen so that the fraction of generation
times shorter than the incubation period equals `presymptomatic_fraction`.

Usage:
```julia
model = BranchingProcess(
    NegBin(2.5, 0.16),
    ringbp_generation_time(presymptomatic_fraction=0.3)
)
```
"""
function ringbp_generation_time(; presymptomatic_fraction::Real=0.3,
                                  omega::Real=2.0)
    0.0 < presymptomatic_fraction < 1.0 || throw(ArgumentError(
        "presymptomatic_fraction must be in (0, 1), got $presymptomatic_fraction"))
    omega > 0.0 || throw(ArgumentError("omega must be positive, got $omega"))

    # Compute skew-normal alpha from presymptomatic fraction
    # For SN(xi, omega, alpha): P(X < xi) = 0.5 - arctan(alpha)/π
    # => alpha = tan(π(0.5 - presymp_frac))
    alpha = tan(Float64(π) * (0.5 - Float64(presymptomatic_fraction)))
    delta = alpha / sqrt(1.0 + alpha^2)
    om = Float64(omega)

    return function (inc_period::Float64)
        mu = inc_period + om * delta * sqrt(2.0 / π)
        sigma = om * sqrt(1.0 - 2.0 * delta^2 / π)
        truncated(Normal(mu, max(sigma, 0.01)), 0.0, Inf)
    end
end
