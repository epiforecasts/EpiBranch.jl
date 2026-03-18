"""
    proportion_transmission(R::Real, k::Real; prop_cases::Real=0.2)

Compute the proportion of transmission caused by the most infectious
fraction `prop_cases` of cases, under a Negative Binomial offspring
distribution with mean `R` and dispersion `k`.

This is the "80/20 rule" metric for superspreading: with `prop_cases=0.2`,
returns the proportion of all transmission events caused by the top 20% of
transmitters.

Uses the incomplete beta function via the regularised beta function.
"""
function proportion_transmission(R::Real, k::Real; prop_cases::Real=0.2)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    0.0 < prop_cases < 1.0 || throw(ArgumentError("prop_cases must be in (0, 1), got $prop_cases"))

    # The proportion of transmission from the top (1 - prop_cases) fraction
    # is 1 - I_x(k+1, 0) where x is the quantile of the Gamma distribution
    # and I_x is the regularised incomplete beta function.
    #
    # The offspring NegBin(k, p) has the same top-fraction transmission as
    # a Gamma(k, R/k) continuous approximation.
    #
    # Proportion of transmission from bottom `prop_cases` fraction:
    # = 1 - beta_inc(k+1, 0, q) / B(k+1, 0)  — but this needs care
    #
    # Actually: use the Lorenz curve of the Gamma distribution.
    # Bottom q fraction of cases (by infectiousness) produces fraction:
    #   L(q) = gamma_inc_lower(k+1, gamma_inc_inv(k, q) * k/R * R/k) / Γ(k+1)
    #        = regularised_gamma_lower(k+1, quantile(Gamma(k, R/k), q) * k/R)
    #
    # Simplification: for Gamma(k, θ) where θ = R/k,
    #   L(q) = Γ_reg(k+1, Γ_inv(k, q))
    # where Γ_inv(k, q) is the inverse of the regularised lower incomplete gamma.

    # "Top prop_cases fraction" = top 20% of transmitters
    # Lorenz curve L(q) = fraction of transmission from the bottom q fraction
    # We want 1 - L(1 - prop_cases) = transmission from top prop_cases fraction
    g = Gamma(k, 1.0)
    x = quantile(g, 1.0 - prop_cases)

    g1 = Gamma(k + 1.0, 1.0)
    lorenz_bottom = cdf(g1, x)

    return 1.0 - lorenz_bottom
end

"""
    proportion_transmission(model::BranchingProcess; prop_cases=0.2)

Proportion of transmission from the most infectious fraction of cases,
extracted from the model's offspring distribution (must be NegativeBinomial).
"""
proportion_transmission(d::NegativeBinomial; prop_cases::Real=0.2) =
    proportion_transmission(mean(d), d.r; prop_cases)

proportion_transmission(d::Poisson; prop_cases::Real=0.2) =
    proportion_transmission(mean(d), 1e6; prop_cases)

function proportion_transmission(model::BranchingProcess; prop_cases::Real=0.2)
    model.offspring isa Distribution || throw(ArgumentError(
        "Analytical proportion_transmission only available for single-type models"))
    return proportion_transmission(model.offspring; prop_cases)
end
