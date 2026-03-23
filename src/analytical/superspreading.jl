"""
    proportion_transmission(R::Real, k::Real; prop_cases::Real=0.2)

Compute the proportion of transmission caused by the most infectious
fraction `prop_cases` of cases, under a Negative Binomial offspring
distribution with mean `R` and dispersion `k`.

This is the "80/20 rule" metric for superspreading: with `prop_cases=0.2`,
returns the proportion of all transmission events caused by the top 20% of
transmitters.

Computed via the regularised incomplete beta function.
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

proportion_transmission(d::Distribution; prop_cases::Real=0.2) =
    throw(ArgumentError("proportion_transmission not defined for $(typeof(d)). Use NegativeBinomial or Poisson."))

function proportion_transmission(model::BranchingProcess; prop_cases::Real=0.2)
    return proportion_transmission(_single_type_offspring(model); prop_cases)
end

# ── Proportion of cases from large clusters ──────────────────────────

"""
    proportion_cluster_size(R, k; cluster_size=10)

Proportion of secondary cases that arise from transmission events where
the infector caused at least `cluster_size` secondary cases.

This quantifies case concentration: with high overdispersion (low k),
a large fraction of cases come from a few superspreading events.

Uses the tail expectation of the NegBin distribution:
    E[X | X ≥ c] × P(X ≥ c) / E[X]
"""
function proportion_cluster_size(R::Real, k::Real; cluster_size::Int=10)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    cluster_size >= 1 || throw(ArgumentError("cluster_size must be ≥ 1, got $cluster_size"))

    nb = NegBin(R, k)

    # Proportion of all secondary cases from infectors with ≥ cluster_size cases
    # = Σ_{x≥c} x·P(X=x) / E[X]
    # = 1 - Σ_{x=0}^{c-1} x·P(X=x) / R
    tail_expectation = 0.0
    for x in 0:(cluster_size - 1)
        tail_expectation += x * pdf(nb, x)
    end
    return 1.0 - tail_expectation / R
end

"""
    proportion_cluster_size(d::NegativeBinomial; cluster_size=10)

Proportion of cases from large clusters for a NegBin offspring distribution.
"""
proportion_cluster_size(d::NegativeBinomial; cluster_size::Int=10) =
    proportion_cluster_size(mean(d), d.r; cluster_size)

"""
    proportion_cluster_size(model::BranchingProcess; cluster_size=10)

Proportion of cases from large clusters for a branching process model.
"""
function proportion_cluster_size(model::BranchingProcess; cluster_size::Int=10)
    d = _single_type_offspring(model)
    d isa NegativeBinomial || throw(ArgumentError(
        "proportion_cluster_size requires NegativeBinomial offspring"))
    return proportion_cluster_size(d; cluster_size)
end

# ── Network-adjusted reproduction number ─────────────────────────────

"""
    network_R(mean_contacts, sd_contacts, duration, prob_transmission)

Compute the basic reproduction number adjusted for heterogeneous contact
patterns in a network.

Returns a named tuple `(R=..., R_net=...)`:
- `R`: unadjusted, assuming homogeneous mixing (`β × mean_contacts × duration`)
- `R_net`: network-adjusted, accounting for contact heterogeneity
  (`β × duration × (mean + variance/mean)`)

The adjustment reflects that high-contact individuals both acquire and
transmit more, amplifying R beyond what homogeneous mixing predicts.
"""
function network_R(mean_contacts::Real, sd_contacts::Real,
                   duration::Real, prob_transmission::Real)
    mean_contacts >= 0 || throw(ArgumentError("mean_contacts must be ≥ 0"))
    sd_contacts >= 0 || throw(ArgumentError("sd_contacts must be ≥ 0"))
    duration > 0 || throw(ArgumentError("duration must be positive"))
    0.0 <= prob_transmission <= 1.0 || throw(ArgumentError("prob_transmission must be in [0, 1]"))

    R = prob_transmission * mean_contacts * duration

    var_contacts = sd_contacts^2
    R_net = if mean_contacts > 0
        prob_transmission * duration * (mean_contacts + var_contacts / mean_contacts)
    else
        0.0
    end

    return (R=R, R_net=R_net)
end
