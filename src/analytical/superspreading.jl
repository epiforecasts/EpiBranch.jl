"""
    proportion_transmission(R::Real, k::Real; prop_cases::Real=0.2)

Compute the proportion of transmission caused by the most infectious
fraction `prop_cases` of cases, under a Negative Binomial offspring
distribution with mean `R` and dispersion `k`.

This is the "80/20 rule" metric for superspreading: with `prop_cases=0.2`,
returns the proportion of all transmission events caused by the top 20% of
transmitters.

The result depends only on the dispersion `k`; `R` is accepted for interface
consistency but does not affect it, because the Lorenz curve of the underlying
`Gamma(k, R/k)` is scale-invariant in the mean. Two calls with the same `k` and
different `R` return the same value.

Computed via the regularised incomplete beta function.
"""
function proportion_transmission(R::Real, k::Real; prop_cases::Real = 0.2)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    0.0 < prop_cases < 1.0 ||
        throw(ArgumentError("prop_cases must be in (0, 1), got $prop_cases"))

    # The proportion of transmission from the top `prop_cases` fraction
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
    proportion_transmission(spec::ModelSpec; prop_cases=0.2)

Proportion of transmission from the most infectious fraction of cases,
extracted from the model's offspring distribution (must be NegativeBinomial).

For a `ModelSpec`, the offspring law is folded through every intervention's
[`EpiBranch.analytic_offspring_effect`](@ref) first; a spec carrying an
intervention without one throws, naming the simulation-based alternative,
rather than returning the bare process's proportion as if the interventions
were not there.
"""
function proportion_transmission(d::NegativeBinomial; prop_cases::Real = 0.2)
    return proportion_transmission(mean(d), d.r; prop_cases)
end

function proportion_transmission(d::Poisson; prop_cases::Real = 0.2)
    return proportion_transmission(mean(d), 1.0e6; prop_cases)
end

function proportion_transmission(d::Distribution; prop_cases::Real = 0.2)
    throw(ArgumentError("proportion_transmission not defined for $(typeof(d)). Use NegativeBinomial or Poisson."))
end

function proportion_transmission(model::TransmissionModel; prop_cases::Real = 0.2)
    return proportion_transmission(single_type_offspring(model); prop_cases)
end

function proportion_transmission(spec::ModelSpec; prop_cases::Real = 0.2)
    off = _offspring_through_interventions(spec, single_type_offspring(spec))
    return proportion_transmission(off; prop_cases)
end

# ── Proportion of cases responsible for a share of transmission ──────

"""
    proportion_cases_individual(R::Real, k::Real; prop_transmission::Real=0.8)

Inverse of [`proportion_transmission`](@ref): the proportion of cases
responsible for a given proportion `prop_transmission` of transmission,
under the continuous Gamma approximation to individual reproduction
numbers.

As with `proportion_transmission`, the result depends only on the
dispersion `k`; `R` is accepted for interface consistency but does not
affect it.

This is not the same question as [`proportion_cases_offspring`](@ref),
which ranks realised, integer offspring counts rather than continuous
individual reproduction numbers, and can give a substantially different
answer for the same `R` and `k`.
"""
function proportion_cases_individual(R::Real, k::Real; prop_transmission::Real = 0.8)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    0.0 < prop_transmission < 1.0 ||
        throw(ArgumentError("prop_transmission must be in (0, 1), got $prop_transmission"))

    # Invert the Lorenz curve used by `proportion_transmission`: find the
    # top-`prop_transmission` share of transmission first, then read off the
    # proportion of cases that produced it.
    g1 = Gamma(k + 1.0, 1.0)
    x = quantile(g1, 1.0 - prop_transmission)
    g = Gamma(k, 1.0)
    return 1.0 - cdf(g, x)
end

"""
    proportion_cases_individual(d::NegativeBinomial; prop_transmission=0.8)

Proportion of cases responsible for `prop_transmission` of transmission,
extracted from a Negative Binomial offspring distribution.
"""
function proportion_cases_individual(d::NegativeBinomial; prop_transmission::Real = 0.8)
    return proportion_cases_individual(mean(d), d.r; prop_transmission)
end

function proportion_cases_individual(d::Poisson; prop_transmission::Real = 0.8)
    return proportion_cases_individual(mean(d), 1.0e6; prop_transmission)
end

function proportion_cases_individual(d::Distribution; prop_transmission::Real = 0.8)
    throw(ArgumentError("proportion_cases_individual not defined for $(typeof(d)). Use NegativeBinomial or Poisson."))
end

"""
    proportion_cases_individual(model::BranchingProcess; prop_transmission=0.8)

Proportion of cases responsible for `prop_transmission` of transmission,
extracted from the model's offspring distribution (must be NegativeBinomial
or Poisson).
"""
function proportion_cases_individual(
        model::Union{TransmissionModel, ModelSpec};
        prop_transmission::Real = 0.8
    )
    return proportion_cases_individual(single_type_offspring(model); prop_transmission)
end

"""
    proportion_cases_offspring(d::DiscreteUnivariateDistribution; prop_transmission::Real=0.8)

The proportion of cases responsible for a given proportion `prop_transmission`
of transmission, computed from the realised, integer offspring counts of any
discrete offspring distribution `d` with finite mean — the version usually
reported alongside the "80/20 rule".

Cases are ranked by their actual number of secondary cases, from the most
infectious downwards; the count at the crossing threshold is split
fractionally between the responsible and non-responsible groups so the
target share of transmission is met exactly, rather than rounded to a whole
count.

This is not the same question as [`proportion_cases_individual`](@ref),
which uses a continuous Gamma approximation to individual reproduction
numbers rather than realised offspring counts, and the two can differ
substantially for the same offspring distribution — report both, clearly
labelled, rather than picking one.
"""
function proportion_cases_offspring(d::DiscreteUnivariateDistribution; prop_transmission::Real = 0.8)
    0.0 < prop_transmission < 1.0 ||
        throw(ArgumentError("prop_transmission must be in (0, 1), got $prop_transmission"))

    μ = _law_mean(d)
    isfinite(μ) || throw(ArgumentError("offspring distribution must have a finite mean"))
    μ > 0 || throw(
        ArgumentError("offspring distribution must have a positive mean to define a transmission share")
    )

    lo, hi = _series_range(d)

    cum_cases = 0.0
    cum_transmission = 0.0
    for x in hi:-1:lo
        p_x = pdf(d, x)
        transmission_x = x * p_x / μ
        if cum_transmission + transmission_x >= prop_transmission
            remaining = prop_transmission - cum_transmission
            frac = transmission_x > 0 ? remaining / transmission_x : 0.0
            return cum_cases + frac * p_x
        end
        cum_cases += p_x
        cum_transmission += transmission_x
    end
    return cum_cases
end

"""
    proportion_cases_offspring(R::Real, k::Real; prop_transmission::Real=0.8)

Proportion of cases responsible for `prop_transmission` of transmission,
computed from the realised offspring counts of a Negative Binomial
distribution with mean `R` and dispersion `k`.
"""
function proportion_cases_offspring(R::Real, k::Real; prop_transmission::Real = 0.8)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    return proportion_cases_offspring(NegBin(R, k); prop_transmission)
end

"""
    proportion_cases_offspring(model::BranchingProcess; prop_transmission=0.8)

Proportion of cases responsible for `prop_transmission` of transmission,
computed from the model's realised offspring distribution.
"""
function proportion_cases_offspring(
        model::Union{TransmissionModel, ModelSpec};
        prop_transmission::Real = 0.8
    )
    return proportion_cases_offspring(single_type_offspring(model); prop_transmission)
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
function proportion_cluster_size(R::Real, k::Real; cluster_size::Int = 10)
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
function proportion_cluster_size(d::NegativeBinomial; cluster_size::Int = 10)
    return proportion_cluster_size(mean(d), d.r; cluster_size)
end

"""
    proportion_cluster_size(model::BranchingProcess; cluster_size=10)

Proportion of cases from large clusters for a branching process model.
"""
function proportion_cluster_size(
        model::Union{TransmissionModel, ModelSpec};
        cluster_size::Int = 10
    )
    d = single_type_offspring(model)
    d isa NegativeBinomial || throw(
        ArgumentError(
            "proportion_cluster_size requires NegativeBinomial offspring"
        )
    )
    return proportion_cluster_size(d; cluster_size)
end

# ── Network-adjusted reproduction number ─────────────────────────────

"""
    heterogeneous_contact_R(mean_contacts, sd_contacts, duration, prob_transmission)

Compute the basic reproduction number adjusted for heterogeneous contact
patterns in a network.

Returns a named tuple `(R=..., R_net=...)`:
- `R`: unadjusted, assuming homogeneous mixing (`β × mean_contacts × duration`)
- `R_net`: network-adjusted, accounting for contact heterogeneity
  (`β × duration × (mean + variance/mean)`)

The adjustment reflects that high-contact individuals both acquire and
transmit more, amplifying R beyond what homogeneous mixing predicts.

This is a mean-field, configuration-model result: it depends only on the
mean and variance of the contact (degree) distribution and assumes no
clustering. It is an analytical summary, distinct from an explicit
network simulation, which transmits over a graph that may have the
clustering this formula assumes away. It is a direct port of
`calc_network_R` in superspreading (Lambert et al.,
https://github.com/epiverse-trace/superspreading, MIT).
"""
function heterogeneous_contact_R(
        mean_contacts::Real, sd_contacts::Real,
        duration::Real, prob_transmission::Real
    )
    mean_contacts >= 0 || throw(ArgumentError("mean_contacts must be ≥ 0"))
    sd_contacts >= 0 || throw(ArgumentError("sd_contacts must be ≥ 0"))
    duration > 0 || throw(ArgumentError("duration must be positive"))
    0.0 <= prob_transmission <= 1.0 ||
        throw(ArgumentError("prob_transmission must be in [0, 1]"))

    R = prob_transmission * mean_contacts * duration

    var_contacts = sd_contacts^2
    R_net = if mean_contacts > 0
        prob_transmission * duration * (mean_contacts + var_contacts / mean_contacts)
    else
        0.0
    end

    return (R = R, R_net = R_net)
end
