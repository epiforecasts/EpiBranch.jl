"""
    proportion_transmission(R::Real, k::Real; prop_cases::Real=0.2)

Proportion of all transmission caused by the most infectious fraction
`prop_cases` of cases, with `NegBin(R, k)` offspring. This is the "80/20 rule"
measure of superspreading: with `prop_cases = 0.2`, the share of transmission
caused by the 20% most infectious cases.

Cases are ranked by their individual reproduction number (the expected number
of people they infect, Gamma-distributed with mean `R` and shape `k`). The
result depends only on `k`, so two calls with the same `k` and different `R`
give the same value.

# Examples

```julia
proportion_transmission(2.5, 0.16)   # share of transmission from the top 20%
```
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

Proportion of transmission caused by the most infectious fraction
`prop_cases` of cases, using the model's offspring distribution, which must
be negative binomial (or Poisson). Interventions and population
characteristics in a `ModelSpec` are ignored; for transmission under control
measures, simulate and count secondary cases instead.
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

function proportion_transmission(
        model::Union{TransmissionModel, ModelSpec};
        prop_cases::Real = 0.2
    )
    return proportion_transmission(single_type_offspring(model); prop_cases)
end

# ── Proportion of cases responsible for a share of transmission ──────

"""
    proportion_cases_individual(R::Real, k::Real; prop_transmission::Real=0.8)

Proportion of cases responsible for a share `prop_transmission` of all
transmission (for example the 80% in the "80/20 rule"), with `NegBin(R, k)`
offspring, ranking cases by their individual reproduction number. The inverse
of [`proportion_transmission`](@ref). It depends only on `k`.

!!! note "Two ways to rank cases"
    `proportion_cases_individual` ranks cases by their individual
    reproduction number, a continuous Gamma-distributed quantity.
    [`proportion_cases_offspring`](@ref) ranks them by the number of people
    they actually infected, a whole number. The two answer slightly different
    questions and can differ substantially for the same `R` and `k`. When
    reporting, say which one was used.
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

Proportion of cases responsible for a share `prop_transmission` of
transmission, ranking by individual reproduction number, for a negative
binomial offspring distribution.
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

Proportion of cases responsible for a share `prop_transmission` of
transmission, ranking by individual reproduction number, using the model's
offspring distribution (negative binomial or Poisson).
"""
function proportion_cases_individual(
        model::Union{TransmissionModel, ModelSpec};
        prop_transmission::Real = 0.8
    )
    return proportion_cases_individual(single_type_offspring(model); prop_transmission)
end

"""
    proportion_cases_offspring(d::DiscreteUnivariateDistribution; prop_transmission::Real=0.8)

Proportion of cases responsible for a share `prop_transmission` of all
transmission, ranking cases by the number of people they actually infected.
This answers "what share of infected people caused 80% of onward
infections?". Works for any discrete offspring distribution `d` with a finite
mean.

Cases are ranked from the most secondary cases downwards. Where the target
share is crossed part-way through cases with the same count, those cases are
counted fractionally, so the target share is met exactly.

See the note in [`proportion_cases_individual`](@ref) on how the two
functions differ.
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

Proportion of cases responsible for a share `prop_transmission` of
transmission, ranking by number of people infected, with `NegBin(R, k)`
offspring.
"""
function proportion_cases_offspring(R::Real, k::Real; prop_transmission::Real = 0.8)
    R > 0 || throw(ArgumentError("R must be positive, got $R"))
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    return proportion_cases_offspring(NegBin(R, k); prop_transmission)
end

"""
    proportion_cases_offspring(model::BranchingProcess; prop_transmission=0.8)

Proportion of cases responsible for a share `prop_transmission` of
transmission, ranking by number of people infected, using the model's
offspring distribution.
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

Proportion of all secondary cases caused by cases who each infected at least
`cluster_size` people (superspreading events), with `NegBin(R, k)` offspring.
With strong superspreading (small `k`), a large share of cases come from a
few such events.

Here `cluster_size` is the number of secondary cases of one infector, not the
size of a transmission chain as in [`chain_size_distribution`](@ref).

It is `E[X; X ≥ c] / E[X]` for the offspring distribution `X` and
`c = cluster_size`.
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

Proportion of secondary cases caused by cases who each infected at least
`cluster_size` people, for a negative binomial offspring distribution.
"""
function proportion_cluster_size(d::NegativeBinomial; cluster_size::Int = 10)
    return proportion_cluster_size(mean(d), d.r; cluster_size)
end

"""
    proportion_cluster_size(model::BranchingProcess; cluster_size=10)

Proportion of secondary cases caused by cases who each infected at least
`cluster_size` people, using the model's offspring distribution.
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

Basic reproduction number when people differ in how many contacts they have.
People with many contacts are both more likely to be infected and to infect
more others, which raises R above what the average number of contacts
suggests.

- `mean_contacts`, `sd_contacts`: mean and standard deviation of the number
  of contacts per person per day.
- `duration`: duration of infectiousness, in days.
- `prob_transmission`: probability of transmission per contact.

Returns `(R = ..., R_net = ...)`:

- `R`: assuming everyone has the average number of contacts,
  `prob_transmission × mean_contacts × duration`.
- `R_net`: allowing for the variation in contacts,
  `prob_transmission × duration × (mean + variance / mean)`.

This assumes contacts are formed at random given each person's number of
contacts, with no clustering (friends of friends being friends). A
simulation on an explicit network with clustering, such as `NetworkProcess`,
can give a different answer. It is a port of `calc_network_R` in the R
package superspreading (Lambert et al.,
https://github.com/epiverse-trace/superspreading, MIT).

# Examples

```julia
heterogeneous_contact_R(10.0, 15.0, 5.0, 0.01)
```
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
