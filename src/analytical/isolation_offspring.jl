# ── Isolation's analytic offspring effect ─────────────────────────────
#
# A case isolated at time T after infection keeps a secondary case due at
# time t with probability 1 for t < T, `post_isolation_transmission` (ε) for
# T <= t < T + duration (D), and 1 again after release at T + D. Averaged
# over the contact-interval distribution this is
# p(T) = F_g(T) + ε(F_g(T + D) - F_g(T)) + (1 - F_g(T + D)), which reduces to
# F_g(T) for perfect, unending isolation and to F_g(T) + ε(1 - F_g(T)) when D
# is infinite. A case never isolated at all (ineligible, or tests negative)
# keeps every secondary case, the same as T = Inf.
#
# Each case's own T is its own incubation period plus its own
# onset-to-isolation delay, both drawn independently, so the offspring
# generating function is the expectation of G(1 - p(T)(1 - s)) over cases: a
# mixture with no simpler closed form in general (a negative binomial only
# when every case shares the same p), computed here by numerical integration
# over the incubation period and the delay.

# A fixed value stands for itself; a distribution's mean is exact here
# because each use (test_sensitivity, prob_asymptomatic) only ever enters a
# Bernoulli draw's marginal probability, which is linear in it. A function of
# the individual has no fixed value to read without simulating.
_closed_form_value(x::Real) = float(x)
_closed_form_value(d::Distribution) = float(mean(d))
function _closed_form_value(x)
    throw(
        ArgumentError(
            "Isolation's analytic effect needs a number or a distribution here, " *
                "not a function of the individual; simulate instead."
        )
    )
end

# Fraction of cases that can be isolated through self-reporting at all
# (ignoring ContactTracing's own pathway, which has no closed form). An
# asymptomatic case has no onset time under either built-in eligibility rule,
# so `AllCases`' extra reach over `SymptomaticOnly` — eligibility for a case
# with no symptoms — never actually isolates one: `resolve_individual!`'s
# self-report time is `onset + delay`, `NaN` once `onset` is, so it never
# clears `isfinite(final)`. Both rules therefore have the same closed form, in
# the fraction of cases with an onset time at all. A custom rule's own
# reach is not knowable in general without simulating.
_eligible_fraction(::Union{SymptomaticOnly, AllCases}, prob_asymptomatic) =
    1.0 - _closed_form_value(prob_asymptomatic)
function _eligible_fraction(elig::IsolationEligibility, prob_asymptomatic)
    throw(
        ArgumentError(
            "Isolation's analytic effect only covers SymptomaticOnly and AllCases " *
                "eligibility (got $(typeof(elig))); simulate instead."
        )
    )
end

# `duration` enters the keep-probability nonlinearly (through F_g(T + D)), so
# unlike `test_sensitivity` and `prob_asymptomatic` above, its mean would not
# give an exact answer: it must be a fixed value.
_isolation_duration(D::Real) = float(D)
function _isolation_duration(D)
    throw(
        ArgumentError(
            "Isolation's analytic effect needs a fixed `duration` (a number, or " *
                "`Inf`), not a distribution or a function; simulate instead."
        )
    )
end

# The `ClinicalPresentation` among a model's `attributes`, or `nothing` if
# there is none.
_clinical_presentation(::NoAttributes) = nothing
_clinical_presentation(c::ClinicalPresentation) = c
function _clinical_presentation(attrs::Union{Tuple, AbstractVector})
    i = findfirst(a -> a isa ClinicalPresentation, attrs)
    return i === nothing ? nothing : attrs[i]
end
_clinical_presentation(::Any) = nothing

# Expectation of `f` over a fixed value, or over a continuous or
# discrete-nonparametric distribution (`_mixture_expectation`, shared with
# `ClusterMixed`'s chain-level mixing). Isolation's own random quantities —
# the incubation period, and the delay when given as a distribution — are one
# of these in every documented use.
_closed_form_expectation(x::Real, f) = f(x)
_closed_form_expectation(d::ContinuousUnivariateDistribution, f) = _mixture_expectation(f, d)
_closed_form_expectation(d::DiscreteNonParametric, f) = _mixture_expectation(f, d)
function _closed_form_expectation(x, f)
    throw(
        ArgumentError(
            "Isolation's analytic effect needs a number or a distribution here, " *
                "not a function of the individual; simulate instead."
        )
    )
end

# The generation-time distribution's CDF for a case with incubation period
# `a`: `get_generation_time` already dispatches a fixed distribution, a
# function of the individual (as `incubation_linked_generation_time` is), and
# `NoGenerationTime()`, so a minimal probe individual with that incubation
# period is all this needs.
function _isolation_gen_time_cdf(kernel, a)
    probe = Individual(; id = 1, infection_time = 0.0, state = Dict(:onset_time => a))
    gt = get_generation_time(kernel, probe)
    gt isa NoGenerationTime && throw(
        ArgumentError(
            "Isolation's analytic effect needs a generation-time distribution; " *
                "this model has none."
        )
    )
    return t -> cdf(gt, t)
end

# The probability that a secondary case otherwise due at time `T` after
# infection still happens, given the generation-time CDF `Fg` (for this
# case's own incubation period), leaky transmission `ε` and `duration` `D`.
function _isolation_keep_probability(Fg, ε::Real, D::Real, T::Real)
    FgT = Fg(T)
    isfinite(D) || return FgT + ε * (1.0 - FgT)
    FgTD = Fg(T + D)
    return FgT + ε * (FgTD - FgT) + (1.0 - FgTD)
end

"""
    analytic_offspring_effect(iso::Isolation, offspring::DiscreteUnivariateDistribution, spec)

The offspring law under isolation: a case drawn from `offspring`, thinned by
the probability that each of its secondary cases still happens (see the
comment above this method for the formula), averaged over `spec`'s incubation
period and this isolation's own `onset_to_isolation_delay`, and over whether
the case is isolated at all (`eligibility` and `test_sensitivity`). Returns a
[`GeneratingFunctionOffspring`](@ref).

Needs `spec`'s `attributes` to include [`clinical_presentation`](@ref) and its
process to have a generation-time distribution; `duration` must be a fixed
number (or `Inf`), `onset_to_isolation_delay` and `test_sensitivity` each a
number or a distribution, and `eligibility` [`SymptomaticOnly`](@ref) or
[`AllCases`](@ref). Anything else has no closed form here, and the generic
default throws, pointing at simulation instead.
"""
function analytic_offspring_effect(
        iso::Isolation, offspring::DiscreteUnivariateDistribution, spec
    )
    cp = _clinical_presentation(attributes(spec))
    cp === nothing && throw(
        ArgumentError(
            "Isolation's analytic effect needs the incubation period, from " *
                "attributes = clinical_presentation(...); simulate instead."
        )
    )
    kernel = _single_kernel(spec)
    incubation = cp.incubation_period
    delay = iso.onset_to_isolation_delay
    ε = iso.post_isolation_transmission
    D = _isolation_duration(iso.duration)
    q = _eligible_fraction(iso.eligibility, cp.prob_asymptomatic) *
        _closed_form_value(iso.test_sensitivity)

    # The keep probability for a case whose incubation period is `a` and
    # onset-to-isolation delay is `b`, isolation time T = a + b after
    # infection.
    keep_probability(a, b) =
        _isolation_keep_probability(_isolation_gen_time_cdf(kernel, a), ε, D, a + b)

    isolated_pgf(s) = _closed_form_expectation(
        incubation,
        a -> _closed_form_expectation(
            delay, b -> _pgf(offspring, 1.0 - keep_probability(a, b) * (1.0 - s))
        )
    )
    pgf(s) = (1.0 - q) * _pgf(offspring, s) + q * isolated_pgf(s)

    keep_mean = _closed_form_expectation(
        incubation, a -> _closed_form_expectation(delay, b -> keep_probability(a, b))
    )
    mean_ = (1.0 - q + q * keep_mean) * _law_mean(offspring)

    return GeneratingFunctionOffspring(pgf, mean_)
end
