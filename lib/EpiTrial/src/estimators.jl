"""
Base type for an estimator of vaccine efficacy from [`trial_data`](@ref). A
subtype implements [`estimate`](@ref)`(estimator, data; level)`.
"""
abstract type AbstractEstimator end

Base.broadcastable(e::AbstractEstimator) = Ref(e)

"""
    RiskRatio()

Vaccine efficacy as one minus the ratio of attack rates (events over
participants) in the vaccine and control arms over the whole follow-up, with a
Wald interval on the log risk ratio. When either arm has no events or only
events, 0.5 is added to each cell of the 2×2 table.
"""
struct RiskRatio <: AbstractEstimator end

"""
    CoxHazardRatio()

Vaccine efficacy as one minus the hazard ratio from a Cox proportional hazards
model with arm as the only covariate (fitted with Survival.jl), with a Wald
interval on the log hazard ratio. Gives `NaN` when either arm has no events,
where the partial likelihood has no finite maximum.
"""
struct CoxHazardRatio <: AbstractEstimator end

_z(level) = quantile(Normal(), (1 + level) / 2)

# Efficacy, interval and p-value from an estimate and standard error of a log
# ratio. A higher log ratio means a lower efficacy, so the bounds swap.
function _ve_from_log_ratio(logr, se, level)
    z = _z(level)
    return (ve = 1 - exp(logr), lower = 1 - exp(logr + z * se),
        upper = 1 - exp(logr - z * se), p_value = 2 * cdf(Normal(), -abs(logr / se)))
end

const _NO_ESTIMATE = (ve = NaN, lower = NaN, upper = NaN, p_value = NaN)

"""
    estimate(estimator, data; level = 0.95) -> NamedTuple

Estimate vaccine efficacy from [`trial_data`](@ref). Returns
`(ve, lower, upper, p_value)`: the point estimate, the bounds of a two-sided
`level` confidence interval, and the two-sided p-value against no effect.
"""
function estimate(::RiskRatio, data; level = 0.95)
    vac = data.arm .=== :vaccine
    a, n1 = count(data.event .& vac), count(vac)
    c, n0 = count(data.event .& .!vac), count(.!vac)
    (n1 == 0 || n0 == 0 || a + c == 0) && return _NO_ESTIMATE
    if a == 0 || c == 0 || a == n1 || c == n0
        a, c, n1, n0 = a + 0.5, c + 0.5, n1 + 1, n0 + 1
    end
    logr = log((a / n1) / (c / n0))
    se = sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return _ve_from_log_ratio(logr, se, level)
end

function estimate(::CoxHazardRatio, data; level = 0.95)
    vac = data.arm .=== :vaccine
    (any(data.event .& vac) && any(data.event .& .!vac)) || return _NO_ESTIMATE
    x = reshape(Float64.(vac), :, 1)
    fit = coxph(x, EventTime.(data.exit_time, data.event))
    return _ve_from_log_ratio(only(coef(fit)), only(stderror(fit)), level)
end
