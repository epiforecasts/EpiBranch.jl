"""
    EpiTrial

Simulation-based evaluation of vaccine trials run during an outbreak. A trial
is an EpiBranch model with randomisation as an attribute and the trial vaccine
as an intervention; EpiTrial adds the step from a simulated outbreak to one row
per participant, estimators of vaccine efficacy, and summaries over replicate
trials (power, bias, coverage).
"""
module EpiTrial

using EpiBranch
using DataFrames: DataFrame, groupby, combine, nrow
using Distributions: Normal, quantile, cdf
# Distributions' `estimate(estimator, data)` has the shape wanted here, so the
# trial estimators add methods to it.
import Distributions: estimate
using QuadGK: quadgk
using Random: AbstractRNG, default_rng
using Statistics: mean, std
using Survival: coxph, coef, stderror, EventTime

# The vaccination seams a new vaccination type implements, and the helpers
# EpiBranch's own vaccinations use to record a dose and build their keyword
# constructors. Reusing them keeps a trial dose identical in effect to any
# other EpiBranch vaccination.
import EpiBranch: vaccine_effect, initialise_individual!, required_fields
import EpiBranch: AbstractVaccination, VaccineEffect, LeakyMode, AllOrNothingMode,
                  _record_vaccination!, _check_effect_keywords, _effect_getproperty,
                  _effect_propertynames, _show_keywords

export IndividualRandomisation, arm
export TrialVaccination
export Trial, AbstractEndpoint, Infection, trial_data
export AbstractEstimator, RiskRatio, CoxHazardRatio, estimate
export trial_estimates, operating_characteristics
export expected_ve

include("allocation.jl")
include("trial_vaccination.jl")
include("trial.jl")
include("estimators.jl")
include("operating_characteristics.jl")
include("analytical.jl")

end # module
