module EpiHouseholds

using EpiBranch
using Distributions
using Random
using SurvivalDistributions: hazard, cumhazard, loghazard

# Accessor seam methods extended for `HouseholdProcess`; imported because they
# are part of EpiBranch's public extension API but not brought into scope by
# `using EpiBranch`. Everything else this package builds on (`TransmissionModel`,
# `Transition`, `Individual`, `linelist`, …) is exported by EpiBranch, and the
# population/progression helpers are called qualified — so EpiHouseholds builds
# on the public surface only, with no reach into EpiBranch internals.
import EpiBranch: interventions, attributes, observation,
                  simulate, new_state, add_individuals!, resolve_transitions!,
                  apply_observation!

export HouseholdProcess, household_sizes
export HouseholdInfections, household_infections
export PairwiseSurvivalData, pairwise_surv_loglik
export HouseholdPairsLayout, compile_household_pairs

include("household_process.jl")
include("household_simulate.jl")
include("likelihood.jl")

end # module
