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
# The infectious-window helpers moved to EpiBranch alongside the shared
# `_sellke_race!` primitive; the pairwise likelihood reuses them to read each
# case's window from the same `from`/`until` states the simulator uses.
import EpiBranch: _window_open, _window_close

export HouseholdProcess, household_sizes
export HouseholdInfections, household_infections
export PairwiseSurvivalData, pairwise_surv_loglik

include("household_process.jl")
include("household_simulate.jl")
include("likelihood.jl")

end # module
