module EpiHouseholds

using EpiBranch
using Distributions
using Random

# Accessor seam methods extended for `HouseholdProcess`; imported because they
# are part of EpiBranch's public extension API but not brought into scope by
# `using EpiBranch`. Everything else this package builds on (`TransmissionModel`,
# `Transition`, `Individual`, `linelist`, …) is exported by EpiBranch, and the
# population/progression helpers are called qualified. The deliberate reaches
# into EpiBranch internals are the shared continuous-time engine (the
# `_window_open`/`_window_close` helpers imported below and
# `EpiBranch._sellke_race!` used by the simulator) and the community-hazard
# helpers, reused rather than reimplemented so the household simulate and
# pairwise likelihood stay aligned with the shared engine.
import EpiBranch: new_state, add_individuals!, apply_observation!,
                  _simulate, SimOpts, _resolve_infectious_from,
                  _retry_for_condition, _reconcile_sellke_bookkeeping!,
                  _honours_termination_controls
# The pairwise likelihood reuses the infectious-window helpers to read each
# case's window from the same `from`/`until` states the simulator uses.
import EpiBranch: _window_open, _window_close
# The pairwise survival likelihood is structure-agnostic and lives in EpiBranch;
# this package adds its household methods and shares its community-hazard
# helpers, so the simulator and the likelihood agree on when that term is on.
import EpiBranch: pairwise_surv_loglik, _ext_active, _ext_survival

export HouseholdProcess, household_sizes
export HouseholdInfections, household_infections
export PairwiseSurvivalData, pairwise_surv_loglik
export HouseholdPairsLayout, compile_household_pairs

include("household_process.jl")
include("household_simulate.jl")
include("likelihood.jl")

end # module
