module EpiHouseholds

using EpiBranch
using Distributions
using Random

# Accessor seam methods extended for `HouseholdProcess`; imported because they
# are part of EpiBranch's public extension API but not brought into scope by
# `using EpiBranch`. Everything else this package builds on (`TransmissionModel`,
# `Transition`, `Individual`, `linelist`, …) is exported by EpiBranch, and the
# population/progression helpers are called qualified. The package also uses two
# sets of EpiBranch internals on purpose: the shared continuous-time engine (the
# `_shorthand_window` helper imported below and `EpiBranch._sellke_race!` used by
# the simulator) and the community-hazard helpers. Reusing them keeps the
# household simulator and pairwise likelihood consistent with the shared engine.
import EpiBranch: new_state, add_individuals!, apply_observation!,
                  _simulate, SimOpts, _resolve_infectious_from,
                  _retry_for_condition, _reconcile_sellke_bookkeeping!,
                  _honours_termination_controls
# The pairwise likelihood reads each case's infectious window through the same
# window the simulator's race builds, including its intervention removal.
import EpiBranch: _shorthand_window
# The pairwise survival likelihood works for any contact structure and lives in
# EpiBranch. This package adds household methods and uses EpiBranch's
# community-hazard helpers, so the simulator and the likelihood agree on when
# that term is on.
import EpiBranch: pairwise_surv_loglik, _ext_active, _ext_survival

export HouseholdProcess, household_sizes
export HouseholdInfections, household_infections
export PairwiseSurvivalData, pairwise_surv_loglik
export HouseholdPairsLayout, compile_household_pairs

include("household_process.jl")
include("household_simulate.jl")
include("likelihood.jl")

end # module
