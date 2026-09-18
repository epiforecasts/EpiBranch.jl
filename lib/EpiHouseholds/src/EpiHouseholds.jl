module EpiHouseholds

using EpiBranch
using Distributions
using QuadGK: quadgk
using Random

# Accessor seam methods extended for `HouseholdProcess`; imported because they
# are part of EpiBranch's public extension API but not brought into scope by
# `using EpiBranch`. Everything else this package builds on (`TransmissionModel`,
# `Transition`, `Individual`, `linelist`, …) is exported by EpiBranch, and the
# population/progression helpers are called qualified. The package also uses two
# sets of EpiBranch internals on purpose: the shared continuous-time engine
# (`EpiBranch._sellke_race!` used by the simulator, and the infection-layer reader
# imported below) and the community-hazard helpers. Reusing them keeps the
# household simulator and pairwise likelihood consistent with the shared engine.
import EpiBranch: new_state, add_individuals!, apply_observation!,
                  _simulate, SimOpts, _resolve_infectious_from,
                  _retry_for_condition, _reconcile_sellke_bookkeeping!,
                  _honours_termination_controls
# The infection layer is built and read out of a simulation by EpiBranch's
# helpers, which close each case's window where the simulator's race does.
import EpiBranch: _infection_layer_columns, _infection_layer_fields
# The pairwise survival likelihood works for any contact structure and lives in
# EpiBranch, which scores a household population through `contact_structure`.
# The simulator and the likelihood share EpiBranch's community-hazard helpers
# and agree on when that term applies and what it is.
import EpiBranch: pairwise_surv_loglik, _ext_active, _ext_draw, _valid_external,
                  _normalise_external

export HouseholdProcess, household_sizes
export HouseholdInfections, household_infections
export PairwiseSurvivalData, pairwise_surv_loglik
export HouseholdPairsLayout, compile_household_pairs
export HouseholdOffspring, household_offspring, household_offspring_law
export household_final_size

include("household_process.jl")
include("household_simulate.jl")
include("likelihood.jl")
include("offspring.jl")

end # module
