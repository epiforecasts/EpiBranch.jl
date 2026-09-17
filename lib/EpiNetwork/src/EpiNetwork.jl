module EpiNetwork

using EpiBranch
using Distributions
using Random

# `NetworkProcess` describes the transmission alone over EpiBranch's continuous-time
# simulation surface: it builds a population with `new_state`/`add_individuals!`
# and runs the shared `_simulate` seam, deriving the infectious window from the
# progression composed onto it with a `ModelSpec`. `import` is needed for the
# methods we add to (`population_size`, `_simulate`); the rest are pulled in
# because they are not brought into scope by `using EpiBranch`.
import EpiBranch: population_size, new_state, add_individuals!, apply_observation!,
                  _simulate, SimOpts, _resolve_infectious_from,
                  _retry_for_condition, _reconcile_sellke_bookkeeping!,
                  _honours_termination_controls
# The infection layer is built and read out of a simulation by EpiBranch's
# helpers, which close each node's window where the simulator's race does. The
# simulator and the likelihood also share EpiBranch's community-hazard helpers
# and agree on both.
import EpiBranch: _infection_layer_columns, _infection_layer_fields, _ext_active,
                  _ext_draw, _valid_external, _normalise_external

export NetworkProcess, RoutedNetwork
export NetworkInfections, network_infections
export pairwise_surv_loglik, compile_contact_pairs

include("network_process.jl")
include("routed_network.jl")
include("network_simulate.jl")
include("likelihood.jl")

end # module
