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
# The pairwise likelihood reads each node's infectious window with the same
# helpers the simulator's race uses, and shares EpiBranch's test for whether a
# community hazard is on, so simulation and likelihood agree on both.
import EpiBranch: _window_open, _window_close, _ext_active

export NetworkProcess, RoutedNetwork
export NetworkInfections, network_infections
export pairwise_surv_loglik, compile_contact_pairs

include("network_process.jl")
include("routed_network.jl")
include("network_simulate.jl")
include("likelihood.jl")

end # module
