module EpiNetwork

using EpiBranch
using Distributions
using Random

# `NetworkProcess` is a pure transmission kernel over EpiBranch's continuous-time
# simulation surface: it builds a population with `new_state`/`add_individuals!`
# and runs the shared `_simulate` seam, deriving the infectious window from the
# progression composed onto it with a `ModelSpec`. `import` is needed for the
# methods we add to (`population_size`, `_simulate`); the rest are pulled in
# because they are not brought into scope by `using EpiBranch`.
import EpiBranch: population_size, new_state, add_individuals!, apply_observation!,
                  _simulate, SimOpts, _resolve_infectious_from,
                  _retry_for_condition, _reconcile_sellke_bookkeeping!

export NetworkProcess

include("network_process.jl")
include("network_simulate.jl")

end # module
