module EpiNetwork

using EpiBranch
using Distributions
using Random

# `NetworkProcess` builds on EpiBranch's continuous-time simulation surface:
# it reports the interventions/attributes/observation it carries, builds a
# population with `new_state`/`add_individuals!`, and runs its own `simulate`.
# `import` is needed for the methods we add to (the engine's extension
# surface); the rest are pulled in because they are not brought into scope by
# `using EpiBranch`. Everything here is part of EpiBranch's public extension
# API — the package reaches into no internals.
import EpiBranch:
                  interventions, attributes, observation, population_size,
                  new_state, add_individuals!,
                  simulate, apply_observation!

export NetworkProcess

include("network_process.jl")
include("network_simulate.jl")

end # module
