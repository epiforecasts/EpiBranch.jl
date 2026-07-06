module EpiNetwork

using EpiBranch
using Distributions
using Random

# Seam methods extended for `NetworkProcess`, plus the helpers and types this
# model builds on. `import` is needed for the methods we add to (the engine's
# extension surface); the rest are pulled in because they are not brought into
# scope by `using EpiBranch`. Everything here is part of EpiBranch's public
# extension API — the package reaches into no internals.
import EpiBranch:
                  interventions, attributes, observation, population_size, n_types,
                  single_type_offspring, initialise_state, collect_exposures, contacts_of,
                  new_state, add_individuals!, seed!, get_generation_time,
                  transmission_time, transmission_risks, competing_risk,
                  NoGenerationTime,
                  simulate, apply_observation!

export NetworkProcess
export NetworkRateProcess

include("network_process.jl")
include("network_simulate.jl")
include("network_rate.jl")
include("network_rate_simulate.jl")

end # module
