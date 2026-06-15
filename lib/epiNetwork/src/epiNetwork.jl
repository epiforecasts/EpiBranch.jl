module epiNetwork

using EpiBranch
using Distributions
using Random

# Seam methods extended for `NetworkProcess`, plus the helpers and types this
# model builds on. `import` is needed for the methods we add to (the engine's
# extension surface); the rest are pulled in because they are not brought into
# scope by `using EpiBranch`. `_intervention_vector` is an EpiBranch internal,
# imported rather than copied because the two are co-versioned in this monorepo.
import EpiBranch:
                  interventions, attributes, observation, population_size, n_types,
                  single_type_offspring, initialise_state, collect_exposures, contacts_of,
                  new_state, add_individuals!, seed!, get_generation_time,
                  transmission_time,
                  NoGenerationTime, _intervention_vector

export NetworkProcess

include("network_process.jl")
include("network_simulate.jl")

end # module
