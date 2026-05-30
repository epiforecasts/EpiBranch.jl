"""
Concrete observation models and the `Observed` wrapper. Extends
`simulate` (from `Engine`) and `chain_size_distribution`
(declared in `EpiBranchBase`) so that observation models compose with
the simulation and analytical paths through dispatch.
"""
module Observation

using Distributions
using Random
using ..EpiBranchBase
using ..Engine
import ..EpiBranchBase: chain_size_distribution, single_type_offspring,
                        population_size, latent_period, n_types
import ..Engine: simulate

# Internal helpers from Base
using ..EpiBranchBase: _sample_value

# Observation extension point — concrete observation models add methods
# to extract a scalar detection probability for closed-form analytics.
function scalar_detection_prob end

export PerCaseObservation, Observed
export ThinnedChainSize
export scalar_detection_prob

include("observation_models.jl")
include("observation.jl")

end
